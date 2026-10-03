"""
Experiment 7: give the Granite run something to compare against.

results_database.csv has Granite (granite-4.2-8b) rows for sentence-based-CoT only, so
Granite cannot support a "CoT changes the truth direction" claim on its own. This adds
the plain conditions and the cross-condition transfer on Granite.

How the existing Granite run built its inputs (Extract_Layers_Granite_Ablation.ipynb):
it DECODES DeepSeek's token ids back to text and re-tokenizes with Granite's tokenizer.
So the "Granite CoT" condition is DeepSeek's reasoning trace read by Granite, including
DeepSeek's chat markers ('<｜User｜>', '<｜Assistant｜>', '<think>') as plain text. That
is a fair test of "does externalized reasoning help a different model", and the paper
should describe it that way. This script keeps that input exactly (so its CoT numbers
reproduce the existing rows) and adds:

  no-prompt                       the bare statement, Granite tokenizer
  no-prompt-granite-template      the statement inside Granite's OWN chat template
                                  (apply_chat_template, add_generation_prompt=True)
  sentence-based-CoT              unchanged DeepSeek-trace input (as the existing run)
  sentence-based-CoT-clean        same trace with the DeepSeek marker strings removed,
                                  to check the markers are not driving the result

Cross-condition: no-prompt <-> sentence-based-CoT in both directions, as in Figure 3.

Usage (A100/L4 class GPU; Granite 8B in fp16 needs ~17 GB):
  python experiments/review_experiments/exp7_granite_conditions.py --layers 0-40:2 \
      --cache-dir /content/drive/MyDrive/dsab_cache
"""

import argparse
import sys
from ast import literal_eval
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GRANITE_ID = "ibm-granite/granite-4.2-8b"
GRANITE_SHORT = "granite-4.2-8b"
DEEPSEEK_MARKER_STRINGS = ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"]


def granite_inputs(condition, df, g_tok, ds_tok):
    """Granite token ids for each row under `condition`."""
    if condition == "no-prompt":
        return g_tok(df["statement"].tolist())["input_ids"]
    if condition == "no-prompt-granite-template":
        # Render to text, then tokenize: apply_chat_template(tokenize=True) returns a dict
        # (input_ids, attention_mask) in recent transformers, not a list of ids.
        texts = [
            g_tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True, tokenize=False)
            for s in df["statement"]
        ]
        return g_tok(texts, add_special_tokens=False)["input_ids"]
    # CoT: decode DeepSeek ids to text, as the existing Granite notebook does.
    texts = [ds_tok.decode(literal_eval(x) if isinstance(x, str) else x) for x in df["extracted_statement_ids"]]
    if condition == "sentence-based-CoT-clean":
        for marker in DEEPSEEK_MARKER_STRINGS:
            texts = [t.replace(marker, "") for t in texts]
        texts = [t.replace("<think>\n", "") for t in texts]
    return g_tok(texts)["input_ids"]


def source_condition(condition):
    return {"no-prompt": "no-prompt", "no-prompt-granite-template": "no-prompt",
            "sentence-based-CoT": "sentence-based-CoT", "sentence-based-CoT-clean": "sentence-based-CoT"}[condition]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS)
    ap.add_argument("--conditions", nargs="+",
                    default=["no-prompt", "no-prompt-granite-template", "sentence-based-CoT", "sentence-based-CoT-clean"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    ds_tok = AutoTokenizer.from_pretrained(C.DEEPSEEK_ID)
    model, g_tok = C.load_model(GRANITE_ID)
    pad_id = g_tok.pad_token_id
    layers = C.parse_layers(args.layers, model.config.num_hidden_layers + 1)

    for task in args.tasks:
        if C.already_done("exp7_granite_conditions", train_task=task):
            print(f"\n{task}: already done, skipping")
            continue
        data, acts = {}, {}
        for cond in args.conditions:
            tr = C.load_split(source_condition(cond), task, "train")
            te = C.load_split(source_condition(cond), task, "test")
            data[cond] = (tr, te)
            acts[cond] = tuple(
                C.cached_activations(
                    args.cache_dir, f"{GRANITE_SHORT}__{cond}__{task}__{split}__final",
                    lambda df=df, split=split, cond=cond: C.extract_all_layer_activations(
                        granite_inputs(cond, df, g_tok, ds_tok), model, pad_id, args.batch_size,
                        f"{task} {cond} {split}"))
                for split, df in (("train", tr), ("test", te)))

        pairs = [(c, c) for c in args.conditions]
        if {"no-prompt", "sentence-based-CoT"} <= set(args.conditions):
            pairs += [("no-prompt", "sentence-based-CoT"), ("sentence-based-CoT", "no-prompt")]
        rows = []
        for seed in args.seeds:
            for layer in layers:
                for tr_c, te_c in pairs:
                    y_tr = data[tr_c][0]["y"].to_numpy()
                    y_te = data[te_c][1]["y"].to_numpy()
                    w, m = C.fit_linear_probe(acts[tr_c][0][layer], y_tr, seed)
                    auc, lo, hi = C.auroc_with_ci(y_te, C.probe_scores(w, m, acts[te_c][1][layer]), seed=seed)
                    rows.append(C.result_row(
                        experiment="exp7_granite_conditions", model=GRANITE_SHORT, train_task=task, test_task=task,
                        train_condition=tr_c, test_condition=te_c, readout="final", layer=layer, seed=seed,
                        eval_target="label", subset="all", n_train=len(y_tr), n_test=len(y_te),
                        auroc=auc, ci_low=lo, ci_high=hi))
        C.append_results(rows)
        best = pd.DataFrame(rows).groupby(["train_condition", "test_condition", "layer"])["auroc"].mean() \
                 .groupby(["train_condition", "test_condition"]).max()
        print(f"\n{task} (Granite): best-layer AUROC")
        print(best.round(3).to_string())
        del acts


if __name__ == "__main__":
    main()
