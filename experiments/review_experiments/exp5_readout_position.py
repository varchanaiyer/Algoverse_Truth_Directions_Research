"""
Experiment 5: separate the effect of the readout POSITION from the effect of the CONDITION.

Each condition is currently read at a different token:
  no-prompt                           last token of the statement
  no-prompt-chat-template             <think>\\n, after the template wrapped round the statement
  ablation-instructions-and-template  <think>\\n, after statement + instructions + template
  ablation-instructions-only          last token of the instructions
  ablation-filler-token               last '.' of the filler block
  sentence-based-CoT                  token before the verdict

So when A1 goes 0.933 (no-prompt) -> 0.619 (chat template) -> 0.51 (instructions), it is
not clear whether the instructions hurt or the readout simply moved off the statement.
This reads every condition at the same anchors:

  statement_end  last token of the statement (or the token that contains its final
                 character, when the tokenizer merges it with what follows, e.g. '.\\n\\n')
  prompt_end     end of the user turn (<think>\\n for templated conditions; the full
                 input for conditions without a template, where there is no separate turn)
  final          the current readout (should reproduce results_database.csv)

The model is causal, so the activation at statement_end only depends on tokens up to the
statement: every condition that shares the same prefix there gives the SAME activation.
That is a built-in check, and the summary prints it. The informative comparison is at
prompt_end, where conditions differ only in what sits between statement and readout.

Usage:
  python experiments/review_experiments/exp5_readout_position.py --tasks A1 A2 A3 F3 F4 F5 \
      --layers 0-32:2 --cache-dir /content/drive/MyDrive/dsab_cache
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CONDITIONS = [
    "no-prompt",
    "no-prompt-chat-template",
    "ablation-instructions-only",
    "ablation-instructions-and-template",
    "ablation-filler-token",
    "sentence-based-CoT",
]
THINK_ID = 128013  # <think>
NEWLINE_ID = 198   # \n


def statement_end_index(ids, statement, tokenizer):
    """Index of the token holding the statement's last character: the shortest prefix whose
    decoded text contains the whole statement. Binary search on prefix length."""
    lo, hi = 1, len(ids)
    if statement not in tokenizer.decode(ids):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if statement in tokenizer.decode(ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


def prompt_end_index(ids, prompt_len=None):
    """Last prompt token: '<think>\\n' when present, else prompt_len - 1 for CoT rows, else
    the last token (instructions-only and plain have no template, so the whole input is
    the prompt)."""
    for i in range(len(ids) - 1):
        if ids[i] == THINK_ID and ids[i + 1] == NEWLINE_ID:
            return i + 1
    if prompt_len is not None:
        return int(prompt_len) - 1
    return len(ids) - 1


def cut_ids(df, readout, tokenizer):
    ids = C.ids_for(df, tokenizer)
    out, misses = [], 0
    for k, row_ids in enumerate(ids):
        if readout == "final":
            out.append(row_ids)
            continue
        if readout == "statement_end":
            idx = statement_end_index(row_ids, C.statement_text(df.iloc[k]), tokenizer)
            if idx is None:
                misses += 1
                idx = len(row_ids) - 1
        else:
            idx = prompt_end_index(row_ids, df["prompt_len"].iloc[k] if "prompt_len" in df.columns else None)
        out.append(row_ids[: idx + 1])
    if misses:
        print(f"    warning: statement not found in {misses}/{len(ids)} rows; read at final token for those")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--readouts", nargs="+", default=["statement_end", "prompt_end", "final"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    layers = C.parse_layers(args.layers, model.config.num_hidden_layers + 1)

    for task in args.tasks:
        for cond in args.conditions:
            tr, te = C.load_split(cond, task, "train"), C.load_split(cond, task, "test")
            for readout in args.readouts:
                if C.already_done("exp5_readout_position", train_task=task, train_condition=cond, readout=readout):
                    print(f"  {task} {cond} {readout}: already done, skipping")
                    continue
                acts = {}
                for split, df in (("train", tr), ("test", te)):
                    ids = cut_ids(df, readout, tokenizer)
                    acts[split] = C.cached_activations(
                        args.cache_dir, f"{C.DEEPSEEK_SHORT}__{cond}__{task}__{split}__{readout}",
                        lambda ids=ids, split=split: C.extract_all_layer_activations(
                            ids, model, C.DEEPSEEK_PAD_ID, args.batch_size, f"{task} {cond} {split} {readout}"))
                rows = []
                for seed in args.seeds:
                    for layer in layers:
                        w, m = C.fit_linear_probe(acts["train"][layer], tr["y"].to_numpy(), seed)
                        auc, lo, hi = C.auroc_with_ci(te["y"], C.probe_scores(w, m, acts["test"][layer]), seed=seed)
                        rows.append(C.result_row(
                            experiment="exp5_readout_position", model=C.DEEPSEEK_SHORT,
                            train_task=task, test_task=task, train_condition=cond, test_condition=cond,
                            readout=readout, layer=layer, seed=seed, eval_target="label", subset="all",
                            n_train=len(tr), n_test=len(te), auroc=auc, ci_low=lo, ci_high=hi))
                C.append_results(rows)
                best = max(rows, key=lambda r: r["auroc"])
                print(f"  {task} {cond:<36} {readout:<14} best {best['auroc']:.3f} (layer {best['layer']})")
        summarize(task)


def summarize(task):
    df = pd.read_csv(C.RESULTS_CSV)
    df = df[(df["experiment"] == "exp5_readout_position") & (df["train_task"] == task)]
    table = (df.groupby(["train_condition", "readout", "layer"])["auroc"].mean()
               .groupby(["train_condition", "readout"]).max().unstack("readout"))
    print(f"\n{task}: best-layer in-domain AUROC by condition x readout position")
    print(table.round(3).to_string())
    table.to_csv(C.RESULTS_DIR / f"exp5_{task}_table.csv")


if __name__ == "__main__":
    main()
