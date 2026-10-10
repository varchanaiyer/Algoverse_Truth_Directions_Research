"""
Experiment 10: the model's answer with NO reasoning, item by item, next to the plain probe.

exp3 showed that on items where the model's post-reasoning answer is wrong, the plain probe
(read on the bare statement) still follows the true label (0.92 on F0). exp3 never asked
the model for an answer without reasoning, so it cannot say whether that probe reads the
model's own pre-reasoning belief or something the model never outputs. This collects that
answer and splits the items accordingly.

The no-reasoning answer: the statement goes in DeepSeek's chat template with Bora's
instruction from check_model_accuracy.ipynb, and the assistant turn is pre-filled with an empty think block and the answer
prefix (<think>\\n\\n</think>\\n\\nAnswer:), the usual way to switch reasoning off on the
R1 distills. One forward pass, then the next-token logits for " Yes" vs " No" give p_yes.
No generation. `yes_no_mass` (the probability on those two tokens) checks that the model is
answering in the asked format; a low value means the prompt is not working.

Small models answering without reasoning lean hard towards Yes (the 1.5B distill said Yes to
98% of F0 statements in a local test), and a raw argmax answer then mostly measures that
lean. So the script reports two answers: `direct_raw` (argmax) and `direct` (calibrated:
Yes for the items with the highest p_yes, as many as there are true statements in the task),
plus the AUROC of p_yes against the label, which no lean can change. The item splits use the
calibrated answer.

Per task the script then trains the plain probe exactly as exp3 does (same pooled items,
same 5 folds, same seed, layer chosen on all items) and reports, against the TRUE label:
  direct_wrong                plain probe on items the no-reasoning answer gets wrong
                              (> 0.5: probe follows the truth past the model's quick answer)
  cot_wrong & direct_right    items reasoning got wrong but the quick answer got right
  cot_wrong & direct_wrong    items both answers got wrong (> 0.5 here means the probe reads
                              something neither answer shows)
plus the no-reasoning accuracy per task and how often it agrees with the post-reasoning answer.

Usage:
  python experiments/review_experiments/exp10_direct_answer.py --tasks F0 F1 F2 F3 F4 F5 \
      --cache-dir /content/dsab_cache/exp10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import exp3_probe_vs_verdict as E3  # noqa: E402

# Bora's EVAL_INSTRUCTIONS from validation_code/check_model_accuracy.ipynb, word for word,
# so DeepSeek's no-reasoning accuracy sits directly next to his Llama-3.1-8B numbers.
INSTRUCTION = (
    "Evaluate the statement for whether it is true or false. "
    "write exactly one line and nothing after it: "
    "\nAnswer: Yes or Answer: No"
)
NO_THINK_PREFILL = "\n</think>\n\nAnswer:"  # the template already ends with <think>\n


def direct_prompt_ids(statements, tokenizer):
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": f"{s}\n\n{INSTRUCTION}"}],
                                      add_generation_prompt=True, tokenize=False) + NO_THINK_PREFILL
        for s in statements
    ]
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def direct_answers(id_lists, model, pad_id, yes_id, no_id, batch_size=16):
    """p(" Yes") and p(" No") at the next position, one forward pass per batch."""
    import torch
    from tqdm.auto import tqdm

    device = next(model.parameters()).device
    order = np.argsort([len(x) for x in id_lists])
    p_yes, p_no = np.zeros(len(id_lists)), np.zeros(len(id_lists))
    for i in tqdm(range(0, len(order), batch_size), desc="no-reasoning answers", leave=False):
        idx = order[i:i + batch_size]
        batch = [torch.tensor(id_lists[j]) for j in idx]
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id,
                                                 padding_side="left").to(device)
        lens = torch.tensor([len(t) for t in batch], device=device)
        pos = torch.arange(padded.shape[1], device=device).unsqueeze(0)
        mask = (pos >= (padded.shape[1] - lens).unsqueeze(1)).long()
        with torch.no_grad():
            probs = model(input_ids=padded, attention_mask=mask).logits[:, -1, :].float().softmax(-1)
        p_yes[idx] = probs[:, yes_id].cpu().numpy()
        p_no[idx] = probs[:, no_id].cpu().numpy()
    return p_yes, p_no


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.FACT_TASKS)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seed", type=int, default=0, help="same fold seed as exp3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not all((E3.VERDICT_DIR / f"{t}.csv").exists() for t in args.tasks):
        E3.build_verdicts(args.tasks)

    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    layers = C.parse_layers(args.layers, model.config.num_hidden_layers + 1)
    out_dir = C.RESULTS_DIR / "exp10_direct_answers"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for task in args.tasks:
        if C.already_done("exp10_direct_answer", train_task=task):
            print(f"\n{task}: already done, skipping")
            continue
        v = pd.read_csv(E3.VERDICT_DIR / f"{task}.csv")
        v = v[v["model_verdict"].notna()].reset_index(drop=True)
        y = v["label"].to_numpy().astype(int)
        cot = v["model_verdict"].to_numpy().astype(int)

        # 1. no-reasoning answers
        p_yes, p_no = direct_answers(direct_prompt_ids(v["statement"].tolist(), tokenizer),
                                     model, C.DEEPSEEK_PAD_ID, yes_id, no_id)
        p_rel = p_yes / (p_yes + p_no)
        direct_raw = (p_yes > p_no).astype(int)
        # calibrated: as many Yes answers as there are true statements
        n_true = int(y.sum())
        direct = np.zeros(len(v), dtype=int)
        direct[np.argsort(-p_rel, kind="stable")[:n_true]] = 1
        direct_auc, d_lo, d_hi = C.auroc_with_ci(y, p_rel, seed=args.seed)
        items = v[["split", "statement", "label", "model_verdict"]].rename(columns={"model_verdict": "cot_answer"})
        items["direct_raw"], items["direct_answer"], items["p_yes"], items["yes_no_mass"] = direct_raw, direct, p_rel, p_yes + p_no
        items.to_csv(out_dir / f"{task}.csv", index=False)

        # 2. plain probe, out-of-fold, exactly as exp3
        plain_ids = E3.plain_rows_for(v["statement"].tolist(), task, tokenizer)
        acts = C.cached_activations(
            args.cache_dir, f"{C.DEEPSEEK_SHORT}__no-prompt-matched__{task}__all__final",
            lambda: C.extract_all_layer_activations(plain_ids, model, C.DEEPSEEK_PAD_ID, args.batch_size, f"{task} plain"))
        groups = pd.factorize(v["statement"])[0]
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_of = np.empty(len(v), dtype=int)
        for k, (_, te) in enumerate(splitter.split(np.zeros(len(v)), y * 2 + cot, groups)):
            fold_of[te] = k

        subsets = {
            "all": np.ones(len(v), bool),
            "direct_wrong": direct != y,
            "cot_wrong": cot != y,
            "cot_wrong&direct_right": (cot != y) & (direct == y),
            "cot_wrong&direct_wrong": (cot != y) & (direct != y),
        }
        rows = []
        for layer in layers:
            scores = np.zeros(len(v))
            for k in range(args.folds):
                tr, te = fold_of != k, fold_of == k
                w, m = C.fit_linear_probe(acts[layer][tr], y[tr], args.seed)
                scores[te] = C.probe_scores(w, m, acts[layer][te])
            for name, mask in subsets.items():
                auc, lo, hi = C.auroc_with_ci(y[mask], scores[mask], seed=args.seed)
                rows.append(C.result_row(
                    experiment="exp10_direct_answer", model=C.DEEPSEEK_SHORT, train_task=task, test_task=task,
                    train_condition="no-prompt[label]", test_condition="no-prompt", readout="final", layer=layer,
                    seed=args.seed, eval_target="label", subset=name,
                    n_train=int(round(len(v) * (args.folds - 1) / args.folds)), n_test=int(mask.sum()),
                    auroc=auc, ci_low=lo, ci_high=hi))
        C.append_results(rows)

        df = pd.DataFrame(rows)
        layer = df[df["subset"] == "all"].set_index("layer")["auroc"].idxmax()
        at = df[df["layer"] == layer].set_index("subset")
        s = {
            "task": task, "items": len(v),
            "no_reasoning_yes_rate_raw": round(float(direct_raw.mean()), 3),
            "no_reasoning_accuracy_raw": round(float((direct_raw == y).mean()), 3),
            "no_reasoning_accuracy_calibrated": round(float((direct == y).mean()), 3),
            "no_reasoning_p_yes_AUROC": f"{direct_auc:.3f} [{d_lo:.2f}, {d_hi:.2f}]",
            "with_reasoning_accuracy": round(float((cot == y).mean()), 3),
            "answers_agree": round(float((direct == cot).mean()), 3),
            "median_yes_no_mass": round(float(np.median(p_yes + p_no)), 3),
            "probe_layer": int(layer),
        }
        for name in subsets:
            r = at.loc[name]
            s[f"{name} n"] = int(r.n_test)
            s[f"{name} AUROC"] = f"{r.auroc:.3f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" if r.auroc == r.auroc else "n/a"
        summary.append(s)
        print(pd.Series(s).to_string(), "\n")

    if summary:
        path = C.RESULTS_DIR / "exp10_summary.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        new = pd.concat([old, pd.DataFrame(summary)]).drop_duplicates("task", keep="last")
        new.to_csv(path, index=False)
        print(new.T.to_string())


if __name__ == "__main__":
    main()
