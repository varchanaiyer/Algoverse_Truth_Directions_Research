"""
Experiment 4: when, during the reasoning, does truth become readable?

The sentence-based CoT prefixes are read at one point only: the token before the verdict.
For arithmetic that is after the model has already written the answer in words
("so (13 * 6) - 37 equals 41. Wait a minute, the right side is 37"), which is why A2/A3
reach AUROC 1.000. This reads the same traces at fractions of the way through the
reasoning, so the paper can show the curve instead of just the end point.

Fractions are of the reasoning tokens (prompt_len .. end of extracted prefix):
  0.00  end of the prompt (<think>\\n). Same input as ablation-instructions-and-template.
  0.25, 0.50, 0.75
  1.00  the current pre-verdict readout. Should reproduce results_database.csv.

For each fraction:
  - in-domain probe (train/test at that fraction), with bootstrap CI
  - plain -> fraction transfer (plain-statement probe scored at that fraction): does the
    no-CoT truth direction start to fire partway through, or only at the end?
  - TF-IDF bag-of-words on the decoded prefix text at that fraction (CPU, same settings
    as lexical_shortcut_baseline.py). Where the probe climbs but TF-IDF does not, the
    probe is reading more than the words on the page.

Usage:
  python experiments/review_experiments/exp4_early_readout.py --tasks A1 A2 A3 F3 F4 F5 \
      --fractions 0 0.25 0.5 0.75 1 --layers 0-32:2 --cache-dir /content/drive/MyDrive/dsab_cache
  python experiments/review_experiments/exp4_early_readout.py --tfidf-only   # CPU only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COT = "sentence-based-CoT"
PLAIN = "no-prompt"


def truncate(df, frac):
    """Id lists cut at `frac` of the way through the reasoning part of each prefix."""
    ids = C.ids_for(df)
    out = []
    for row_ids, prompt_len in zip(ids, df["prompt_len"].astype(int)):
        n_reason = len(row_ids) - prompt_len
        out.append(row_ids[: prompt_len + int(round(frac * n_reason))])
    return out


def tfidf_auroc(train_texts, y_tr, test_texts, y_te, seed):
    """Mirrors evaluate_tfidf_bow in lexical_shortcut_baseline.py (unigrams, lowercase,
    LogisticRegression). Only the reasoning text is used: the prompt is identical across
    items apart from the statement, so including it adds the statement's own words."""
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), token_pattern=r"(?u)\b[A-Za-z0-9']+\b")
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    if Xtr.shape[1] == 0:
        return float("nan"), float("nan"), float("nan")
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, y_tr)
    return C.auroc_with_ci(y_te, clf.predict_proba(Xte)[:, 1], seed=seed)


def run_tfidf(tasks, fractions, seeds, tokenizer):
    rows = []
    for task in tasks:
        tr, te = C.load_split(COT, task, "train"), C.load_split(COT, task, "test")
        for frac in fractions:
            if C.already_done("exp4_early_readout", model="tfidf-bow", train_task=task, readout=f"frac={frac:.2f}"):
                continue
            texts = []
            for df in (tr, te):
                ids = truncate(df, frac)
                texts.append([tokenizer.decode(i[p:]) for i, p in zip(ids, df["prompt_len"].astype(int))])
            for seed in seeds:
                if frac == 0:
                    auc = lo = hi = float("nan")  # no reasoning text yet
                else:
                    auc, lo, hi = tfidf_auroc(texts[0], tr["y"], texts[1], te["y"], seed)
                rows.append(C.result_row(
                    experiment="exp4_early_readout", model="tfidf-bow", train_task=task, test_task=task,
                    train_condition=COT, test_condition=COT, readout=f"frac={frac:.2f}", layer=-1, seed=seed,
                    eval_target="label", subset="all", n_train=len(tr), n_test=len(te),
                    auroc=auc, ci_low=lo, ci_high=hi))
            print(f"  {task} frac={frac:.2f} tfidf AUROC {rows[-1]['auroc']:.3f}")
            C.append_results(rows)
            rows = []


def run_probes(tasks, fractions, layers_arg, seeds, cache_dir, batch_size, model, tokenizer):
    n_layers = model.config.num_hidden_layers + 1
    layers = C.parse_layers(layers_arg, n_layers)
    for task in tasks:
        tr, te = C.load_split(COT, task, "train"), C.load_split(COT, task, "test")
        ptr = C.load_split(PLAIN, task, "train")
        plain_tr = C.cached_activations(
            cache_dir, f"{C.DEEPSEEK_SHORT}__{PLAIN}__{task}__train__final",
            lambda: C.extract_all_layer_activations(C.ids_for(ptr, tokenizer), model, C.DEEPSEEK_PAD_ID,
                                                    batch_size, f"{task} plain train"))
        for frac in fractions:
            if C.already_done("exp4_early_readout", model=C.DEEPSEEK_SHORT, train_task=task,
                              readout=f"frac={frac:.2f}"):
                print(f"  {task} frac={frac:.2f}: already done, skipping")
                continue
            acts = {}
            for split, df in (("train", tr), ("test", te)):
                ids = truncate(df, frac)
                acts[split] = C.cached_activations(
                    cache_dir, f"{C.DEEPSEEK_SHORT}__{COT}__{task}__{split}__frac{frac:.2f}",
                    lambda ids=ids, split=split: C.extract_all_layer_activations(
                        ids, model, C.DEEPSEEK_PAD_ID, batch_size, f"{task} {split} frac={frac:.2f}"))
            rows = []
            for seed in seeds:
                for layer in layers:
                    w, m = C.fit_linear_probe(acts["train"][layer], tr["y"].to_numpy(), seed)
                    auc, lo, hi = C.auroc_with_ci(te["y"], C.probe_scores(w, m, acts["test"][layer]), seed=seed)
                    rows.append(C.result_row(
                        experiment="exp4_early_readout", model=C.DEEPSEEK_SHORT, train_task=task, test_task=task,
                        train_condition=COT, test_condition=COT, readout=f"frac={frac:.2f}", layer=layer,
                        seed=seed, eval_target="label", subset="all", n_train=len(tr), n_test=len(te),
                        auroc=auc, ci_low=lo, ci_high=hi))
                    w, m = C.fit_linear_probe(plain_tr[layer], ptr["y"].to_numpy(), seed)
                    auc, lo, hi = C.auroc_with_ci(te["y"], C.probe_scores(w, m, acts["test"][layer]), seed=seed)
                    rows.append(C.result_row(
                        experiment="exp4_early_readout", model=C.DEEPSEEK_SHORT, train_task=task, test_task=task,
                        train_condition=PLAIN, test_condition=COT, readout=f"frac={frac:.2f}", layer=layer,
                        seed=seed, eval_target="label", subset="all", n_train=len(ptr), n_test=len(te),
                        auroc=auc, ci_low=lo, ci_high=hi))
            C.append_results(rows)
            best = max((r for r in rows if r["train_condition"] == COT), key=lambda r: r["auroc"])
            print(f"  {task} frac={frac:.2f}: best in-domain {best['auroc']:.3f} at layer {best['layer']}")
    plot(tasks)


def plot(tasks):
    import matplotlib.pyplot as plt

    df = pd.read_csv(C.RESULTS_CSV)
    df = df[df["experiment"] == "exp4_early_readout"].copy()
    df["frac"] = df["readout"].str.replace("frac=", "").astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    series = [
        ("In-domain probe (best layer)", (df["model"] == C.DEEPSEEK_SHORT) & (df["train_condition"] == COT)),
        ("Plain probe -> CoT (best layer)", (df["model"] == C.DEEPSEEK_SHORT) & (df["train_condition"] == PLAIN)),
        ("TF-IDF on reasoning text", df["model"] == "tfidf-bow"),
    ]
    for ax, (title, mask) in zip(axes, series):
        sub = df[mask]
        for task in tasks:
            t = sub[sub["train_task"] == task].groupby(["frac", "layer"])["auroc"].mean().groupby("frac").max()
            if len(t):
                ax.plot(t.index, t.values, marker="o", label=task)
        ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Fraction of reasoning read")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("AUROC")
    axes[-1].legend(fontsize=8)
    out = C.RESULTS_DIR / "exp4_early_readout.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tfidf-only", action="store_true", help="CPU only: skip the model")
    args = ap.parse_args()

    if args.tfidf_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(C.DEEPSEEK_ID)
        run_tfidf(args.tasks, args.fractions, args.seeds, tokenizer)
        plot(args.tasks)
        return
    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    run_tfidf(args.tasks, args.fractions, args.seeds, tokenizer)
    run_probes(args.tasks, args.fractions, args.layers, args.seeds, args.cache_dir, args.batch_size, model, tokenizer)


if __name__ == "__main__":
    main()
