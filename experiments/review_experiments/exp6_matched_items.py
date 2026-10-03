"""
Experiment 6: compare plain and CoT on the SAME statements.

The CoT datasets lost items during filtering (generation cut-offs, verdict-extraction
failures, lexical cleaning), e.g. A2 train 700 -> 673, F5 test 593 -> 562. The plain
condition still uses every item. So "plain in-domain AUROC vs CoT in-domain AUROC" in
results_database.csv compares two different item sets, and the dropped items are not
random: they skew towards traces the model struggled with.

Every CoT statement is in the plain split of the same name (checked: no CoT train item
is in plain test or vice versa), so plain rows are matched to CoT rows by statement text.

Step 1 (CPU): write datasets/matched_items/{task}_{split}.csv with the plain statement
and label for exactly the CoT items, in CoT row order, and print what was dropped.
Step 2 (GPU): rerun in-domain (plain, CoT) and cross-condition (plain <-> CoT) probes on
the matched items only, with seeds and bootstrap CIs, next to the full-plain numbers.

Usage:
  python experiments/review_experiments/exp6_matched_items.py --prepare-only
  python experiments/review_experiments/exp6_matched_items.py --layers all \
      --cache-dir /content/drive/MyDrive/dsab_cache
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COT = "sentence-based-CoT"
PLAIN = "no-prompt"
MATCHED = "no-prompt-matched"
MATCHED_DIR = C.DATA_DIR / "matched_items"


def prepare(tasks):
    MATCHED_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for task in tasks:
        for split in ["train", "test"]:
            # Full plain file, never subsampled: it is the lookup table for the CoT rows.
            plain = pd.read_csv(C.CONDITION_DICT[PLAIN] / f"{task}_{split}.csv")
            plain["y"] = C.to_binary_labels(plain["label"])
            cot = C.load_split(COT, task, split)
            plain["key"] = plain["statement"].str.strip()
            cot_keys = [C.statement_text(r) for _, r in cot.iterrows()]
            first = plain.drop_duplicates("key").set_index("key")
            missing = [k for k in cot_keys if k not in first.index]
            if missing:
                raise ValueError(f"{task} {split}: {len(missing)} CoT statements not in plain, e.g. {missing[0]!r}")
            matched = first.loc[cot_keys].reset_index()
            label_clash = int((matched["y"].to_numpy() != cot["y"].to_numpy()).sum())
            matched[["key", "label"]].rename(columns={"key": "statement"}).to_csv(
                MATCHED_DIR / f"{task}_{split}.csv", index=False)

            dropped = plain[~plain["key"].isin(set(cot_keys))]
            report.append({
                "task": task, "split": split, "plain": len(plain), "cot": len(cot),
                "dropped": len(dropped),
                "dropped_true_share": round(float(dropped["y"].mean()), 3) if len(dropped) else float("nan"),
                "kept_true_share": round(float(cot["y"].mean()), 3),
                "label_mismatches": label_clash,
                "duplicate_statements_in_plain": int(plain["key"].duplicated().sum()),
            })
    report = pd.DataFrame(report)
    print(report.to_string(index=False))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(C.RESULTS_DIR / "exp6_dropped_items.csv", index=False)


def load_matched(task, split):
    df = pd.read_csv(MATCHED_DIR / f"{task}_{split}.csv")
    df["y"] = C.to_binary_labels(df["label"])
    return df


def run(tasks, layers_arg, seeds, cache_dir, batch_size):
    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    layers = C.parse_layers(layers_arg, model.config.num_hidden_layers + 1)

    def acts(cond, task, split, df):
        return C.cached_activations(
            cache_dir, f"{C.DEEPSEEK_SHORT}__{cond}__{task}__{split}__final",
            lambda: C.extract_all_layer_activations(C.ids_for(df, tokenizer), model, C.DEEPSEEK_PAD_ID,
                                                    batch_size, f"{task} {cond} {split}"))

    for task in tasks:
        data = {
            PLAIN: (C.load_split(PLAIN, task, "train"), C.load_split(PLAIN, task, "test")),
            MATCHED: (load_matched(task, "train"), load_matched(task, "test")),
            COT: (C.load_split(COT, task, "train"), C.load_split(COT, task, "test")),
        }
        A = {c: (acts(c, task, "train", tr), acts(c, task, "test", te)) for c, (tr, te) in data.items()}
        pairs = [(PLAIN, PLAIN), (MATCHED, MATCHED), (COT, COT), (MATCHED, COT), (COT, MATCHED)]
        rows = []
        for seed in seeds:
            for layer in layers:
                for tr_c, te_c in pairs:
                    y_tr = data[tr_c][0]["y"].to_numpy()
                    y_te = data[te_c][1]["y"].to_numpy()
                    w, m = C.fit_linear_probe(A[tr_c][0][layer], y_tr, seed)
                    auc, lo, hi = C.auroc_with_ci(y_te, C.probe_scores(w, m, A[te_c][1][layer]), seed=seed)
                    rows.append(C.result_row(
                        experiment="exp6_matched_items", model=C.DEEPSEEK_SHORT, train_task=task, test_task=task,
                        train_condition=tr_c, test_condition=te_c, readout="final", layer=layer, seed=seed,
                        eval_target="label", subset="all", n_train=len(y_tr), n_test=len(y_te),
                        auroc=auc, ci_low=lo, ci_high=hi))
        C.append_results(rows)
        df = pd.DataFrame(rows)
        best = df.groupby(["train_condition", "test_condition", "layer"])["auroc"].mean() \
                 .groupby(["train_condition", "test_condition"]).max()
        print(f"\n{task}: best-layer AUROC")
        print(best.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    prepare(args.tasks)
    if not args.prepare_only:
        run(args.tasks, args.layers, args.seeds, args.cache_dir, args.batch_size)


if __name__ == "__main__":
    main()
