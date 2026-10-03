"""
Experiment 8: shrink cosine_similarity_mahalanobis.csv and add cross-condition cosines.

The file is 1.4 GB because every row carries two full 4096-float probe weight vectors as
space-separated text (weight_i, weight_j) next to one cosine number. It is also plain
cosine similarity: nothing in cosine_similarity_utils.py whitens by a covariance, so the
"mahalanobis" in the name is wrong and should not reach the paper.

The stored cosine column does NOT match its own weight columns: recomputing cosine from
weight_i/weight_j disagrees with cosine_similarity in ~84% of rows, sometimes with the
sign flipped (stored 0.955, recomputed -0.970). The cosine column was committed on 9/8;
the weight columns were attached on 9/9 by Compute_Probe_Weights.ipynb, which retrains
the probes, and the code that produced the original cosine column is not in the repo. So
every cosine below is recomputed from the stored vectors (one consistent set of probes),
and the old column is kept only as cosine_stored for comparison.

This streams the file once (CPU, ~15 s) and writes:
  results/cosine_task_pairs.csv     condition, layer, task_i, task_j, cosine (recomputed),
                                    cosine_stored (old column), no vectors
  results/probe_weights.npz         each unique probe vector once, float32, keyed
                                    "{condition}|{task}|{layer}"; a few hundred MB smaller
                                    and loadable with np.load
  results/cosine_cross_condition.csv  NEW: the same task's probe under two conditions at
                                    the same layer (e.g. no-prompt F5 vs sentence-based-CoT
                                    F5). This is the number the transfer story needs: do the
                                    plain and CoT probes point the same way?
It also checks that the stored vectors reproduce the stored cosine values.

Reference point: two random directions in 4096 dimensions have |cosine| around 0.016,
so values of 0.1-0.2 are already far from chance.

Usage:
  python experiments/review_experiments/exp8_cosine_summary.py \
      --input experiments/cosine_similarity_mahalanobis.csv   # after `git lfs pull`
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def stream(path):
    csv.field_size_limit(sys.maxsize)
    pairs, vectors, checks = [], {}, []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            cond, layer = row["condition"], int(row["layer"])
            ti, tj = row["task_i"], row["task_j"]
            cos = float(row["cosine_similarity"])
            for task, col in ((ti, "weight_i"), (tj, "weight_j")):
                key = f"{cond}|{task}|{layer}"
                if key not in vectors and row.get(col):
                    vectors[key] = np.array(row[col].split(), dtype=np.float32)
            ka, kb = f"{cond}|{ti}|{layer}", f"{cond}|{tj}|{layer}"
            recomputed = C_cos(vectors[ka], vectors[kb]) if ka in vectors and kb in vectors else float("nan")
            pairs.append((cond, layer, ti, tj, recomputed, cos))
            checks.append(abs(recomputed - cos))
    cols = ["condition", "layer", "task_i", "task_j", "cosine", "cosine_stored"]
    return pd.DataFrame(pairs, columns=cols), vectors, checks


def C_cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(C.REPO_ROOT / "experiments" / "cosine_similarity_mahalanobis.csv"))
    args = ap.parse_args()

    with open(args.input) as f:
        if f.read(40).startswith("version https://git-lfs"):
            sys.exit("This is the Git LFS pointer, not the data. Run `git lfs pull` first.")

    pairs, vectors, checks = stream(args.input)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(C.RESULTS_DIR / "cosine_task_pairs.csv", index=False)
    np.savez_compressed(C.RESULTS_DIR / "probe_weights.npz", **vectors)
    print(f"{len(pairs)} task-pair rows, {len(vectors)} unique probe vectors")
    print(f"conditions: {sorted(pairs['condition'].unique())}")
    checks = np.array(checks)
    print(f"stored cosine vs recomputed from stored vectors: {(checks > 0.01).sum()}/{len(checks)} rows "
          f"differ by > 0.01 (max {np.nanmax(checks):.2f}). Use the recomputed `cosine` column.")

    # Cross-condition, same task, same layer.
    conds = sorted(pairs["condition"].unique())
    rows = []
    for a, b in itertools.combinations(conds, 2):
        for task in C.TASKS:
            for layer in sorted(pairs["layer"].unique()):
                ka, kb = f"{a}|{task}|{layer}", f"{b}|{task}|{layer}"
                if ka in vectors and kb in vectors:
                    rows.append({"condition_a": a, "condition_b": b, "task": task, "layer": layer,
                                 "cosine": C_cos(vectors[ka], vectors[kb])})
    cross = pd.DataFrame(rows)
    cross.to_csv(C.RESULTS_DIR / "cosine_cross_condition.csv", index=False)

    rng = np.random.default_rng(0)
    dim = next(iter(vectors.values())).shape[0]
    rand = [C_cos(rng.standard_normal(dim), rng.standard_normal(dim)) for _ in range(2000)]
    print(f"random-direction reference: mean |cos| = {np.mean(np.abs(rand)):.3f} (dim {dim})")

    if len(cross):
        print("\nSame task, plain vs CoT probe, cosine at layers 8/16/24/32 and the max over layers:")
        for (a, b), g in cross.groupby(["condition_a", "condition_b"]):
            piv = g.pivot_table(index="task", columns="layer", values="cosine")
            show = piv[[c for c in [8, 16, 24, 32] if c in piv.columns]].copy()
            show["max"] = piv.max(axis=1)
            show["argmax_layer"] = piv.idxmax(axis=1)
            print(f"\n  {a}  vs  {b}")
            print(show.reindex(C.TASKS).round(3).to_string())


if __name__ == "__main__":
    main()
