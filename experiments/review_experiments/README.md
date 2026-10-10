# Review experiments (exp3 to exp9)

These scripts test the main claim of the paper, that CoT restores a truth direction that is
missing for complex statements. Each one is a standalone script
that writes to `results/review_results.csv`. That file has the same columns as
`results_database.csv` plus `experiment`, `readout`, `seed`, `eval_target`, `subset`,
`n_train`, `n_test`, `ci_low` and `ci_high`, so seeds and readouts sit side by side and
nothing is overwritten.

The probe, the activation extraction and the hyperparameters are copied from the existing
notebooks (`Figure_3_Cross_Condition_Generalization.ipynb`), so the numbers are comparable
with what you already have.

## What each experiment answers

| # | Script | Question | Needs |
|---|---|---|---|
| 3 | `exp3_probe_vs_verdict.py` | On the items where the model's answer is wrong, does the probe follow the true label or the model's answer? | parse: CPU, probes: GPU |
| 4 | `exp4_early_readout.py` | How readable is truth at 0/25/50/75/100% of the way through the reasoning, and does the plain probe start to work partway through? | TF-IDF: CPU, probes: GPU |
| 5 | `exp5_readout_position.py` | How much of the drop from plain to template/instructions comes from the readout moving off the statement? | GPU |
| 6 | `exp6_matched_items.py` | Do plain vs CoT numbers change when both use exactly the same statements? | prepare: CPU, probes: GPU |
| 7 | `exp7_granite_conditions.py` | Does Granite show the same plain vs CoT pattern? (Currently it only has the CoT condition.) | GPU (~17 GB) |
| 8 | `exp8_cosine_summary.py` | Do the plain and CoT probes point the same way? Also shrinks the 1.4 GB cosine file. | CPU |
| 9 | `exp9_lexical_baselines.py` | The main results table: probes next to word-only baselines, with confidence intervals. | CPU |

`smoke_test.py` runs exp3 to exp7 end to end on a tiny random model on CPU, to catch code
errors before you spend GPU time. Run it after any change.

## GPU results (Colab run, 10/7, A100 80 GB)

Everything below is in `results/review_results.csv`. The sanity check reproduced the
existing A2 numbers in `results_database.csv` exactly before anything else ran.

- **exp3, probe vs the model's own answer.** On the factual items where the model's final
  (post-reasoning) answer is wrong, AUROC against the true label, layer chosen on all items:

  | task | n wrong | CoT probe | plain probe on plain | plain probe on CoT | verdict probe |
  |---|---|---|---|---|---|
  | F0 | 161 | 0.155 [0.09, 0.23] | 0.916 [0.87, 0.96] | 0.312 [0.21, 0.41] | 0.030 |
  | F1 | 91 | 0.662 [0.51, 0.80] | 0.862 [0.75, 0.96] | 0.618 [0.49, 0.74] | 0.482 |
  | F2 | 100 | 0.054 [0.01, 0.11] | 0.729 [0.62, 0.82] | 0.161 [0.09, 0.25] | 0.017 |
  | F3 | 78 | 0.057 [0.02, 0.11] | 0.815 [0.72, 0.90] | 0.266 [0.16, 0.39] | 0.025 |
  | F4 | 90 | 0.025 [0.00, 0.06] | 0.578 [0.44, 0.71] | 0.208 [0.10, 0.34] | 0.006 |
  | F5 | 92 | 0.015 [0.00, 0.05] | 0.627 [0.38, 0.86] | 0.282 [0.11, 0.49] | 0.002 |

  After reasoning, the probe follows the model's committed answer, wrong ones included (F1
  is the exception). The plain probe, read on the bare statement, follows the true label on
  the same items for F0 to F3. Limit: the model was never asked for a no-reasoning answer,
  so these numbers describe what a probe can read from the activations, and say nothing yet
  about what the model would have answered without reasoning.
- **exp4, reading partway through the reasoning** (`results/exp4_early_readout.png`). In-domain
  AUROC stays near 0.5 to 0.65 for A1 to A3 and F3 to F5 at 25, 50 and 75 percent of the
  reasoning, then jumps to 0.96 to 1.00 at the pre-verdict cut. The plain probe on CoT
  stays near 0.55 until that last point too. F0 to F2 start near 0.99, dip to 0.82 to 0.90
  mid-trace and recover at the end.
- **exp5, readout position** (`results/exp5_*_table.csv`). Read at the last token of the
  statement, the templated conditions score A1 0.996 and A2 0.81, above plain (0.93, 0.69).
  The drops to 0.51 to 0.62 only appear when reading at the think token.
- **exp6, matched items.** Using only the statements that survived CoT filtering moves the
  plain numbers by 0.02 at most.
- **exp7, Granite.** Plain arithmetic is weak (A2 0.74, A3 0.64), plain to CoT transfer on
  arithmetic is about 0.8, the reverse about 0.6, and removing DeepSeek's chat markers
  changes nothing. Granite reads DeepSeek's reasoning traces in every CoT condition here.

## Already run on CPU

These outputs are in `results/` and in `datasets/`:

- `datasets/CoT_datasets/verdicts/{task}.csv`: the model's own answer for every CoT item.
  Model accuracy: A1 99.3%, A2 99.8%, A3 100%, F0 89.0%, F1 94.3%, F2 93.3%, F3 94.7%, F4 94.7%, F5 94.5%.
  Items where the answer is wrong (train + test pooled): F0 161, F1 91, F2 100, F3 78, F4 90,
  F5 92. Arithmetic has 7, 2 and 0, so **exp3 can only settle the factual tasks.**
- `datasets/matched_items/`: plain statements for exactly the CoT items. Look at
  `results/exp6_dropped_items.csv` first. In arithmetic, nearly every item the CoT
  filtering dropped was a TRUE statement (A2 train 27/27, A2 test 14/14, A3 8/8). The
  filter removes true statements selectively, so the paper should say so, and exp6 measures
  whether it moves the numbers.
- `results/main_table.md`: probes next to the word-only baselines. `TF-IDF CoT full`
  reproduces `baseline_comparison_results.csv` exactly.
- `results/cosine_*.csv`, `results/probe_weights.npz`: see exp8 below.

## Things the CPU runs found that the paper should handle

1. **The stored cosine column cannot be reproduced.** In `cosine_similarity_mahalanobis.csv`
   the `cosine_similarity` value disagrees with its own `weight_i`/`weight_j` vectors in
   6,698 of 8,019 rows, sometimes with the sign flipped. The column was committed 9/8; the
   vectors were attached 9/9 by `Compute_Probe_Weights.ipynb` from retrained probes; the code
   that made the original column is not in the repo. `results/cosine_task_pairs.csv` has the
   values recomputed from the vectors. The computation is plain cosine too, so maybe drop
   mahalanobis from the file name?
2. **Plain and CoT probes are close to orthogonal** (`results/cosine_cross_condition.csv`):
   same-task cosine between the no-prompt and sentence-based-CoT probes is at most 0.06 for
   A1-A3 and 0.10-0.30 for F0-F5, against about 0.01 for random directions. Yet the plain
   probe transfers to CoT at 0.93 AUROC on arithmetic. So maybe the paper could say the
   plain probe's direction stays informative after CoT, and avoid calling the two probes the
   same direction? For comparison, the two CoT variants (zero-shot vs sentence-based) agree
   at 0.50-0.76.
3. **Statement words predict the label backwards across the split.** TF-IDF on the bare
   statement scores 0.15 (F0) and 0.18 (F1), far below chance. In F0, 356 cities appear in
   both train and test, every time with the opposite label (Xining is in China is true in
   train, Xining placed in another country is false in test). That comes from the dataset
   construction, and it helps the paper: probes still reach 0.99, so they are not memorizing
   names. Maybe the Datasets section could describe it?
4. **For arithmetic, the reasoning words only reveal the answer at the very end.** TF-IDF
   on the reasoning text is at or below chance at 25/50/75% of the trace (A2: 0.55, 0.44,
   0.45) and jumps to 0.77 at the pre-verdict cut. For F0-F2 it is about 0.83 from the
   first quarter (the model states the city's country early). The GPU run (exp4) shows the
   probe follows the same shape on arithmetic: near chance mid-trace, 1.00 at the cut.
5. **The instructions end with the literal words Answer: No.** The instructions-only
   ablation reads its activation at that No token for every item, true or false. exp5
   separates this from the effect of the instructions themselves.
6. **The shared CoT instructions mention cities even for arithmetic** (they ask where each
   named city actually is). Some A1 traces go off trying to find the cities in an equation.
   Maybe one line in Limitations?
7. Small: 2 A1 statements appear in both the plain train and plain test split.

## Running the GPU parts (Colab)

Open `run_review_experiments.ipynb` in Colab on an **A100** (or L4) runtime and run it top
to bottom. It is self-contained: it clones this repo at commit `60c8954`, writes the
experiment code from its own cells, runs a 5-minute sanity check that reproduces two
existing numbers, then runs exp3, exp6, exp4, exp5 and exp7 in that order.

- Results go to `MyDrive/dsab_review_results/` as each task finishes
  (`DSAB_RESULTS_DIR`), so a disconnect loses nothing. After reconnecting, rerun the setup
  cells and then the interrupted cell: finished tasks are skipped.
- Activations are cached on Colab's local disk for exp3, exp6 and exp7 only, and deleted
  when each finishes. exp4 and exp5 are not cached: they would need ~17 GB and ~38 GB.
- exp4, exp5 and exp7 use every other layer (`--layers 0-32:2`) to halve the time; exp3
  and exp6 use all layers.

Command-line equivalent:

```bash
export DSAB_RESULTS_DIR=/content/drive/MyDrive/dsab_review_results
python experiments/review_experiments/exp3_probe_vs_verdict.py --tasks F0 F1 F2 F3 F4 F5 --cache-dir /content/dsab_cache/exp3
python experiments/review_experiments/exp6_matched_items.py --cache-dir /content/dsab_cache/exp6
python experiments/review_experiments/exp4_early_readout.py --layers 0-32:2
python experiments/review_experiments/exp5_readout_position.py --tasks A1 A2 A3 F3 F4 F5 --layers 0-32:2
python experiments/review_experiments/exp7_granite_conditions.py --layers 0-40:2 --cache-dir /content/dsab_cache/exp7
python experiments/review_experiments/exp9_lexical_baselines.py
```

Rough time on an A100: exp3 ~45 min, exp6 ~1 h (3 seeds), exp4 ~1.5 h, exp5 ~1 h,
exp7 ~1.5 h. Run `smoke_test.py` (CPU, ~4 min) after changing any of the scripts.

## Reading the results

- exp3: `summarize()` prints, per probe, the AUROC against the TRUE label on items the
  model got wrong, at the layer with the best AUROC on all items. Above 0.5 with a CI that excludes 0.5:
  the probe follows the truth. Below 0.5: it follows the model's answer. The
  `[verdict]` probe (trained on the model's answers) is the reference for what a pure
  verdict probe looks like.
- exp4: `results/exp4_early_readout.png`, three panels (in-domain probe, plain probe on
  CoT, TF-IDF), AUROC against fraction of reasoning read.
- exp5: `results/exp5_{task}_table.csv`, condition x readout. The `statement_end` column
  should be (nearly) identical across conditions that share a prefix, which checks the
  extraction. The comparison between conditions is within the `prompt_end` column.
- exp6 and exp7: printed best-layer tables, and all rows in `review_results.csv`.
- Confidence intervals are 95% bootstrap over test items. The seed only changes the probe's
  initialization, and full-batch Adam converges to nearly the same weights each time, so
  seed spread will look tiny. Maybe report the bootstrap CI as the error bar?
