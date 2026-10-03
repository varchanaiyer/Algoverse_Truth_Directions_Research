# Review experiments (exp3 to exp9)

These scripts answer the questions a reviewer will ask about the main claim, "CoT restores a
truth direction that is missing for complex statements." Each one is a standalone script
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

## Already run (CPU parts)

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
- `results/exp4_early_readout.png`: the TF-IDF third of the exp4 figure.

## Things the CPU runs found that the paper needs to handle

1. **The stored cosine column cannot be reproduced.** In `cosine_similarity_mahalanobis.csv`
   the `cosine_similarity` value disagrees with its own `weight_i`/`weight_j` vectors in
   6,698 of 8,019 rows, sometimes with the sign flipped. The column was committed 9/8; the
   vectors were attached 9/9 by `Compute_Probe_Weights.ipynb` from retrained probes; the code
   that made the original column is not in the repo. Use `results/cosine_task_pairs.csv`
   (recomputed from the vectors). Also, the computation is plain cosine, so drop
   "mahalanobis" from the name.
2. **Plain and CoT probes are close to orthogonal** (`results/cosine_cross_condition.csv`):
   same-task cosine between the no-prompt and sentence-based-CoT probes is at most 0.06 for
   A1-A3 and 0.10-0.30 for F0-F5, against about 0.01 for random directions. Yet the plain
   probe transfers to CoT at 0.93 AUROC on arithmetic. So "the same direction" is not the
   right description; something like "the plain probe's direction is still informative
   after CoT" is. For comparison, the two CoT variants (zero-shot vs sentence-based) agree
   at 0.50-0.76.
3. **Statement words predict the label backwards across the split.** TF-IDF on the bare
   statement scores 0.15 (F0) and 0.18 (F1), far below chance. In F0, 356 cities appear in
   both train and test, every time with the opposite label ("Xining is in China" true in
   train, "Xining is in X" false in test). That is a design choice inherited from the
   dataset construction, and a useful one: probes still reach 0.99, so they are not
   memorizing names. Describe it in the Datasets section.
4. **For arithmetic, the reasoning words only reveal the answer at the very end.** TF-IDF
   on the reasoning text is at or below chance at 25/50/75% of the trace (A2: 0.55, 0.44,
   0.45) and jumps to 0.77 at the pre-verdict cut. For F0-F2 it is about 0.83 from the
   first quarter (the model states the city's country early). If the exp4 probe curve for
   arithmetic rises before 100%, the model has the answer internally before it writes it,
   which is a strong result.
5. **The instructions end with the literal words "Answer: No".** The instructions-only
   ablation reads its activation at that "No" token for every item, true or false. exp5
   separates this from the effect of the instructions themselves.
6. **The shared CoT instructions mention cities even for arithmetic** ("for each city
   named, where it actually is"). Some A1 traces go off trying to find the cities in an
   equation. Worth one line in Limitations.
7. Small: 2 A1 statements appear in both the plain train and plain test split.

## Running the GPU parts (Colab)

Use `run_review_experiments.ipynb` in this folder, or the commands below. A single A100
(40 GB) or L4 (24 GB) is enough. Point every script at the same `--cache-dir` on Drive:
the scripts share activations (plain and CoT at the final token), so later runs skip most
of the extraction.

```bash
pip install -r experiments/review_experiments/requirements.txt
python experiments/review_experiments/smoke_test.py   # CPU, ~5 min, must finish cleanly

CACHE=/content/drive/MyDrive/dsab_cache
python experiments/review_experiments/exp3_probe_vs_verdict.py --tasks F0 F1 F2 F3 F4 F5 --cache-dir $CACHE
python experiments/review_experiments/exp4_early_readout.py   --cache-dir $CACHE
python experiments/review_experiments/exp6_matched_items.py   --cache-dir $CACHE
python experiments/review_experiments/exp5_readout_position.py --tasks A1 A2 A3 F3 F4 F5 --cache-dir $CACHE
python experiments/review_experiments/exp7_granite_conditions.py --cache-dir $CACHE
python experiments/review_experiments/exp9_lexical_baselines.py   # rebuilds the main table
```

Order of priority: **3, 6, 4**, then 5 and 7. If GPU time is short, add `--layers 0-32:2`
(every other layer) to any script, which halves the probe time; the best layer will
still be clear. exp6 defaults to 5 seeds; `--seeds 0 1 2` is fine.

Rough cost on an A100, all layers: exp3 ~1 h for the six factual tasks, exp4 ~1.5 h,
exp6 ~1.5 h (5 seeds), exp5 ~1 h for six tasks, exp7 ~1.5 h. Most of it is probe training
(1000 Adam steps x 33 layers x folds/seeds), so `--layers 0-32:2` really does halve it.

## Reading the results

- exp3: `summarize()` prints, per probe, the AUROC against the TRUE label on items the
  model got wrong, at the layer furthest from 0.5. Above 0.5 with a CI that excludes 0.5:
  the probe follows the truth. Below 0.5: it follows the model's answer. The
  `[verdict]` probe (trained on the model's answers) is the reference for what a pure
  verdict probe looks like.
- exp4: `results/exp4_early_readout.png`, three panels (in-domain probe, plain probe on
  CoT, TF-IDF), AUROC against fraction of reasoning read.
- exp5: `results/exp5_{task}_table.csv`, condition x readout. The `statement_end` column
  should be (nearly) identical across conditions that share a prefix; that is a check, not
  a finding. Compare conditions within the `prompt_end` column.
- exp6 and exp7: printed best-layer tables, and all rows in `review_results.csv`.
- Confidence intervals are 95% bootstrap over test items. The seed only changes the probe's
  initialization, and full-batch Adam converges to nearly the same weights each time, so
  seed spread will look tiny. Report the bootstrap CI as the error bar.
