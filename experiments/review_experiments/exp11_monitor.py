"""
Experiment 11: the three additions for the DAI (Deployable AI) version of the paper.

  monitor  Can a probe read on the statement, BEFORE any reasoning, flag the answers that
           chain-of-thought gets wrong? Every item gets three answers: the CoT answer (from
           the saved traces), the model's no-reasoning answer (exp10) and the plain probe's
           answer (out-of-fold, same folds as exp3/exp10). A detector flags an item when
           its answer disagrees with the CoT answer. Reported per task and detector:
             caught / cot_errors   (recall: share of CoT errors that get flagged)
             precision             (share of flags that are real CoT errors)
             false_flag_rate       (share of correct CoT answers flagged by mistake)
             kept_accuracy         (CoT accuracy on the items NOT flagged, plus coverage)
             error_AUROC [CI]      threshold-free: how well the detector's confidence that
                                   the CoT answer is wrong ranks the real errors
           The no-reasoning answer is the baseline: it costs the same single forward pass,
           so the probe only adds something where it beats it. A precision-recall plot per
           task is saved too.
  cost     Tokens the probe needs (the statement, one forward pass) against tokens the
           reasoning needs (the generated trace), per task, from the saved data. With
           --time-generation N it also times N real generations against N probe passes.
  control  Shuffled-label control (Hewitt and Liang 2019): the same probe, same folds and
           layers, trained on randomly permuted labels and scored against the true labels.
           It should sit at 0.5; anything well above means the probe is picking up an
           artifact of the data or the pipeline. It does NOT test whether the probe itself
           does the combining on F3 to F5 (that needs per-city probes, see the README).

Usage (Colab, A100/L4, ~30 min for F0-F5 with all three parts):
  python experiments/review_experiments/exp11_monitor.py --tasks F0 F1 F2 F3 F4 F5 \
      --cache-dir /content/dsab_cache/exp11 --time-generation 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import exp3_probe_vs_verdict as E3  # noqa: E402
import exp10_direct_answer as E10  # noqa: E402

COT = "sentence-based-CoT"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def oof_probe_scores(acts, y, cot, statements, layers, folds, seed):
    """Out-of-fold plain-probe scores for every layer, folds identical to exp3/exp10."""
    groups = pd.factorize(pd.Series(statements))[0]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(splitter.split(np.zeros(len(y)), y * 2 + cot, groups)):
        fold_of[te] = k
    return fold_of, {
        layer: _oof(acts[layer], y, fold_of, folds, seed) for layer in layers
    }


def _oof(X, labels, fold_of, folds, seed):
    scores = np.zeros(len(labels))
    for k in range(folds):
        tr, te = fold_of != k, fold_of == k
        w, m = C.fit_linear_probe(X[tr], labels[tr], seed)
        scores[te] = C.probe_scores(w, m, X[te])
    return scores


def detector_row(task, name, y, cot, answer, err_conf, seed):
    """Flag = detector answer disagrees with the CoT answer."""
    cot_wrong = cot != y
    flag = answer != cot
    n_err, n_flag = int(cot_wrong.sum()), int(flag.sum())
    caught = int((flag & cot_wrong).sum())
    kept = ~flag
    auc, lo, hi = C.auroc_with_ci(cot_wrong.astype(int), err_conf, seed=seed)
    return {
        "task": task, "detector": name, "items": len(y), "cot_errors": n_err,
        "cot_accuracy": round(float((cot == y).mean()), 3),
        "flagged": n_flag, "caught": caught,
        "recall": round(caught / n_err, 3) if n_err else float("nan"),
        "precision": round(caught / n_flag, 3) if n_flag else float("nan"),
        "false_flag_rate": round(float((flag & ~cot_wrong).sum() / max(1, (~cot_wrong).sum())), 3),
        "coverage": round(float(kept.mean()), 3),
        "kept_accuracy": round(float((cot[kept] == y[kept]).mean()), 3) if kept.any() else float("nan"),
        "error_AUROC": f"{auc:.3f} [{lo:.2f}, {hi:.2f}]",
    }


def load_or_make_direct(task, v, model, tokenizer):
    """exp10's per-item no-reasoning answers if they are saved and line up, else recompute."""
    path = C.RESULTS_DIR / "exp10_direct_answers" / f"{task}.csv"
    if path.exists():
        d = pd.read_csv(path)
        if len(d) == len(v) and (d["statement"].values == v["statement"].values).all():
            return d["p_yes"].to_numpy(), "exp10 file"
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    p_yes, p_no = E10.direct_answers(E10.direct_prompt_ids(v["statement"].tolist(), tokenizer),
                                     model, C.DEEPSEEK_PAD_ID, yes_id, no_id)
    return p_yes / (p_yes + p_no), "recomputed"


def run(args):
    if not all((E3.VERDICT_DIR / f"{t}.csv").exists() for t in args.tasks):
        E3.build_verdicts(args.tasks)
    model, tokenizer = C.load_model(C.DEEPSEEK_ID)
    layers = C.parse_layers(args.layers, model.config.num_hidden_layers + 1)
    out_dir = C.RESULTS_DIR / "exp11_monitor"
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_rows, cost_rows, control_rows, pr_curves = [], [], [], {}

    for task in args.tasks:
        v = pd.read_csv(E3.VERDICT_DIR / f"{task}.csv")
        v = v[v["model_verdict"].notna()].reset_index(drop=True)
        y = v["label"].to_numpy().astype(int)
        cot = v["model_verdict"].to_numpy().astype(int)
        statements = v["statement"].tolist()

        plain_ids = E3.plain_rows_for(statements, task, tokenizer)
        acts = C.cached_activations(
            args.cache_dir, f"{C.DEEPSEEK_SHORT}__no-prompt-matched__{task}__all__final",
            lambda: C.extract_all_layer_activations(plain_ids, model, C.DEEPSEEK_PAD_ID,
                                                    args.batch_size, f"{task} plain"))
        fold_of, scores = oof_probe_scores(acts, y, cot, statements, layers, args.folds, args.seed)
        auc_by_layer = {l: C.auroc_with_ci(y, s, n_boot=1)[0] for l, s in scores.items()}
        layer = max(auc_by_layer, key=auc_by_layer.get)

        # ---- monitor ----
        if "monitor" in args.parts:
            p_true = sigmoid(scores[layer])
            probe_ans = (scores[layer] > 0).astype(int)
            p_yes, src = load_or_make_direct(task, v, model, tokenizer)
            quick_ans = (p_yes > 0.5).astype(int)
            # confidence that the CoT answer is wrong = probability the detector puts on the
            # opposite answer
            probe_err = np.where(cot == 1, 1 - p_true, p_true)
            quick_err = np.where(cot == 1, 1 - p_yes, p_yes)
            rows = [
                detector_row(task, "plain probe", y, cot, probe_ans, probe_err, args.seed),
                detector_row(task, "no-reasoning answer", y, cot, quick_ans, quick_err, args.seed),
            ]
            # stricter versions: flag only when the detector is at least 90% sure
            for name, err in (("plain probe (>=0.9 sure)", probe_err), ("no-reasoning answer (>=0.9 sure)", quick_err)):
                strict = np.where(err >= 0.9, 1 - cot, cot)
                rows.append(detector_row(task, name, y, cot, strict, err, args.seed))
            for r in rows:
                r["probe_layer"], r["quick_answer_source"] = layer, src
            monitor_rows += rows
            cot_wrong = (cot != y).astype(int)
            if cot_wrong.sum() and cot_wrong.sum() < len(cot_wrong):
                pr_curves[task] = {name: precision_recall_curve(cot_wrong, err)[:2]
                                   for name, err in (("plain probe", probe_err), ("no-reasoning answer", quick_err))}
            pd.DataFrame({
                "split": v["split"], "statement": v["statement"], "label": y, "cot_answer": cot,
                "no_reasoning_p_yes": p_yes, "probe_p_true": p_true, "probe_answer": probe_ans,
                "probe_flags_cot": (probe_ans != cot).astype(int), "quick_flags_cot": (quick_ans != cot).astype(int),
            }).to_csv(out_dir / f"{task}_items.csv", index=False)
            print(pd.DataFrame(rows)[["detector", "flagged", "caught", "cot_errors", "recall", "precision",
                                      "false_flag_rate", "kept_accuracy", "error_AUROC"]].to_string(index=False))

        # ---- cost ----
        if "cost" in args.parts:
            cot_df = pd.concat([C.load_split(COT, task, s) for s in ["train", "test"]], ignore_index=True)
            from ast import literal_eval
            gen = [literal_eval(x) if isinstance(x, str) else list(x) for x in cot_df["generated_statement_ids"]]
            reason_tokens = np.array([len(g) - int(p) for g, p in zip(gen, cot_df["prompt_len"])])
            stmt_tokens = np.array([len(x) for x in plain_ids])
            cost_rows.append({
                "task": task,
                "probe_tokens_median": int(np.median(stmt_tokens)),
                "reasoning_tokens_median": int(np.median(reason_tokens)),
                "reasoning_tokens_p90": int(np.percentile(reason_tokens, 90)),
                "token_ratio": round(float(np.median(reason_tokens) / np.median(stmt_tokens)), 1),
            })

        # ---- shuffled-label control ----
        if "control" in args.parts:
            rng = np.random.default_rng(args.seed)
            # the probe's layer plus every 4th layer: enough for a best-case-over-layers
            # number without paying for shuffles x all layers x folds
            control_layers = sorted(set(layers[::4]) | {layer})
            shuffled_auc = {l: [] for l in control_layers}
            for s in range(args.shuffles):
                y_shuf = rng.permutation(y)
                for l in control_layers:
                    sc = _oof(acts[l], y_shuf, fold_of, args.folds, args.seed + s)
                    shuffled_auc[l].append(C.auroc_with_ci(y, sc, n_boot=1)[0])
            mean_by_layer = {l: float(np.mean(a)) for l, a in shuffled_auc.items()}
            control_rows.append({
                "task": task, "probe_layer": layer,
                "real_AUROC_at_layer": round(auc_by_layer[layer], 3),
                "shuffled_AUROC_at_layer": round(mean_by_layer[layer], 3),
                "shuffled_AUROC_max_over_layers": round(max(mean_by_layer.values()), 3),
                "selectivity": round(auc_by_layer[layer] - mean_by_layer[layer], 3),
                "shuffles": args.shuffles,
            })
            print(f"  {task} control: real {auc_by_layer[layer]:.3f} vs shuffled {mean_by_layer[layer]:.3f} (layer {layer})")
        del acts

    if monitor_rows:
        _merge(pd.DataFrame(monitor_rows), C.RESULTS_DIR / "exp11_monitor_summary.csv", ["task", "detector"])
        _plot_pr(pr_curves)
    if cost_rows:
        cost = pd.DataFrame(cost_rows)
        if args.time_generation:
            cost = _time_generation(cost, args, model, tokenizer)
        _merge(cost, C.RESULTS_DIR / "exp11_cost.csv", ["task"])
        print(cost.to_string(index=False))
    if control_rows:
        _merge(pd.DataFrame(control_rows), C.RESULTS_DIR / "exp11_shuffled_control.csv", ["task"])
        print(pd.DataFrame(control_rows).to_string(index=False))


def _time_generation(cost, args, model, tokenizer):
    """Wall-clock for N statements: one probe forward pass each vs a full reasoning trace."""
    import torch

    task = args.tasks[0]
    cot_df = C.load_split(COT, task, "test").head(args.time_generation)
    prompts = [ids[:int(p)] for ids, p in zip(C.ids_for(cot_df), cot_df["prompt_len"])]
    stmts = E3.plain_rows_for([C.statement_text(r) for _, r in cot_df.iterrows()], task, tokenizer)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    device = next(model.parameters()).device
    # warm-up so neither timing includes one-off CUDA/kernel start-up cost
    C.extract_all_layer_activations(stmts[:2], model, C.DEEPSEEK_PAD_ID, 1, "warm-up")
    with torch.no_grad():
        model.generate(torch.tensor([prompts[0]]).to(device), max_new_tokens=4, do_sample=False,
                       pad_token_id=C.DEEPSEEK_PAD_ID)
    sync(); t0 = time.time()
    C.extract_all_layer_activations(stmts, model, C.DEEPSEEK_PAD_ID, 1, "time probe")
    sync(); t_probe = (time.time() - t0) / len(stmts)
    t0 = time.time()
    for p in prompts:
        with torch.no_grad():
            model.generate(torch.tensor([p]).to(device),
                           max_new_tokens=args.max_new_tokens, do_sample=False,
                           pad_token_id=C.DEEPSEEK_PAD_ID)
    sync(); t_gen = (time.time() - t0) / len(prompts)
    cost["timed_on"] = f"{task}, {len(prompts)} items, batch 1"
    cost["probe_seconds_per_item"] = round(t_probe, 4)
    cost["reasoning_seconds_per_item"] = round(t_gen, 2)
    cost["time_ratio"] = round(t_gen / t_probe, 1)
    return cost


def _merge(new, path, keys):
    if path.exists():
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates(keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def _plot_pr(curves):
    if not curves:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(3.2 * len(curves), 3.2), squeeze=False)
    for ax, (task, c) in zip(axes[0], curves.items()):
        for name, (prec, rec) in c.items():
            ax.plot(rec, prec, label=name)
        ax.set_title(task)
        ax.set_xlabel("CoT errors caught (recall)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0][0].set_ylabel("flags that are errors (precision)")
    axes[0][-1].legend(fontsize=7)
    plt.tight_layout()
    out = C.RESULTS_DIR / "exp11_monitor_pr.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.FACT_TASKS)
    ap.add_argument("--parts", nargs="+", default=["monitor", "cost", "control"],
                    choices=["monitor", "cost", "control"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seed", type=int, default=0, help="same fold seed as exp3/exp10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--shuffles", type=int, default=3, help="label permutations for the control")
    ap.add_argument("--time-generation", type=int, default=0, help="time N real generations (0 = skip)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
