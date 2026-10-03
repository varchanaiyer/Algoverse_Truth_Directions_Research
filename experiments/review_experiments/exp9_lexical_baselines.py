"""
Experiment 9: put the word-only baselines in the same table as the probes.

lexical_shortcut_baseline.py writes its own CSVs, separate from results_database.csv,
and runs on the CoT data only. This reruns the same baselines (same vectorizer settings,
same LogisticRegression) on three kinds of input, adds bootstrap CIs, writes them to
results/review_results.csv in the same row format as the probes, and builds the main
comparison table the paper needs.

Inputs:
  cot-full        the full extracted CoT prefix (prompt + statement + reasoning). This is
                  what lexical_shortcut_baseline.py uses, so it should reproduce those numbers.
  cot-reasoning   only the reasoning text after the prompt. The prompt is the same for every
                  item except the statement, so this isolates what the REASONING words give away.
  plain-statement the bare statement. Can the statement's own words (city and country
                  names, numbers) predict its truth without any model at all?
Plus the final-token one-hot baseline on the CoT prefix, as in the original script.

Output:
  results/main_table.csv and .md: per task, best-layer probe AUROC (in-domain plain, in-domain
  CoT, plain->CoT, CoT->plain, from results_database.csv) next to the four baselines.

Usage (CPU, ~1 minute):
  python experiments/review_experiments/exp9_lexical_baselines.py
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COT = "sentence-based-CoT"
PLAIN = "no-prompt"
TOKEN_PATTERN = r"(?u)\b[A-Za-z0-9']+\b"  # same as lexical_shortcut_baseline.py


def reasoning_only(text):
    return text.split("<think>", 1)[-1]


def last_word(text):
    tokens = re.findall(r"[A-Za-z']+|\d+", str(text).rstrip())
    return tokens[-1].lower() if tokens else "<EMPTY>"


def tfidf_scores(train_texts, y_train, test_texts, seed):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), token_pattern=TOKEN_PATTERN)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(vec.fit_transform(train_texts), y_train)
    return clf.predict_proba(vec.transform(test_texts))[:, 1]


def final_token_scores(train_texts, y_train, test_texts, seed):
    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr = enc.fit_transform(np.array([last_word(t) for t in train_texts]).reshape(-1, 1))
    Xte = enc.transform(np.array([last_word(t) for t in test_texts]).reshape(-1, 1))
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def run_baselines(tasks, seed):
    rows = []
    for task in tasks:
        ctr, cte = C.load_split(COT, task, "train"), C.load_split(COT, task, "test")
        ptr, pte = C.load_split(PLAIN, task, "train"), C.load_split(PLAIN, task, "test")
        cot_full = (ctr["extracted_statement_texts"].fillna(""), cte["extracted_statement_texts"].fillna(""))
        cot_reason = (cot_full[0].map(reasoning_only), cot_full[1].map(reasoning_only))
        variants = [
            ("tfidf-bow", "cot-full", cot_full, ctr, cte, tfidf_scores),
            ("tfidf-bow", "cot-reasoning", cot_reason, ctr, cte, tfidf_scores),
            ("tfidf-bow", "plain-statement", (ptr["statement"], pte["statement"]), ptr, pte, tfidf_scores),
            ("final-token", "cot-full", cot_full, ctr, cte, final_token_scores),
        ]
        for model_name, readout, (tr_text, te_text), tr, te, fn in variants:
            scores = fn(tr_text.tolist(), tr["y"], te_text.tolist(), seed)
            auc, lo, hi = C.auroc_with_ci(te["y"], scores, seed=seed)
            cond = PLAIN if readout == "plain-statement" else COT
            rows.append(C.result_row(
                experiment="exp9_lexical_baselines", model=model_name, train_task=task, test_task=task,
                train_condition=cond, test_condition=cond, readout=readout, layer=-1, seed=seed,
                eval_target="label", subset="all", n_train=len(tr), n_test=len(te),
                auroc=auc, ci_low=lo, ci_high=hi))
    C.append_results(rows)
    return pd.DataFrame(rows)


def probe_best(db, train_c, test_c):
    sub = db[(db["model"] == C.DEEPSEEK_SHORT) & (db["train_task"] == db["test_task"])
             & (db["train_condition"] == train_c) & (db["test_condition"] == test_c)]
    idx = sub.groupby("train_task")["auroc"].idxmax()
    best = sub.loc[idx].set_index("train_task")
    return best.apply(lambda r: f"{r.auroc:.3f} (L{int(r.layer)})", axis=1)


def build_table(baselines):
    db = pd.read_csv(C.REPO_ROOT / "experiments" / "results_database.csv")
    table = pd.DataFrame(index=C.TASKS)
    table["probe plain"] = probe_best(db, PLAIN, PLAIN)
    table["probe CoT"] = probe_best(db, COT, COT)
    table["probe plain->CoT"] = probe_best(db, PLAIN, COT)
    table["probe CoT->plain"] = probe_best(db, COT, PLAIN)

    def fmt(r):
        return f"{r.auroc:.3f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"

    for (model_name, readout), label in [
        (("tfidf-bow", "plain-statement"), "TF-IDF statement"),
        (("tfidf-bow", "cot-reasoning"), "TF-IDF CoT reasoning"),
        (("tfidf-bow", "cot-full"), "TF-IDF CoT full"),
        (("final-token", "cot-full"), "final word CoT"),
    ]:
        sub = baselines[(baselines["model"] == model_name) & (baselines["readout"] == readout)]
        table[label] = sub.set_index("train_task").apply(fmt, axis=1)

    table.index.name = "task"
    table.to_csv(C.RESULTS_DIR / "main_table.csv")
    try:
        md = table.to_markdown()
    except ImportError:
        md = table.to_string()
    (C.RESULTS_DIR / "main_table.md").write_text(
        "Probe columns: best-layer AUROC from results_database.csv (DeepSeek-R1-Distill-8B), "
        "layer in brackets. Probe values have no CI yet: rerun with exp6 seeds/bootstrap.\n"
        "Baselines: AUROC with 95% bootstrap CI over test items.\n\n" + md + "\n")
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    baselines = run_baselines(args.tasks, args.seed)
    build_table(baselines)


if __name__ == "__main__":
    main()
