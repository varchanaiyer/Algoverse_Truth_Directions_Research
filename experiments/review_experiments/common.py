"""
Shared helpers for the review experiments (exp3-exp9).

Everything here is lifted from the existing notebooks so results stay comparable with
results_database.csv:
  - extract_all_layer_activations is the Figure_3_Cross_Condition_Generalization version
    (length-sorted batches, left padding, final-token residual stream at every layer).
  - fit_linear_probe is the same bias-free, mean-centred probe (Adam, lr 1e-3,
    weight_decay 0.1, 1000 full-batch steps), with one change: it takes a seed.

What is new:
  - a seed on every probe, plus a bootstrap confidence interval on every AUROC. The seed
    only changes the probe's random init, and full-batch Adam lands in nearly the same
    place every time, so seed spread on its own will look tiny. The bootstrap over test
    items is the error bar that reflects how few test items there are (~290 for A1-A3).
  - a results file with more key columns (experiment, readout, seed, eval target, subset)
    so reruns with different seeds or readouts sit side by side instead of overwriting
    each other the way drop_duplicates(keep='last') does in results_database.csv.

GPU-only imports (torch, transformers) are deferred into the functions that need them so
the CPU experiments (exp8, exp9, and the data-prep halves of exp3/exp6) run on a laptop.
"""

import gc
import os
import re
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "review_results.csv"

TASKS = ["A1", "A2", "A3", "F0", "F1", "F2", "F3", "F4", "F5"]
ARITH_TASKS = ["A1", "A2", "A3"]
FACT_TASKS = ["F0", "F1", "F2", "F3", "F4", "F5"]

DEEPSEEK_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_SHORT = "deepseek-r1-distill-8b"  # same label results_database.csv uses
DEEPSEEK_PAD_ID = 128001  # pad id hardcoded in the Figure 3 notebook

# Same folders as CONDITION_DICT in the existing notebooks.
CONDITION_DICT = {
    "no-prompt": DATA_DIR / "plain_dataset",
    "cot-zero-shot": DATA_DIR / "CoT_datasets" / "lexically_cleaned",
    "sentence-based-CoT": DATA_DIR / "CoT_datasets" / "sentence_based_lexically_cleaned",
    "no-prompt-chat-template": DATA_DIR / "plain_template_dataset",
    "ablation-filler-token": DATA_DIR / "ablation_datasets" / "filler_token_only",
    "ablation-instructions-and-template": DATA_DIR / "ablation_datasets" / "instructions_and_template",
    "ablation-instructions-only": DATA_DIR / "ablation_datasets" / "instructions_only",
}

DEEPSEEK_MARKERS = ["<｜begin▁of▁sentence｜>", "<｜User｜>"]


# ---------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------

def to_binary_labels(series):
    """Labels are stored as True/False in most files and 0/1 in others (F5). pandas reads
    True/False as bool already; strings are handled too in case a file was re-saved."""
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    return series.map(lambda v: mapping[str(v).strip().lower()]).astype(int).to_numpy()


def load_split(condition, task, split):
    path = CONDITION_DICT[condition] / f"{task}_{split}.csv"
    df = pd.read_csv(path)
    df["y"] = to_binary_labels(df["label"])
    return df


def statement_text(row):
    """The bare statement for any condition's row. Plain files have a `statement` column;
    every other file starts its text with the statement (after DeepSeek's BOS/User markers
    when a chat template was used) and follows it with a blank line or <｜Assistant｜>."""
    if "statement" in row and isinstance(row["statement"], str):
        return row["statement"].strip()
    text = row.get("generated_statement_texts", row.get("extracted_statement_texts"))
    for marker in DEEPSEEK_MARKERS:
        text = text.replace(marker, "")
    return re.split(r"\n\n|<｜Assistant｜>", text, maxsplit=1)[0].strip()


def ids_for(df, tokenizer=None):
    """Token-id lists for each row. Plain files carry no ids, so they are tokenized here
    exactly as the Figure 3 notebook does (tokenizer default, which adds BOS)."""
    if "extracted_statement_ids" in df.columns:
        return [literal_eval(x) if isinstance(x, str) else list(x) for x in df["extracted_statement_ids"]]
    if tokenizer is None:
        raise ValueError("plain dataset rows need a tokenizer to produce ids")
    return tokenizer(df["statement"].tolist())["input_ids"]


# ---------------------------------------------------------------------------------------
# Model + activations (GPU)
# ---------------------------------------------------------------------------------------

def load_model(model_id=DEEPSEEK_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_all_layer_activations(id_lists, model, pad_token_id, batch_size=4, desc="Extracting"):
    """Final-token residual stream at every layer, one array per layer, rows in input order.

    Same algorithm as the Figure 3 notebook: sort by length, left-pad, read [:, -1, :],
    un-sort. Readout position is controlled by the caller truncating each id list, which
    is exact for a causal model: the activation at position k only sees tokens 0..k, so
    reading the last token of ids[:k+1] equals reading position k of the full sequence.
    """
    import torch
    from tqdm.auto import tqdm

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tensors = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
    lengths = torch.tensor([len(t) for t in tensors])
    sort_idx = torch.argsort(lengths)
    inverse_idx = torch.argsort(sort_idx)
    sorted_tensors = [tensors[i] for i in sort_idx.tolist()]

    n_layers = model.config.num_hidden_layers + 1
    per_layer = [[] for _ in range(n_layers)]
    device = next(model.parameters()).device

    with tqdm(total=len(sorted_tensors), desc=desc, leave=False) as pbar:
        for i in range(0, len(sorted_tensors), batch_size):
            batch = sorted_tensors[i:i + batch_size]
            padded = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=pad_token_id, padding_side="left"
            ).to(device)
            # Mask by length, not by `padded != pad_token_id`: the pad id can also be a real
            # token inside a sequence (Granite reuses EOS as pad).
            lens = torch.tensor([len(t) for t in batch], device=device)
            positions = torch.arange(padded.shape[1], device=device).unsqueeze(0)
            attention_mask = (positions >= (padded.shape[1] - lens).unsqueeze(1)).long()
            with torch.no_grad():
                out = model(input_ids=padded, attention_mask=attention_mask, output_hidden_states=True)
            for layer_idx, hidden in enumerate(out.hidden_states):
                per_layer[layer_idx].append(hidden[:, -1, :].float().cpu())
            del out, padded, attention_mask
            pbar.update(len(batch))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [torch.cat(acts, dim=0)[inverse_idx].numpy() for acts in per_layer]


def cached_activations(cache_dir, key, compute_fn):
    """Optional disk cache (float16 .npz) keyed by a string the caller builds from model,
    condition, task, split and readout. Several experiments read the same activations
    (plain and sentence-based-CoT at the final token), so pointing them all at one cache
    folder on Drive saves most of the GPU time after the first run."""
    if cache_dir is None:
        return compute_fn()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        data = np.load(path)
        return [data[f"l{i}"].astype(np.float32) for i in range(len(data.files))]
    acts = compute_fn()
    np.savez(path, **{f"l{i}": a.astype(np.float16) for i, a in enumerate(acts)})
    return acts


# ---------------------------------------------------------------------------------------
# Probes + scoring
# ---------------------------------------------------------------------------------------

def fit_linear_probe(X_train, y_train, seed=0):
    """Bias-free logistic probe on mean-centred activations (as in the notebooks), seeded."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(X_train, dtype=torch.float32)
    train_mean = X.mean(dim=0)
    Xc = (X - train_mean).to(device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)

    probe = nn.Linear(X.shape[1], 1, bias=False).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(1000):
        opt.zero_grad()
        loss_fn(probe(Xc).squeeze(-1), y).backward()
        opt.step()
    return probe.weight.detach().cpu().numpy().flatten(), train_mean.numpy()


def probe_scores(w, train_mean, X):
    return (X - train_mean) @ w


def auroc_with_ci(y, scores, n_boot=1000, seed=0):
    """AUROC plus a 95% percentile bootstrap interval over test items. Returns NaNs when
    the subset has a single class (AUROC undefined) instead of the 0.5 the notebooks use,
    so an undefined number can never be read as chance."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(y, scores)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    if not boots:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def parse_layers(arg, n_layers):
    """--layers all | 8,12,16 | 0-32:4"""
    if arg in (None, "all"):
        return list(range(n_layers))
    if ":" in arg:
        span, step = arg.split(":")
        lo, hi = span.split("-")
        return list(range(int(lo), min(int(hi), n_layers - 1) + 1, int(step)))
    return [int(x) for x in arg.split(",") if int(x) < n_layers]


# ---------------------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------------------

RESULT_KEYS = [
    "experiment", "model", "train_task", "test_task", "train_condition", "test_condition",
    "readout", "layer", "seed", "eval_target", "subset",
]
RESULT_COLUMNS = RESULT_KEYS + ["n_train", "n_test", "auroc", "ci_low", "ci_high"]


def append_results(rows, csv_path=None):
    """Append rows, replacing only rows that match on every key column (so a rerun of the
    same seed and readout replaces itself, and a new seed adds rows)."""
    if not rows:
        return
    csv_path = Path(csv_path or RESULTS_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)[RESULT_COLUMNS]
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=RESULT_KEYS, keep="last")
    new.to_csv(csv_path, index=False)


def result_row(**kw):
    row = {c: kw.get(c, "") for c in RESULT_COLUMNS}
    row.update({k: v for k, v in kw.items() if k in RESULT_COLUMNS})
    return row
