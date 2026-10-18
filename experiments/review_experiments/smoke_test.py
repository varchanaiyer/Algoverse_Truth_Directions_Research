"""
CPU smoke test for exp3-exp7, exp10 and exp11: runs every GPU script end to end with a tiny RANDOM model
(2 layers, hidden size 32) on ~40 rows per split, so code errors show up in a minute on
a laptop instead of an hour into a Colab session. The AUROC numbers it prints are
meaningless; only "did every script finish" matters.

Nothing touches the real outputs: results, verdict files, matched-item files and the
activation cache all go to a temporary folder.

Usage:
  python experiments/review_experiments/smoke_test.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dsab_smoke_"))
C.RESULTS_DIR = TMP / "results"
C.RESULTS_CSV = C.RESULTS_DIR / "review_results.csv"

_real_load_split = C.load_split


def small_split(condition, task, split):
    df = _real_load_split(condition, task, split)
    return df.groupby("y", group_keys=False).head(20).reset_index(drop=True)


def tiny_model(model_id=C.DEEPSEEK_ID):
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = LlamaConfig(vocab_size=max(len(tok), 128256), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=8192)
    return LlamaForCausalLM(cfg).eval(), tok


C.load_split = small_split
C.load_model = tiny_model


def run(module_name, argv):
    print(f"\n===== {module_name} {' '.join(argv)}")
    module = __import__(module_name)
    sys.argv = [module_name] + argv
    module.main()


if __name__ == "__main__":
    import exp3_probe_vs_verdict as E3
    import exp6_matched_items as E6

    E3.VERDICT_DIR = TMP / "verdicts"
    E6.MATCHED_DIR = TMP / "matched"
    _real_load_matched = E6.load_matched
    E6.load_matched = lambda task, split: _real_load_matched(task, split).groupby(
        "y", group_keys=False).head(20).reset_index(drop=True)

    cache = str(TMP / "cache")
    run("exp3_probe_vs_verdict", ["--tasks", "F0", "--folds", "2", "--cache-dir", cache])
    run("exp4_early_readout", ["--tasks", "A2", "F0", "--fractions", "0", "0.5", "1", "--cache-dir", cache])
    run("exp5_readout_position", ["--tasks", "A2", "--cache-dir", cache])
    run("exp5_readout_position", ["--tasks", "A2", "--cache-dir", cache])  # second run reads the cache
    run("exp6_matched_items", ["--tasks", "A2", "--seeds", "0", "1", "--cache-dir", cache])
    run("exp7_granite_conditions", ["--tasks", "A2", "--cache-dir", cache])
    run("exp10_direct_answer", ["--tasks", "F0", "--folds", "2", "--cache-dir", cache])
    run("exp10_direct_answer", ["--tasks", "F0", "--folds", "2", "--cache-dir", cache])  # second run skips
    run("exp11_monitor", ["--tasks", "F0", "--folds", "2", "--shuffles", "1", "--time-generation", "2", "--max-new-tokens", "8", "--cache-dir", cache])
    print(f"\nAll scripts finished. Throwaway outputs in {TMP}")
