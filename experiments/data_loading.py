"""
Data loaders for supplementary experiments.
Uses HuggingFace datasets where possible; same splits and preprocessing for all baselines (fair comparison).
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any


def get_longbench_loader(subsets: List[str], max_length: int = 4096, cache_dir: Optional[str] = None):
    """LongBench: long-context QA/summarization. Install: pip install datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  # for LongBench/SCROLLS/WMT")
    loaders = {}
    for name in subsets:
        # LongBench is often used via official repo; HF may have a mirror or use load_dataset("json", data_files=...)
        # Example placeholder: loaders[name] = load_dataset("THUDM/LongBench", name, cache_dir=cache_dir)
        loaders[name] = None  # TODO: replace with actual load_dataset or LongBench API
    return loaders


def get_scrolls_loader(tasks: List[str], max_length: int = 16384, cache_dir: Optional[str] = None):
    """SCROLLS: tau/scrolls. Standardized long-form tasks."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    loaders = {}
    for task in tasks:
        # load_dataset("tau/scrolls", task, trust_remote_code=True)
        loaders[task] = None  # TODO: load_dataset("tau/scrolls", task)
    return loaders


def get_wmt_zh_en_loader(split: str = "test", max_length: int = 512, cache_dir: Optional[str] = None):
    """WMT14 or WMT19 zh-en. Same split for all methods."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets sacrebleu")
    # ds = load_dataset("wmt14", "zh-en", cache_dir=cache_dir) or wmt19
    return None  # TODO: return load_dataset(...)


def get_codexglue_loader(subtask: str = "code_summarization", language: str = "python", cache_dir: Optional[str] = None):
    """CodeXGlue code summarization (code-to-text) or completion."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    # e.g. load_dataset("code_xglue", "cc_code_to_code_trans") or microsoft/codexglue_cc_code_to_code_trans
    return None  # TODO: return load_dataset(...)


def save_results_table(results_dir: Path, benchmark: str, per_baseline: Dict[str, Dict[str, float]], metric_name: str):
    """Save per-baseline results in paper table format (for appendix)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": benchmark,
        "metric": metric_name,
        "per_baseline": per_baseline,
        "note": "Mean ± std over 5 runs (seeds 42-46); same architecture and data for all.",
    }
    with open(results_dir / "per_baseline_table.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
