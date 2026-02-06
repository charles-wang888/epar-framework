#!/usr/bin/env python3
"""
Run supplementary experiments for EPAR paper: LongBench, SCROLLS, WMT Zh-En, CodeXGlue.

Protocol: Same as main paper Section 5.1 (110M, seeds 42-46, 5 runs, RoPE/ALiBi/Shaw/Transformer-XL/Ours).
Usage:
  python run_supplementary_experiments.py --benchmark longbench [--subset narrativeqa] [--quick]
  python run_supplementary_experiments.py --benchmark scrolls --task gov_report
  python run_supplementary_experiments.py --benchmark wmt_zh_en
  python run_supplementary_experiments.py --benchmark codexglue
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root so we can import position_attention_model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_protocol():
    config_path = Path(__file__).resolve().parent / "configs" / "paper_protocol.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_longbench(args):
    """LongBench: long-context benchmark. Data from THUDM/LongBench or HF."""
    protocol = load_protocol()
    config_path = Path(__file__).resolve().parent / "longbench" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    subsets = (args.subset.split(",") if args.subset else cfg.get("subsets", ["narrativeqa"]))
    if args.quick:
        subsets = subsets[:1]
    print(f"[LongBench] Protocol: {protocol['model']}; subsets: {subsets}")
    # TODO: Load dataset (e.g. from datasets or LongBench repo), build 110M model with each position encoding, train/eval, aggregate over seeds
    # try:
    #     from datasets import load_dataset
    #     # ds = load_dataset("THUDM/LongBench", subset) or use local LongBench script
    # except Exception as e:
    #     print("Install: pip install datasets; LongBench may require clone from https://github.com/THUDM/LongBench", e)
    results_dir = Path(__file__).resolve().parent / "results" / "longbench"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "longbench",
        "subsets": subsets,
        "protocol": protocol,
        "baselines": ["standard", "rope", "alibi", "shaw", "transformer_xl", "ours_basic", "ours_enhanced"],
        "results": "TODO: fill after running training/eval per baseline",
    }
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[LongBench] Placeholder results written to {results_dir / 'results.json'}")
    return out


def run_scrolls(args):
    """SCROLLS: tau/scrolls on HuggingFace."""
    protocol = load_protocol()
    config_path = Path(__file__).resolve().parent / "scrolls" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tasks = [args.task] if args.task else (cfg.get("tasks", ["gov_report"])[:1] if args.quick else cfg.get("tasks", ["gov_report"]))
    print(f"[SCROLLS] Protocol: {protocol['model']}; tasks: {tasks}")
    # TODO: load_dataset("tau/scrolls", task), same model loop, save per-baseline table
    results_dir = Path(__file__).resolve().parent / "results" / "scrolls"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "scrolls",
        "tasks": tasks,
        "protocol": protocol,
        "baselines": ["standard", "rope", "alibi", "shaw", "transformer_xl", "ours_basic", "ours_enhanced"],
        "results": "TODO: fill after running",
    }
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[SCROLLS] Placeholder results written to {results_dir / 'results.json'}")
    return out


def run_wmt_zh_en(args):
    """WMT Zh-En: non-English machine translation."""
    protocol = load_protocol()
    config_path = Path(__file__).resolve().parent / "wmt_zh_en" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[WMT Zh-En] Protocol: {protocol['model']}")
    # TODO: load_dataset("wmt14", "zh-en") or wmt19, same protocol, BLEU per baseline
    results_dir = Path(__file__).resolve().parent / "results" / "wmt_zh_en"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "wmt_zh_en",
        "protocol": protocol,
        "baselines": ["standard", "rope", "alibi", "shaw", "transformer_xl", "ours_basic", "ours_enhanced"],
        "results": "TODO: fill after running",
    }
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[WMT Zh-En] Placeholder results written to {results_dir / 'results.json'}")
    return out


def run_codexglue(args):
    """CodeXGlue: code summarization or completion."""
    protocol = load_protocol()
    config_path = Path(__file__).resolve().parent / "codexglue" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[CodeXGlue] Protocol: {protocol['model']}; subtask: {cfg.get('subtask', 'code-summarization')}")
    # TODO: load CodeXGlue code summarization, same protocol, ROUGE-L/BLEU per baseline
    results_dir = Path(__file__).resolve().parent / "results" / "codexglue"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "codexglue",
        "protocol": protocol,
        "baselines": ["standard", "rope", "alibi", "shaw", "transformer_xl", "ours_basic", "ours_enhanced"],
        "results": "TODO: fill after running",
    }
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[CodeXGlue] Placeholder results written to {results_dir / 'results.json'}")
    return out


def main():
    parser = argparse.ArgumentParser(description="EPAR supplementary experiments (LongBench, SCROLLS, WMT Zh-En, CodeXGlue)")
    parser.add_argument("--benchmark", choices=["longbench", "scrolls", "wmt_zh_en", "codexglue"], required=True)
    parser.add_argument("--subset", type=str, default=None, help="LongBench: comma-separated subsets")
    parser.add_argument("--task", type=str, default=None, help="SCROLLS: single task name")
    parser.add_argument("--quick", action="store_true", help="Minimal run (one subset/task, for testing)")
    args = parser.parse_args()

    if args.benchmark == "longbench":
        run_longbench(args)
    elif args.benchmark == "scrolls":
        run_scrolls(args)
    elif args.benchmark == "wmt_zh_en":
        run_wmt_zh_en(args)
    elif args.benchmark == "codexglue":
        run_codexglue(args)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
