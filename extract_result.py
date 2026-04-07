#!/usr/bin/env python3
"""Extract and summarize results from a VLMEvalKit _result.pkl file."""

import pickle
import argparse
import json
import os
from collections import Counter


def extract_result(pkl_path: str, verbose: bool = False, json_path: str = None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    total = len(data)
    hits = sum(v["hit"] for v in data.values())
    accuracy = hits / total * 100 if total > 0 else 0.0

    print(f"File: {pkl_path}")
    print(f"Total samples : {total}")
    print(f"Correct (hit=1): {hits}")
    print(f"Wrong   (hit=0): {total - hits}")
    print(f"Accuracy        : {accuracy:.2f}%")

    if verbose:
        print("\n--- Per-sample results ---")
        for qid, v in sorted(data.items()):
            status = "CORRECT" if v["hit"] else "WRONG"
            print(f"  [{qid}] {status} | {v['log'].strip()}")

    if json_path is None:
        json_path = os.path.splitext(pkl_path)[0] + ".json"

    output = {
        "file": pkl_path,
        "total": total,
        "correct": hits,
        "wrong": total - hits,
        "accuracy": round(accuracy, 4),
        "samples": {str(qid): v for qid, v in data.items()},
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved to: {json_path}")

    return data, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract results from a _result.pkl file")
    parser.add_argument("--pkl_path", default="/fs/nexus-scratch/yliang17/Research/VLM/VLMEvalKit/outputs/Qwen3-VL-4B-sat_mix_qa_only_10k_e1/Qwen3-VL-4B-sat_mix_qa_only_10k_e1_CV-Bench-2D_chatgpt-0125_result.pkl", help="Path to the _result.pkl file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-sample details")
    parser.add_argument("--json_path", default=None, help="Output JSON path (default: same as pkl with .json extension)")
    args = parser.parse_args()

    extract_result(args.pkl_path, verbose=args.verbose, json_path=args.json_path)
