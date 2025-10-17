"""
tester.py  –  reusable accuracy evaluator
-----------------------------------------
Usage:
    python -m arc25_full.tester submission.json ARC-AGI-2/data/training
"""
import sys
import json
import numpy as np
import glob
from pathlib import Path


def compute_accuracy(submission_path: str, ground_truth_dir: str, verbose: bool = False) -> tuple:
    """Return (correct, total, percent) comparing submission outputs to ground truth."""
    with open(submission_path) as f:
        preds_data = json.load(f)

    # convert to id → [attempt_1, attempt_2, ...]
    preds = {tid: [a["attempt_1"], a["attempt_2"]] for tid, a in preds_data.items()}

    total = 0
    correct = 0

    for gt_path in glob.glob(str(Path(ground_truth_dir) / "*.json")):
        task_id = Path(gt_path).stem
        if task_id not in preds:
            continue
        total += 1
        gt = json.load(open(gt_path))["test"][0]["output"]
        ok = any(np.array_equal(np.array(a), np.array(gt)) for a in preds[task_id])
        if ok:
            correct += 1
        if verbose:
            print(f"{task_id} {'✅' if ok else '❌'}")

    acc = (correct / total * 100) if total else 0.0
    return correct, total, acc


# --- CLI entry point ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m arc25_full.tester <submission.json> <path/to/data/training>")
        sys.exit(1)

    sub_path, data_dir = sys.argv[1:]
    correct, total, acc = compute_accuracy(sub_path, data_dir, verbose=True)
    if total:
        print(f"\nSolved {correct}/{total} tasks ({acc:.2f}% accuracy)")
    else:
        print("No matching task IDs found in predictions.")