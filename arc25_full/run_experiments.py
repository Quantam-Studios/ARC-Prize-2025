import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# --- import the pieces you already have ---
from .solver import solve_all_tasks
from .search import fit_on_pairs, fit_on_pairs_modified
from .grids import Grid
from .tester import compute_accuracy


# ---------------------------------------------------------------------
# Helper to estimate how many transform sequences exist for a given depth
# ---------------------------------------------------------------------
def max_combos_for_depth(depth: int) -> int:
    from .search import library
    lib_size = len(library())
    return sum(lib_size ** i for i in range(1, depth + 1))


# ---------------------------------------------------------------------
# Helper: compute local accuracy on training set
# (direct port of tester.py’s logic)
# ---------------------------------------------------------------------
def compute_accuracy(submission_path: str, ground_truth_dir: str) -> float:
    """Compare solver predictions against known outputs, return accuracy."""
    import json, numpy as np, glob

    with open(submission_path) as f:
        preds = {task_id: [a["attempt_1"] for a in attempts]
                  + [a["attempt_2"] for a in attempts]
                  for task_id, attempts in json.load(f).items()}

    total, correct = 0, 0
    for gt_path in glob.glob(f"{ground_truth_dir}/*.json"):
        task_id = Path(gt_path).stem
        if task_id not in preds:
            continue
        js = json.load(open(gt_path))
        gt_out = js["test"][0]["output"]
        for attempt in preds[task_id]:
            total += 1
            if np.array_equal(np.array(attempt), np.array(gt_out)):
                correct += 1
                break  # stop after first success

    return (correct / len(preds)) * 100 if preds else 0.0


# ---------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------
def run_full_experiment_suite():
    """Run all solver variants across both task sets and depths, with timing + accuracy."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    task_sets = {
        "training": "ARC-AGI-2/data/training",
        "evaluation": "ARC-AGI-2/data/evaluation",
    }
    variants = {
        "baseline": fit_on_pairs,
        "heuristic": fit_on_pairs_modified,
    }
    depths = [2, 3]

    for variant_name, fit_func in variants.items():
        import arc25_full.search as search
        search.fit_on_pairs = fit_func  # monkeypatch the search routine

        for set_name, eval_dir in task_sets.items():
            for depth in depths:
                limit = max_combos_for_depth(depth)
                out_name = f"{variant_name}_{set_name}_d{depth}.json"
                out_path = results_dir / out_name
                timing_path = results_dir / f"{out_name.replace('.json', '_timing_summary.txt')}"

                # 🔹 SKIP check
                if out_path.exists() and timing_path.exists():
                    print(f"⏩ Skipping {out_name} (already complete)")
                    continue

                print("\n" + "=" * 70)
                print(f"RUNNING: variant={variant_name}  dataset={set_name}  depth={depth}")
                print(f"Limit = {limit:,}")
                print("=" * 70)

                # --- timing ---
                start = time.time()
                sub = solve_all_tasks(eval_dir, limit=None, out_path=str(out_path), max_len=depth)
                with open(out_path, "w") as f:
                    json.dump(sub, f)
                end = time.time()

                total_time = end - start
                avg_time = total_time / len(sub)

                # --- accuracy ---
                if set_name == "training":
                    acc = compute_accuracy(str(out_path), eval_dir)
                else:
                    acc = 0.0  # evaluation ground truth unknown

                with open(timing_path, "w") as f:
                    f.write(f"Tasks evaluated: {len(sub)}\n")
                    f.write(f"Depth: {depth}\n")
                    f.write(f"Total time: {total_time:.2f}s\n")
                    f.write(f"Average per task: {avg_time:.2f}s\n")
                    if set_name == "training":
                        f.write(f"Accuracy: {acc:.2f}%\n")
                    else:
                        f.write("Accuracy: N/A (hidden evaluation set)\n")

                print(f"Wrote {out_path} with {len(sub)} tasks.")
                print(f"Timing and accuracy summary written to {timing_path}\n")


if __name__ == "__main__":
    run_full_experiment_suite()