import json, numpy as np, glob

with open("submission_fixed_train.json") as f:
    preds = {d["id"]: d["attempts"] for d in json.load(f)}

total = 0
correct = 0

for gt_path in glob.glob("ARC-AGI-2/data/training/*.json"):
    task_id = gt_path.split("\\")[-1].split(".")[0]
    if task_id in preds:
        total += 1
        gt = json.load(open(gt_path))["test"][0]["output"]

        ok = any(np.array_equal(np.array(a), np.array(gt)) for a in preds[task_id])
        if ok:
            correct += 1
        print(task_id, "✅" if ok else "❌")

if total > 0:
    accuracy = correct / total * 100
    print(f"\nSolved {correct} / {total} tasks ({accuracy:.2f}% accuracy)")
else:
    print("No matching task IDs found in predictions.")