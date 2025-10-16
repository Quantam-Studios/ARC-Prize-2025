import json

with open("submission_train.json") as f:
    old = json.load(f)

new = []
for task_id, attempts_list in old.items():
    attempts_dict = attempts_list[0]
    new.append({
        "id": task_id,
        "attempts": [attempts_dict["attempt_1"], attempts_dict["attempt_2"]],
    })

with open("submission_fixed.json", "w") as f:
    json.dump(new, f)