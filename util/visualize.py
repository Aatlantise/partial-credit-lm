import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_blimp_results(results_dir):
    all_data = []

    # Categories mapped to task prefixes (Simplified for BabyLM)
    categories = {
        "Agreement": ["adjunct_island", "distractor", "subject_verb"],
        "Morphology": ["irregular", "regular"],
        "Semantics": ["animate", "cause_effect"],
        "Arg Structure": ["transitive", "intransitive", "passive"],
        "Binding": ["principle_a"],
    }

    for filename in sorted(os.listdir(results_dir), key=lambda x: float(x.replace(".json", ""))):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                model_name = filename.replace(".json","")

                # Extract mean accuracy across all BLiMP tasks
                # Note: path may vary slightly by harness version (results or results['tasks'])
                res_dict = data.get("results", data)

                for task, metrics in res_dict.items():
                    if "blimp" in task:
                        acc = metrics.get("acc,none", metrics.get("acc"))
                        all_data.append({
                            "Model": model_name,
                            "Task": task,
                            "Accuracy": acc
                        })

    return pd.DataFrame(all_data)


# Run the parser
df = parse_blimp_results("../lm-evaluation-harness/eval_results")
df.to_csv("blimp_results.csv")

# Plotting the comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Model", y="Accuracy", ci="sd")
plt.xlabel("Epsilon")
plt.ylabel("BLiMP Accuracy")
plt.title("Mean BLiMP Accuracy Across Models")
plt.ylim(0, 1)  # Standard range for 10M token models
plt.axhline(0.5, ls='--', color='red', label="Random Chance")
plt.legend()
plt.show()