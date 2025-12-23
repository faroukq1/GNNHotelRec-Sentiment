import pandas as pd
import json

# List of your JSONL files
jsonl_files_skip = [
    "gnn_balanced_results/gat/gat_result_part_1_balanced_ver2.jsonl",
    "gnn_balanced_results/gat/gat_result_part_2_balanced_ver2.jsonl",
    "gnn_balanced_results/gat/gat_result_part_3_balanced_ver2.jsonl",
    "gnn_balanced_results/gat/gat_result_part_4_balanced_ver2.jsonl",
    "gnn_balanced_results/gat/gat_result_part_5_balanced_ver2.jsonl"
]

jsonl_files = [
    "gnn_balanced_results/gat/gat_results_balanced_improved.jsonl"
]

all_results = []
for file in jsonl_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            all_results.append(json.loads(line))

df = pd.DataFrame(all_results)

df.to_csv("gnn_balanced_results/gat/gat_results_balanced_improved.csv", index=False)
print("All results saved to gat_results_balanced_improved.csv")
