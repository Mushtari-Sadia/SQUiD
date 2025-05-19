
import os
import json
from collections import defaultdict

methods_remap = {"TS":"TS","TST":"TST","TST-L":"TST-L","baseline":"baseline","ensemble/TS_TST":"ensemble/TS-TST","ensemble/TS_TST-L":"ensemble/TS-TST-L","ensemble/TS_TST_TST-L":"ensemble/TS-TST-TST-L"}
def compute_average(data,domains):
    domain_scores = defaultdict(lambda: {"match_accuracy": [], "match_in_right_column_accuracy": [], "avg_row_accuracy": []})
    overall_scores = {"match_accuracy": [], "match_in_right_column_accuracy": [], "avg_row_accuracy": []}

    # Process each entry
    for key, metrics in data.items():
        domain = key.split("_")[0]
        if domain not in domains:
            continue
        for metric in ["match_accuracy", "match_in_right_column_accuracy", "avg_row_accuracy"]:
            domain_scores[domain][metric].append(metrics[metric])
            overall_scores[metric].append(metrics[metric])

    # Compute averages
    def compute_avg(scores):
        return {k: sum(v) / len(v) for k, v in scores.items()}

    domain_averages = {domain: compute_avg(scores) for domain, scores in domain_scores.items()}
    domain_averages["overall"] = compute_avg(overall_scores)
    return domain_averages

def compute_correctly_produced_join_rows(data,domains):
    domain_scores = {}

    # Process each entry
    for key, metrics in data.items():
        domain = key.split("_")[0]
        if domain not in domains:
            continue
        if domain not in domain_scores:
            domain_scores[domain] = 0
        if metrics["match_accuracy"] != 0:
            domain_scores[domain] += 1

    return domain_scores

def compute_average_hallucination(data,domains):
    domain_scores = defaultdict(lambda: [])
    overall_scores = []

    # Process each entry
    for key, hal in data.items():
        domain = key.split("_")[0]
        if domain not in domains:
            continue
        if hal < 0:
            continue
        domain_scores[domain].append(hal)
        overall_scores.append(hal)

    # Compute averages
    def compute_avg(scores):
        return {k: sum(v) / len(v) for k, v in scores.items()}

    domain_averages = {domain: sum(scores) / len(scores) for domain, scores in domain_scores.items()}
    if len(overall_scores) > 0:
        domain_averages["overall"] = sum(overall_scores) / len(overall_scores)
    else:
        domain_averages["overall"] = 0
    return domain_averages


def compute_results_all():
    methods = ["TS","TST","TST-L","baseline","ensemble/TS_TST","ensemble/TS_TST-L","ensemble/TS_TST_TST-L"]
    datasets = ["BIRD/bird_dataset","SyntheticText/merged_dataset2"]
    models = ["llama","claude","openai","deepseek","qwen"]
    domains = {"BIRD/bird_dataset": ["california","authors","superhero","books","computer","mental"], "SyntheticText/merged_dataset2": ["finance", "tourism", "education"]}
    # result_types = ["results","hallucination"]

    overall = {}

    for dataset in datasets:
        for model in models:
            for method in methods:
                    overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = None
                    results_path = f"results/database_generation/{method}/{dataset}/text_cot_{model}/evaluation/results.json"
                    if os.path.exists(results_path):
                        with open(results_path) as f:
                            data = json.load(f)
                            overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = compute_average(data,domains[dataset])
                    else:
                        print(f"Results file not found: {results_path}")
    return overall

# def compute_results_hallucination():
#     methods = ["TS","TST","TST-L","baseline"]
#     datasets = ["BIRD/bird_dataset","SyntheticText/merged_dataset2"]
#     models = ["llama","claude","openai","deepseek","qwen"]
#     domains = {"BIRD/bird_dataset": ["california","authors","superhero","books","computer","mental"], "SyntheticText/merged_dataset2": ["finance", "tourism", "education"]}
#     # result_types = ["results","hallucination"]

#     overall = {}

#     for dataset in datasets:
#         for model in models:
#             for method in methods:
#                     overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = None
#                     results_path = f"results/database_generation/{method}/{dataset}/text_cot_{model}/evaluation/hallucination.json"
#                     if os.path.exists(results_path):
#                         with open(results_path) as f:
#                             data = json.load(f)
#                             overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = compute_average_hallucination(data,domains[dataset])
#     return overall

import os
import json

def compute_results_hallucination():
    methods = ["TS", "TST", "TST-L", "baseline"]
    datasets = ["BIRD/bird_dataset", "SyntheticText/merged_dataset2"]
    models = ["llama", "claude", "openai", "deepseek", "qwen"]
    domains = {
        "BIRD/bird_dataset": ["california", "authors", "superhero", "books", "computer", "mental"],
        "SyntheticText/merged_dataset2": ["finance", "tourism", "education"]
    }

    overall = {}

    for dataset in datasets:
        domain_keys = domains[dataset]
        for model in models:
            key_prefix = f"{dataset.split('/')[0]}_{model}"
            raw_data_per_method = {}

            # Step 1: Load raw hallucination data for each method
            for method in methods:
                results_path = f"results/database_generation/{method}/{dataset}/text_cot_{model}/evaluation/hallucination.json"
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        data = json.load(f)
                        raw_data_per_method[method] = data
                else:
                    raw_data_per_method[method] = {}

            # Step 2: Find common keys across all methods and filter out any with value -1
            common_keys = set.intersection(*(set(data.keys()) for data in raw_data_per_method.values()))
            filtered_keys = [
                k for k in common_keys
                if all(raw_data_per_method[m].get(k, -1) != -1 for m in methods)
            ]

            # Step 3: Compute average only over the filtered, valid keys
            for method in methods:
                filtered_data = {k: raw_data_per_method[method][k] for k in filtered_keys}
                avg_data = compute_average_hallucination(filtered_data, domain_keys)
                overall[f"{key_prefix}_{method}"] = avg_data

    return overall


def compute_correctly_produced_joined_rows_all():
    # methods = ["TS","TST","TST-L","baseline","ensemble/TS_TST","ensemble/TS_TST-L","ensemble/TS_TST_TST-L"]
    methods = ["baseline","ensemble/TS_TST_TST-L"]
    datasets = ["BIRD/bird_dataset","SyntheticText/merged_dataset2"]
    models = ["llama","claude","openai","deepseek","qwen"]
    domains = {"BIRD/bird_dataset": ["california","authors","superhero","books","computer","mental"], "SyntheticText/merged_dataset2": ["finance", "tourism", "education"]}
    # result_types = ["results","hallucination"]

    overall = {}

    for dataset in datasets:
        for model in models:
            for method in methods:
                    overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = None
                    results_path = f"results/database_generation/{method}/{dataset}/text_cot_{model}/evaluation/results.json"
                    if os.path.exists(results_path):
                        with open(results_path) as f:
                            data = json.load(f)
                            overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = compute_correctly_produced_join_rows(data,domains[dataset])
    return overall


def compute_referential_integrity_pctg(data,domains):
    total_ref_int_pctg = 0
    count_non_db = 0
    total_count_none = 0

    for item in data:
        db_name = item.get('db_name')
        if db_name:
            # Check if the domain is in the list of domains
            if db_name.split("_")[0] not in domains:
                continue

            # Check if the referential integrity is satisfied
            joined_rows = item.get('joined_rows', [])
            ref_int_pctg = 0
            for row in joined_rows:
                # print(row)
                count_none = sum(1 for value in row.values() if value is None)
                total_count_none += count_none
                count_total = len(row)
                # print(count_total)
                if count_total > 0:
                    ref_int_pctg += (count_total - count_none) / count_total
            if len(joined_rows) > 0:
                ref_int_pctg /= len(joined_rows)
            else:
                count_non_db += 1
                ref_int_pctg = 0
            
            total_ref_int_pctg += ref_int_pctg
    avg_ref_int_pctg = total_ref_int_pctg / len(data) if len(data) > 0 else 0
    return avg_ref_int_pctg

def compute_referential_integrity_all():
    # methods = ["TS","TST","TST-L","baseline","ensemble/TS_TST","ensemble/TS_TST-L","ensemble/TS_TST_TST-L"]
    methods = ["baseline","ensemble/TS_TST_TST-L"]
    
    datasets = ["BIRD/bird_dataset","SyntheticText/merged_dataset2"]
    # datasets = ["BIRD/bird_dataset"]
    models = ["llama","claude","openai","deepseek","qwen"]
    # models = ["claude","llama"]
    domains = {"BIRD/bird_dataset": ["california","authors","superhero","books","computer","mental"], "SyntheticText/merged_dataset2": ["finance", "tourism", "education"]}
    # result_types = ["results","hallucination"]

    overall = {}

    for dataset in datasets:
        for model in models:
            for method in methods:
                    overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = None
                    results_path = f"results/database_generation/{method}/{dataset}/text_cot_{model}.json"
                    if os.path.exists(results_path):
                        with open(results_path) as f:
                            data = json.load(f)
                            overall[f"{dataset.split("/")[0]}_{model}_{methods_remap[method]}"] = compute_referential_integrity_pctg(data,domains[dataset])
    return overall

if __name__ == "__main__":
    # Compute results for all methods
    results_all = compute_results_all()
    with open("results/results_all.json", "w") as f:
        json.dump(results_all, f, indent=2)

    # Compute results for hallucination
    results_hallucination = compute_results_hallucination()
    with open("results/results_hallucination.json", "w") as f:
        json.dump(results_hallucination, f, indent=2)

    # Compute correctly produced joined rows
    correctly_produced_joined_rows = compute_correctly_produced_joined_rows_all()
    with open("results/correctly_produced_joined_rows.json", "w") as f:
        json.dump(correctly_produced_joined_rows, f, indent=2)

    # Compute referential integrity
    referential_integrity = compute_referential_integrity_all()
    with open("results/referential_integrity.json", "w") as f:
        json.dump(referential_integrity, f, indent=2)