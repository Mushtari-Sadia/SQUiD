import json
from pathlib import Path

def generate_latex_table(data):
    # Define the divisor for the BIRD and Synthetic datasets
    bird_divisor = 12
    synthetic_divisor = 24

    # Extract datasets, models, and methods from the keys
    datasets = set()
    models = set()
    methods = set()
    
    # Parse JSON keys to get datasets, models, and methods
    for key in data.keys():
        # Skip entries with 'ensemble' in the key
        # if "ensemble" in key:
        #     continue
        dataset, model, method = key.split("_")
        datasets.add(dataset)
        models.add(model)
        methods.add(method)
    
    datasets = sorted(datasets)  # Sort for consistency
    models = sorted(models)      # Sort for consistency
    methods = sorted(methods)    # Sort for consistency
    
    # Prepare LaTeX tables for each dataset
    latex_tables = {}
    for dataset in datasets:
        # LaTeX table header for the current dataset
        latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|" + "l|" * len(methods) + "}\n"
        latex_table += "\\hline\n"
        
        # Add header: methods
        latex_table += "Model & " + " & ".join(methods) + " \\\\ \\hline\n"
        
        # Choose the divisor based on the dataset
        divisor = bird_divisor if dataset == "BIRD" else synthetic_divisor
        
        # Iterate over each model and create rows for the table
        for model in models:
            latex_table += model  # First column is model name
            for method in methods:
                # Get the value for each model_method combination
                key = f"{dataset}_{model}_{method}"
                raw_value = data.get(key, {})  # Each entry is a dict of domains
                if isinstance(raw_value, dict):
                    value = sum(raw_value.values())
                    count = len(raw_value)
                else:
                    value = raw_value
                percentage_value = round((value / (divisor*count)) * 100, 2)

                latex_table += f" & {percentage_value}"
            latex_table += " \\\\ \\hline\n"
        
        latex_table += "\\end{tabular}\n\\caption{Percentage of databases produced correctly (" + dataset + ")}\n\\end{table}\n"
        
        latex_tables[dataset] = latex_table
    
    return latex_tables

# Read the JSON file
with open('results/correctly_produced_joined_rows.json', 'r') as f:
    data = json.load(f)

# Define the filename for the output LaTeX file
filename = "tex/pctg_of_correct_dbs.tex"

# Generate the LaTeX tables
latex_tables = generate_latex_table(data)

# Write the LaTeX tables to the file
with open(filename, 'w') as f:
    for dataset, table in latex_tables.items():
        f.write(table + "\n\n")

print(f"Saved: {filename}")
