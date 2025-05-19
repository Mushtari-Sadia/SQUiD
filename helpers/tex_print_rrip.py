import json
from collections import defaultdict
from pathlib import Path

# Load the JSON file
with open("results/referential_integrity.json") as f:
    data = json.load(f)

# Parse into structured format: dataset → model → method → score
structured = defaultdict(lambda: defaultdict(dict))

for key, score in data.items():
    try:
        dataset, model, method = key.split("_", 2)
    except ValueError:
        print(f"Skipping malformed key: {key}")
        continue
    structured[dataset][model][method] = score

# Desired method order
# Desired method order without ensemble methods
desired_order = [
    "TS", "TST", "TST-L",
    "ensemble/TS-TST", "ensemble/TS-TST-L", "ensemble/TS-TST-TST-L",
    "baseline"
]

# Extract methods actually present in the data excluding ensembles
existing_methods = {
    key.split("_", 2)[-1] 
    for key in data.keys() 
    # if "ensemble" not in key.split("_", 2)[-1]
}
all_methods = [m for m in desired_order if m in existing_methods]


# Helper function for escaping LaTeX
def escape_latex(s):
    return s.replace('_', r'\_').replace('*', r'\*')

# LaTeX generation
def generate_rrip_table(caption):
    latex = []
    latex.append(r"\begin{tabular}{ll" + "c" * len(all_methods) + "}")
    latex.append(r"\toprule")
    header = ["Dataset", "Model"] + [escape_latex(m) for m in all_methods]
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")

    for dataset in sorted(structured.keys()):
        for model in sorted(structured[dataset].keys()):
            row = [escape_latex(dataset), escape_latex(model)]

            method_scores = {
                method: structured[dataset][model].get(method)
                for method in all_methods
            }
            max_score = max([s for s in method_scores.values() if s is not None], default=None)
            best_methods = {m for m, s in method_scores.items() if s == max_score}

            for method in all_methods:
                score = method_scores.get(method)
                if score is not None:
                    cell = f"{score:.3f}"
                    if method in best_methods:
                        cell = r"\textbf{" + cell + "}"
                else:
                    cell = "--"
                row.append(cell)

            latex.append(" & ".join(row) + r" \\")

        latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    return "\n".join(latex)

# Save the table
table_code = generate_rrip_table("RRIP Score per Model and Method")
Path("tex/rrip_table.tex").write_text(table_code)
print("Saved: tex/rrip_table.tex")
