import json
from collections import defaultdict
from pathlib import Path

model_display_names = {
    "llama": "Llama-8b-Instruct",
    "deepseek": "Deepseek-v2.5",
    "openai": "GPT-4o",
    "mistral": "Mistral-7B",
    "claude": "Claude 3.7 Sonnet",
    "qwen": "Qwen3-8B",
}

# Load your JSON
with open("results/results_all.json") as f:
    data = json.load(f)

# Parse into structured format: dataset → model → domain → method → metrics
structured = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for key, domain_data in data.items():
    try:
        dataset, model, method = key.split("_", 2)
    except ValueError:
        print(f"Skipping malformed key: {key}")
        continue
    for domain, metrics in domain_data.items():
        structured[dataset][model][domain][method] = metrics

# Gather all methods for columns
desired_order = [
    "TS", "TST", "TST-L",
    "ensemble/TS_TST", "ensemble/TS_TST-L", "ensemble/TS_TST_TST-L",
    "baseline"
]

existing_methods = {key.split("_", 2)[-1] for key in data.keys()}
all_methods = [m for m in desired_order if m in existing_methods]

def escape_latex(s):
    return s.replace('_', '\\_')

def generate_table(metric_key, caption):
    latex = []
    latex.append(r"\begin{tabular}{lll" + "c" * len(all_methods) + "}")
    latex.append(r"\toprule")
    header = ["Dataset", "Model", "Domain"] + [escape_latex(m) for m in all_methods]
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")

    for dataset in sorted(structured.keys()):
        for model in sorted(structured[dataset].keys()):
            # Sort domains: put "overall" (as "average") last
            domains = sorted(structured[dataset][model].keys(), key=lambda d: (d.lower() == "overall", d))

            for domain in domains:
                is_overall = domain.lower() == "overall"
                display_domain = "average" if is_overall else domain

                row_lines = []

                if is_overall:
                    row_lines.append(r"\cmidrule(lr){1-" + str(3 + len(all_methods)) + r"}")
                    row_lines.append(r"\rowcolor{gray!20}")

                display_model = model_display_names.get(model, model)
                row = [escape_latex(dataset), escape_latex(display_model), escape_latex(display_domain)]


                # Find best method(s) for this metric
                method_scores = {
                    method: structured[dataset][model][domain][method][metric_key]
                    for method in all_methods
                    if method in structured[dataset][model][domain]
                }
                max_score = max(method_scores.values()) if method_scores else None
                best_methods = {m for m, s in method_scores.items() if abs(s - max_score) < 1e-6}

                for method in all_methods:
                    metrics = structured[dataset][model][domain].get(method)
                    if metrics:
                        val = metrics[metric_key]
                        cell = f"{val:.2f}"
                        if method in best_methods:
                            cell = r"\textbf{" + cell + "}"
                    else:
                        cell = "--"
                    row.append(cell)

                row_lines.append(" & ".join(row) + r" \\")
                latex.extend(row_lines)

            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    return "\n".join(latex)

# Table specs
table_specs = {
    "match_accuracy": "tex/value_coverage_table.tex",
    "match_in_right_column_accuracy": "tex/column_consistency_table.tex",
    "avg_row_accuracy": "tex/row_coverage_table.tex"
}

# Generate and save each table
for metric, filename in table_specs.items():
    caption = metric.replace("_", " ").capitalize() + " per Domain"
    table_code = generate_table(metric, caption)
    Path(filename).write_text(table_code)
    print(f"Saved: {filename}")
