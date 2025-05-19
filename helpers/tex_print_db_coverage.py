import csv
from collections import defaultdict

def generate_latex_table(input_csv_path, output_tex_path, methods=None):
    # Ordered list of methods to include (and their order in the table)
    default_methods = [
        'TS', 'TST', 'TST-L',
        'ensemble_TS_TST', 'ensemble_TS_TST_L', 'ensemble_TS_TST_TST_L',
        'baseline'
    ]
    if methods is None:
        methods = default_methods

    # Map method to its corresponding column names
    method_cols = {
        'TS': ['TS_rc', 'TS_vc', 'TS_cc'],
        'TST': ['TST_rc', 'TST_vc', 'TST_cc'],
        'TST-L': ['TST-L_rc', 'TST-L_vc', 'TST-L_cc'],
        'ensemble_TS_TST': ['ensemble_TS_TST_rc', 'ensemble_TS_TST_vc', 'ensemble_TS_TST_cc'],
        'ensemble_TS_TST_L': ['ensemble_TS_TST_L_rc', 'ensemble_TS_TST_L_vc', 'ensemble_TS_TST_L_cc'],
        'ensemble_TS_TST_TST_L': ['ensemble_TS_TST_TST_L_rc', 'ensemble_TS_TST_TST_L_vc', 'ensemble_TS_TST_TST_L_cc'],
        'baseline': ['baseline_rc', 'baseline_vc', 'baseline_cc'],
    }

    # Read and group CSV rows by model
    grouped_rows = defaultdict(list)
    with open(input_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            grouped_rows[row['model']].append(row)

    # Header: column alignment
    alignments = 'll' + 'ccc' * len(methods)
    header = r"""
\begin{tabular}{""" + alignments + "}\n" + r"""\toprule
& & """ + ' & '.join([f"\\multicolumn{{3}}{{c}}{{{method.replace('_', r'\_')}}}" for method in methods]) + r""" \\""" + "\n" + \
"Model & Difficulty & " + " & ".join(["RC & VC & CC"] * len(methods)) + r" \\" + "\n" + r"\midrule"

    def fmt(val):
        try:
            return f"{float(val):.2f}"
        except ValueError:
            return val

    # Rows
    latex_rows = []
    for model, rows in grouped_rows.items():
        for i, row in enumerate(rows):
            cells = []
            if i == 0:
                cells.append(rf"\multirow{{{len(rows)}}}{{*}}{{{model}}}")
            else:
                cells.append("")

            cells.append(row['difficulty'])

            for method in methods:
                for col in method_cols[method]:
                    cells.append(fmt(row[col]))

            latex_rows.append(" & ".join(cells) + r" \\")

    # Footer
    footer = r"""\bottomrule
\end{tabular}"""

    # Write to file
    with open(output_tex_path, 'w') as texfile:
        texfile.write(header + '\n')
        texfile.write('\n'.join(latex_rows) + '\n')
        texfile.write(footer + '\n')

# ✅ Usage with correct method order
# generate_latex_table("tex/scores.csv", "tex/db_coverage_scores_beautified.tex")

# ✅ Usage excluding TST-L and baseline
# generate_latex_table("tex/scores.csv", "tex/db_filtered_scores.tex", methods=['ensemble_TS_TST_TST_L','baseline'])
generate_latex_table("tex/scores.csv", "tex/db_filtered_scores_2.tex", methods=['TS', 'TST', 'TST-L',
        'ensemble_TS_TST', 'ensemble_TS_TST_L', 'ensemble_TS_TST_TST_L'])
