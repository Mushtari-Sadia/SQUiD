import os
import json
from collections import defaultdict

# Directory to search for input files
root_dir = 'results/database_generation'
datapaths = ['BIRD/bird_dataset','SyntheticText/merged_dataset2']
model_names = ['qwen', 'llama', 'claude', 'openai', 'deepseek']

# current_ensemble_combo = ['TS', 'TST']
# current_ensemble_combo = ['TS', 'TST-L']
current_ensemble_combo = ['TS', 'TST', 'TST-L']
ensemble_name = ""
for item in current_ensemble_combo:
    ensemble_name += item + "_"
ensemble_name = ensemble_name[:-1]  # Remove the last underscore
# model name and datapath name has to be same for each ensemble
for datapath in datapaths:
    for model_name in model_names:
        output_path = f"results/database_generation/ensemble/{ensemble_name}/{datapath}/text_cot_{model_name}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Temporary dictionary to accumulate rows and ground truth by db_name
        temp_data = defaultdict(lambda: {'joined_rows': [], 'ground_truth_key_value': None, 'schema': None, 'domain': None})
        for method in current_ensemble_combo:
            file_path = f"results/database_generation/{method}/{datapath}/text_cot_{model_name}.json"
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    for entry in data:
                        db_name = entry.get('db_name')
                        joined_rows = entry.get('joined_rows', [])
                        joined_rows = joined_rows[:15]
                        ground_truth = entry.get('ground_truth_key_value')
                        schema = entry.get('schema')
                        domain = entry.get('domain')

                        if db_name:
                            temp_data[db_name]['joined_rows'].extend(joined_rows)
                            if ground_truth and temp_data[db_name]['ground_truth_key_value'] is None:
                                temp_data[db_name]['ground_truth_key_value'] = ground_truth
                            if schema and temp_data[db_name]['schema'] is None:
                                temp_data[db_name]['schema'] = schema
                            if domain and temp_data[db_name]['domain'] is None:
                                temp_data[db_name]['domain'] = domain
                except json.JSONDecodeError as e:
                    print(f"Skipping {file_path} due to JSON error: {e}")

            # Convert temp_data to list of dicts
        merged_data = [
            {
                'db_name': db_name,
                'joined_rows': values['joined_rows'],
                'ground_truth_key_value': values['ground_truth_key_value'],
                'schema': values['schema'],
                'domain': values['domain']
            }
            for db_name, values in temp_data.items()
        ]

        # Write to output JSON
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)

        print(f"Merged output written to {output_path}")
