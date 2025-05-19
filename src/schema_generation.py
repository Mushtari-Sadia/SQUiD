import argparse
import sys
import os
import numpy as np
import re
import json

from utils import load_config, load
from model import Model
from prompt import *


def log(log_file,system_prompt,user_prompt,schema):
    print("System Prompt:",file=log_file)
    print(system_prompt,file=log_file)
    print("User Prompt:",file=log_file)
    print(user_prompt,file=log_file)
    print("Output:",file=log_file)
    print("@"*80,file=log_file)
    print(schema,file=log_file)
    print("@"*80,file=log_file)
    print("*"*80,file=log_file)



def generate_schema(config):
    """
    parameters:
    prompt: 'cot' or 'direct'
    method: 'triplet' or 'text'
    model: model object
    datapath: dataset path string

    returns:
    None
    """


    method = config['method']
    model = Model(config['model_name'])
    datapath = config['datapath']
    base_data_path = config['base_data_path']
    prompt_type = config['prompt_type']

    data = load(config)

    updated_json = []
    domain_to_schema = {}
    idx_for_domain = {}
    for idx,text in enumerate(data["texts"]):
        if data["domain"][idx] not in idx_for_domain:
            idx_for_domain[data["domain"][idx]] = []
        idx_for_domain[data["domain"][idx]].append(idx)
    # create directory if doesnt exist {config['results_dir']}/{datapath}
    os.makedirs(os.path.dirname(config['results_dir']), exist_ok=True)
    os.makedirs(os.path.dirname(config['results_dir'] + f"{datapath}/"), exist_ok=True)

    results_path = os.path.join(config['results_dir'], f"{datapath}/{method}_{prompt_type}_{model.llm}.json")
    schema_mapping = os.path.join(config['results_dir'], f"{datapath}/{method}_{prompt_type}_{model.llm}_schema.json")
    # load the json file if already exists
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            updated_json = json.load(f)
    # if os.path.exists(schema_mapping):
    #     with open(schema_mapping, "r") as f:
    #         domain_to_schema = json.load(f)
    
    processed_idx = len(updated_json)
    num_of_entries = len(data["texts"])
    log_path = config['logs_dir']
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, f"schema_with_{method}_{prompt_type}_{model.llm}.txt")
    
    with open(log_path, 'w') as log_file:
        for idx,text in enumerate(data["texts"]):
            # check if the idx is less than the length of updated_json
            # if idx < processed_idx:
            #     continue
            
            for j in range(5):
                next_entry_same_domain = idx_for_domain[data["domain"][idx]][j]
                if next_entry_same_domain != idx:
                    text += data["texts"][next_entry_same_domain]

            updated_dict = {}
            if prompt_type=="cot":
                system_prompt, user_prompt = get_schema_prompt_with_cot_and_text(text)
            else:
                system_prompt, user_prompt = get_schema_prompt_with_direct_and_text(text)
                    
            schema = ""
            if data["domain"][idx] not in domain_to_schema:
                domain_to_schema[data["domain"][idx]] = ""
            if domain_to_schema[data["domain"][idx]] != "":
                schema = domain_to_schema[data["domain"][idx]]
            else:
                schema = model.predict(system_prompt,user_prompt) 
                match = re.search(r"```python\n(.*?)```\n", schema, re.DOTALL)
                if match:
                    schema = match.group(1)
                domain_to_schema[data["domain"][idx]] = schema
            
            updated_dict['text'] = data["texts"][idx]
            updated_dict['ground_truth_entities'] = data["gt_entities"][idx]
            updated_dict['ground_truth_key_value'] = data["gt_key_value"][idx]
            updated_dict['domain'] = data["domain"][idx]
            updated_dict['difficulty'] = data["difficulty"][idx]
            updated_dict['predicted_schema'] = schema
            updated_json.append(updated_dict)
            log(log_file,system_prompt,user_prompt,schema)

            

        with open(results_path, "w") as f2:
            json.dump(updated_json, f2, indent=4)
        with open(schema_mapping, "w") as f3:
            json.dump(domain_to_schema, f3, indent=4)


if __name__ == "__main__":
    experiments = load_config("configs/config.yaml")
    config = experiments['schema_generation']
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="llama or openai or deepseek")
    parser.add_argument("--prompt_type", required=True, help="cot or direct")
    parser.add_argument("--method", required=True, help="text or triplet")
    args = parser.parse_args()

    config['model_name'] = args.model_name
    config['prompt_type'] = args.prompt_type
    config['method'] = args.method
    
    generate_schema(config)

        
