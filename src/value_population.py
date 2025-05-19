from utils import load_config, extract_schema, load_data, load_checkpoint, save_checkpoint
from model import Model
from prompt import get_value_population_prompt_TS, get_value_population_prompt_TST_L, get_value_population_prompt_TST
from value_identification import get_superkey
import logging
import argparse
import json
import os

def value_population_TS(params):
    model = params["model"]
    model_name = params["model_name"]
    data = params["data"]
    logger = params["logger"]
    results_path = params["results_path"]
    datapath = params["datapath"]
    
    s = f"{results_path}{method}/{datapath}.json"
    head, tail = s.rsplit("/", 1)
    os.makedirs(head, exist_ok=True)
    updated_json = load_checkpoint(f"{head}/{model_name}-checkpoint.pkl")
    processed_idx = len(updated_json)
    for idx,text in enumerate(data['texts']):
        if idx < processed_idx:
            continue
        
        schema = data['schema'][idx]   
        schema = extract_schema(schema)
        schema = eval(schema)
        

        entry = {}
        entry['text'] = text
        entry['schema'] = schema
        entry['domain'] = data['domain'][idx]
        entry['difficulty'] = data['difficulty'][idx]
        entry['ground_truth_entities'] = data['ground_truth_entities'][idx]
        entry['ground_truth_key_value'] = data['ground_truth_key_value'][idx]
        entry['identified_values'] = data['identified_values'][idx]
        

        system_prompt, user_prompt = get_value_population_prompt_TS(entry)

        output = model.predict(system_prompt, user_prompt)
        entry['output'] = output

        
        logger.info("**************************************************")
        logger.info(f"Text: {text}")
        logger.info("--------------------------------------------------")
        logger.info(f"Schema: {schema}")
        logger.info("--------------------------------------------------")
        logger.info("System Prompt:")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info(system_prompt)
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info("User Prompt:")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info(user_prompt)
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info("Output:")
        logger.info("--------------------------------------------------")
        logger.info(f"{output}")
        logger.info("--------------------------------------------------")       
        logger.info("**************************************************")
        # add domain, difficulty, ground_truth_entities, ground_truth_key_value to updated_json
        
        updated_json.append(entry)
        save_checkpoint(f"{head}/{model_name}-checkpoint.pkl",updated_json)
        with open(s, "w") as f:
            json.dump(updated_json, f, indent=2)

def value_population_TST(params):
    model = params["model"]
    model_name = params["model_name"]
    data = params["data"]
    logger = params["logger"]
    results_path = params["results_path"]
    datapath = params["datapath"]
    
    s = f"{results_path}{method}/{datapath}.json"
    head, tail = s.rsplit("/", 1)
    os.makedirs(head, exist_ok=True)
    updated_json = load_checkpoint(f"{head}/{model_name}-checkpoint.pkl")
    processed_idx = len(updated_json)
    for idx,text in enumerate(data['texts']):
        if idx < processed_idx:
            continue
        
        schema = data['schema'][idx]   
        schema = extract_schema(schema)
        schema = eval(schema)
        superkey = get_superkey(schema)
        identified_values = data['identified_values'][idx]
        domain = data['domain'][idx]
        difficulty = data['difficulty'][idx]
        ground_truth_entities = data['ground_truth_entities'][idx]
        ground_truth_key_value = data['ground_truth_key_value'][idx]
        

        entry = {}
        entry['text'] = text
        entry['schema'] = schema
        entry['superkey'] = superkey
        entry['identified_values'] = identified_values
        entry['domain'] = domain
        entry['difficulty'] = difficulty
        entry['ground_truth_entities'] = ground_truth_entities
        entry['ground_truth_key_value'] = ground_truth_key_value
        

        system_prompt, user_prompt = get_value_population_prompt_TST(entry)

        output = model.predict(system_prompt, user_prompt)
        entry['output'] = output

        
        logger.info("**************************************************")
        logger.info(f"Text: {text}")
        logger.info("--------------------------------------------------")
        logger.info(f"Schema: {schema}")
        logger.info("--------------------------------------------------")
        logger.info("System Prompt:")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info(system_prompt)
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info("User Prompt:")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info(user_prompt)
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info("Output:")
        logger.info("--------------------------------------------------")
        logger.info(f"{output}")
        logger.info("--------------------------------------------------")       
        logger.info("**************************************************")
        # add domain, difficulty, ground_truth_entities, ground_truth_key_value to updated_json
        
        updated_json.append(entry)
        save_checkpoint(f"{head}/{model_name}-checkpoint.pkl",updated_json)
        with open(s, "w") as f:
            json.dump(updated_json, f, indent=2)

def value_population_TST_L(params):
    model = params["model"]
    model_name = params["model_name"]
    data = params["data"]
    logger = params["logger"]
    results_path = params["results_path"]
    datapath = params["datapath"]
    
    s = f"{results_path}{method}/{datapath}.json"
    head, tail = s.rsplit("/", 1)
    os.makedirs(head, exist_ok=True)
    updated_json = load_checkpoint(f"{head}/{model_name}-checkpoint.pkl")
    processed_idx = len(updated_json)

    # print(f"Processed index: {processed_idx}")
    # print(f"Total entries: {len(data['texts'])}")

    for idx,text in enumerate(data['texts']):
        if idx < processed_idx:
            continue
        
        schema = data['schema'][idx]   
        schema = extract_schema(schema)
        schema = eval(schema)
        superkey = get_superkey(schema)
        identified_values = data['identified_values'][idx]
        domain = data['domain'][idx]
        difficulty = data['difficulty'][idx]
        ground_truth_entities = data['ground_truth_entities'][idx]
        ground_truth_key_value = data['ground_truth_key_value'][idx]
        

        entry = {}
        entry['text'] = text
        entry['schema'] = schema
        entry['superkey'] = superkey
        entry['identified_values'] = identified_values
        entry['domain'] = domain
        entry['difficulty'] = difficulty
        entry['ground_truth_entities'] = ground_truth_entities
        entry['ground_truth_key_value'] = ground_truth_key_value
        

        system_prompt, user_prompt = get_value_population_prompt_TST_L(entry)

        output = model.predict(system_prompt, user_prompt)
        entry['output'] = output

        
        logger.info("**************************************************")
        logger.info(f"Text: {text}")
        logger.info("--------------------------------------------------")
        logger.info(f"Schema: {schema}")
        logger.info("--------------------------------------------------")
        logger.info("System Prompt:")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info(system_prompt)
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info("User Prompt:")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info(user_prompt)
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        logger.info("Output:")
        logger.info("--------------------------------------------------")
        logger.info(f"{output}")
        logger.info("--------------------------------------------------")       
        logger.info("**************************************************")
        # add domain, difficulty, ground_truth_entities, ground_truth_key_value to updated_json
        
        updated_json.append(entry)
        save_checkpoint(f"{head}/{model_name}-checkpoint.pkl",updated_json)
        with open(s, "w") as f:
            json.dump(updated_json, f, indent=2)

if __name__ == "__main__":
    experiments = load_config("configs/config.yaml")
    config = experiments["value_population"]

    base_data_path = config['base_data_path']
    results_path = config['results_dir']
    logs_path = config['logs_dir']
    os.makedirs(logs_path, exist_ok=True)
    method = config['method']
    num_of_entries = config['num_of_entries']

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Value Population Script')
    parser.add_argument('--method', type=str, default=method, help='TS,TST,TST-L')
    parser.add_argument('--num_of_entries', type=int, default=num_of_entries, help='Number of entries to process')
    parser.add_argument('--model_name', type=str, default=config['model_name'], help='Model name to use')
    parser.add_argument("--dataset", required=False, help="bird")

    # Parse the arguments
    args = parser.parse_args()
    method = args.method
    num_of_entries = args.num_of_entries
    model_name = args.model_name

    if args.dataset=="bird":
        config['datapath'] = config['datapath'].replace("SyntheticText/merged_dataset2","BIRD/bird_dataset")
        config['symbolic_path'] = config['symbolic_path'].replace("SyntheticText/merged_dataset2","BIRD/bird_dataset")

    model = Model(model_name)
    config['datapath'] = config['datapath'].replace("<<model_name>>",model_name)
    symbolic_path = config['symbolic_path']
    datapath = config['datapath']

    logging.basicConfig(
            filename=f'{logs_path}value_population_{method}.log',
            filemode='w',  # or 'a' to append
            level=logging.INFO,
            format='%(message)s'  # Only show the actual log message
        )
    logger = logging.getLogger(__name__)

    print(f"Method: {method}")

    if method == "TS":
        s = symbolic_path
        data = load_data(s,num_of_entries)

        params = {
            "model": model,
            "model_name": model_name,
            "data": data,
            "logger": logger,
            "results_path": results_path,
            "datapath": datapath
        }
        value_population_TS(params)
    elif method == "TST":
        print("Method: TST is running")
        s = symbolic_path
        data = load_data(s,num_of_entries)

        params = {
            "model": model,
            "model_name": model_name,
            "data": data,
            "logger": logger,
            "results_path": results_path,
            "datapath": datapath
        }
        value_population_TST(params)
    elif method == "TST-L":
        print("Method:TST-L is running")
        s = f"{base_data_path}llm/{datapath}.json"
        # print(s)
        data = load_data(s,num_of_entries)

        params = {
            "model": model,
            "model_name": model_name,
            "data": data,
            "logger": logger,
            "results_path": results_path,
            "datapath": datapath
        }
        value_population_TST_L(params)

    
         


    