from utils import load_config, load, load_data, load_checkpoint, save_checkpoint
from model import Model
from prompt import get_baseline_prompt, get_join_query
from database_generation import create_database
import logging
import argparse
import json
import os
import re

def baseline_method(params):
    model_name = params["model_name"]
    model = params["model"]
    data = params["data"]
    logger = params["logger"]
    results_path = params["results_path"]
    datapath = params["datapath"]
    
    s = f"{results_path}/{datapath}/text_cot_{model_name}.json"
    head, tail = s.rsplit("/", 1)
    head = head+"/"+model_name
    os.makedirs(head, exist_ok=True)
    updated_json = load_checkpoint(f"{head}/checkpoint.pkl")
    processed_idx = len(updated_json)
    for idx,text in enumerate(data['texts']):
        # if idx < processed_idx:
        #     continue
        
        

        entry = {}
        entry['text'] = text
        entry['domain'] = data['domain'][idx]
        entry['difficulty'] = data['difficulty'][idx]
        entry['ground_truth_entities'] = data['gt_entities'][idx]
        entry['ground_truth_key_value'] = data['gt_key_value'][idx]
        entry['db_name'] = f"{entry['domain']}_{idx}"
        

        system_prompt, user_prompt = get_baseline_prompt(entry)
        if idx >= processed_idx:
            output = model.predict(system_prompt, user_prompt)
            
            # extract the output wrapped with ```sql to ```
            match = re.search(r"```sql(.*?)```", output, re.DOTALL)
            if match:
                output = match.group(1)
            entry['sql_statements'] = output

            system_prompt2, user_prompt2 = get_join_query(entry)
            output2 = model.predict(system_prompt2, user_prompt2)
            # extract the output wrapped with ```sql to ```
            match = re.search(r"```sql\n(.*?)```\n", output2, re.DOTALL)
            if match:
                output2 = match.group(1)
            entry['join_query'] = output2
        else:
            entry['sql_statements'] = updated_json[idx]['sql_statements']
            entry['join_query'] = updated_json[idx]['join_query']
            output = updated_json[idx]['sql_statements']
            output2 = updated_json[idx]['join_query']
       

        

        db_path = f"databases/{datapath}/text_cot_{model_name}/baseline/{entry['db_name']}"
        head2, tail = db_path.rsplit("/", 1)
        os.makedirs(head2, exist_ok=True)
        create_database(db_path,entry['sql_statements'])

        
        logger.info("**************************************************")
        logger.info(f"Text: {text}")
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
        
        updated_json.append(entry)
        save_checkpoint(f"{head}/checkpoint.pkl",updated_json)
        with open(s, "w") as f:
            json.dump(updated_json, f, indent=2)

if __name__ == "__main__":
    experiments = load_config("configs/config.yaml")
    config = experiments["baseline"]

    
    base_data_path = config['base_data_path']
    results_path = config['results_dir']
    logs_path = config['logs_dir']
    num_of_entries = config['num_of_entries']
    config['method'] = ""

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Baseline method script')
    parser.add_argument('--num_of_entries', type=int, default=num_of_entries, help='Number of entries to process')
    parser.add_argument('--model_name', type=str, default=config['model_name'], help='Model name to use')
    parser.add_argument('--datapath', type=str, default=config['datapath'], help='dataset path')

    # Parse the arguments
    args = parser.parse_args()
    if args.datapath:
        config['datapath'] = args.datapath
    num_of_entries = args.num_of_entries
    model_name = args.model_name

    model = Model(model_name)
    config['datapath'] = config['datapath'].replace("<<model_name>>",model_name)
    datapath = config['datapath']

    os.makedirs(logs_path, exist_ok=True)
    logging.basicConfig(
            filename=f'{logs_path}baseline.log',
            filemode='w',  # or 'a' to append
            level=logging.INFO,
            format='%(message)s'  # Only show the actual log message
        )
    logger = logging.getLogger(__name__)


    
    print("Baseline method is running")
    s = f"{base_data_path}{datapath}.json"
    data = load(config)

    params = {
        "model_name": model_name,
        "model": model,
        "data": data,
        "logger": logger,
        "results_path": results_path,
        "datapath": datapath
    }
    baseline_method(params)

    
         


    