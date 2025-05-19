import re
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
import re
import random
import numpy as np
random.seed(42)
np.random.seed(42)
import json
import yaml
import re
import math
import os
import json_repair

def extract_from_output(output: str, schema: dict):
    pyschema = sqlschema_to_pythonschema(schema)
    ret_dict = {}
    # read the output line by line
    for line in output.split("\n"):
        if line.startswith("extract"):
            # print(line)
            extract(ret_dict, line, pyschema)
    return ret_dict

def sqlschema_to_pythonschema(schema):
    """
    Convert a SQL schema to a Python schema. return a dictionary of table name to list of column names and types.
    """
    sqltype_to_python_type = {
            "INTEGER": int,
            "TEXT": str,
            "REAL": float,
            "BOOLEAN": bool,
            "DATE": str,
            "DATETIME": str,
            "DECIMAL(10,2)": float,
            "FLOAT": float
        }
    
    python_schema = {}
    for table in schema:
        table_name = table["table_name"]
        columns = {}
        for col in table["columns"]:
            # columns.append(f"{col['name']}: {sqltype_to_python_type[col['type']]}")
            try:
                columns[col['name']] = sqltype_to_python_type[col['type']]
            except:
                # if the type is not in the sqltype_to_python_type, keep it as string
                columns[col['name']] = str
        python_schema[table_name] = columns
    return python_schema

def extract(data_value: dict, args_str: str, schema: dict):
    """
    Extract information from the data_value using the args_str.
    data_value is a dictionary.
    args_str is a string. Example: "extract person: id, first_name, last_name, age"
    Update the data_value with the extracted information.
    """
    print(f"args_str: {args_str}")
    table_name = args_str.split(":")[0].split(" ")[1]
    
    try: 
        table_extract_schema = schema[table_name]
    except:
        # table not found in the extract_schema
        print(f"Table '{table_name}' not found in the extract_schema")
        return data_value
    
    # parse the args_str
    try:
        columns = args_str.split(":",1)[1].split(";")
        columns_dict = {}
        for col in columns:
            col = col.strip()
            if col == "":
                continue
            if ":" in col:
                col_name, col_value = col.split(":")
                columns_dict[col_name.replace("\"","").strip()] = col_value.strip()
            else:
                print(f"Error parsing the column '{col}'")
                return data_value
        
        # if all items in columns are empty except for the first one, return the data_value
    except:
        print(f"Error parsing the args_str: '{args_str}'")
        return data_value
    
    # if len(columns) != len(table_extract_schema):
    #     if len(columns) > len(table_extract_schema):
    #         columns = columns[:len(table_extract_schema)]
    #     else:
    #         columns = columns + ["?"] * (len(table_extract_schema) - len(columns))

    # check if the columns are in the table_extract_schema
    instance_dict = {}

    for i, col in enumerate(table_extract_schema):
        if col not in columns_dict:
            columns_dict[col] = "#"
        columns_dict[col] = columns_dict[col].replace("\"", "").strip()
        columns_dict[col] = columns_dict[col].replace("\'", "").strip()
        columns_dict[col] = columns_dict[col].replace("?", "#").strip()

        try:
            # if table_extract_schema[col] in [int, float]:
            #     columns[i] = columns[i].replace("\"", "").strip()
            #     columns[i] = columns[i].replace("\'", "").strip()
            #     instance_dict[col] = table_extract_schema[col](columns[i])
            # else:
            instance_dict[col] = table_extract_schema[col](columns_dict[col])
        except:
            # will keep as string if the conversion fails
            print(f"Error converting '{columns_dict[col]}' to '{table_extract_schema[col]}', keeping as string")
            instance_dict[col] = columns_dict[col]
    
    if table_name in data_value:
        data_value[table_name].append(instance_dict)
    else:
        data_value[table_name] = [instance_dict]

    print(f"data_value: {data_value}")
    return data_value


# def extract(data_value: dict, args_str: str, schema: dict):
#     """
#     Extract information from the data_value using the args_str.
#     data_value is a dictionary.
#     args_str is a string. Example: "extract person: id, first_name, last_name, age"
#     Update the data_value with the extracted information.
#     """
#     print(f"args_str: {args_str}")
#     table_name = args_str.split(":")[0].split(" ")[1]
    
#     try: 
#         table_extract_schema = schema[table_name]
#     except:
#         # table not found in the extract_schema
#         print(f"Table '{table_name}' not found in the extract_schema")
#         return data_value
    
#     # parse the args_str
#     try:
#         columns = args_str.split(":")[1].split(";")
#         columns = [col.strip() for col in columns]
#         # if all items in columns are empty except for the first one, return the data_value
#         if all(col == "?" for col in columns[1:]):
#             return data_value

#     except:
#         print(f"Error parsing the args_str: '{args_str}'")
#         return data_value
    
#     if len(columns) != len(table_extract_schema):
#         if len(columns) > len(table_extract_schema):
#             columns = columns[:len(table_extract_schema)]
#         else:
#             columns = columns + ["?"] * (len(table_extract_schema) - len(columns))

#     # check if the columns are in the table_extract_schema
#     instance_dict = {}
#     for i, col in enumerate(table_extract_schema):
#         # instance_dict[col] = table_extract_schema[col](columns[i])
#         columns[i] = columns[i].replace("\"", "").strip()
#         columns[i] = columns[i].replace("\'", "").strip()
#         columns[i] = columns[i].replace("?", "#").strip()
#         try:
#             # if table_extract_schema[col] in [int, float]:
#             #     columns[i] = columns[i].replace("\"", "").strip()
#             #     columns[i] = columns[i].replace("\'", "").strip()
#             #     instance_dict[col] = table_extract_schema[col](columns[i])
#             # else:
#             instance_dict[col] = table_extract_schema[col](columns[i])
#         except:
#             # will keep as string if the conversion fails
#             print(f"Error converting '{columns[i]}' to '{table_extract_schema[col]}', keeping as string")
#             instance_dict[col] = columns[i]
    
#     if table_name in data_value:
#         data_value[table_name].append(instance_dict)
#     else:
#         data_value[table_name] = [instance_dict]

#     print(f"data_value: {data_value}")
#     return data_value



def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(datapath,num_of_entries=None):
    decoder = json.JSONDecoder()
    texts = []
    entities = []
    ids = []
    key_value = []
    schema = []
    identified_values = []
    output = []
    domain = []
    difficulty = []
    
    with open(datapath) as f:
            json_data = json.load(f)
            for idx, line in enumerate(json_data):
                # print(line)
                id = idx
                texts.append(line['text'])
                entities.append(line['ground_truth_entities'])
                key_value.append(line['ground_truth_key_value'])
                domain.append(line['domain'])
                difficulty.append(line['difficulty'])

                if 'predicted_schema' in line:
                    schema.append(line['predicted_schema'])
                if 'schema' in line:
                    schema.append(line['schema'])
                if 'identified_values' in line:
                    identified_values.append(line['identified_values'])
                if 'output' in line:
                    output.append(line['output'])

                ids.append(id)
                if num_of_entries is not None and idx >= num_of_entries:
                    break
    data = {}
    data["texts"] = texts
    data["ground_truth_entities"] = entities
    data["ground_truth_key_value"] = key_value
    data["domain"] = domain
    data["difficulty"] = difficulty
    data["ids"] = ids
    if len(schema) >= 1:
        data["schema"] = schema
    if len(identified_values) >= 1:
        data["identified_values"] = identified_values
    if len(output) >= 1:
        data["output"] = output
    return data

def load(config):
    datapath = config['datapath']
    base_data_path = config['base_data_path']
    sro_path = None
    if "sro_path" in config:
        sro_path = config['sro_path']
    triplets = config['method'] == "triplets"
    
    
    data = load_data(f"{base_data_path}/{datapath}.json")
    texts = data["texts"]
    gt_entities = data["ground_truth_entities"]
    gt_key_value = data["ground_truth_key_value"]
    ids = data["ids"]
    domain = data["domain"]
    difficulty = data["difficulty"]
    # num_of_entries = math.ceil(config['num_of_entries']*len(texts))
    num_of_entries = config['num_of_entries']
    if triplets:
        with open(sro_path, 'rb') as file:
                # Load the data from the pickle file
                sro = pickle.load(file)
        return {"texts":texts[:num_of_entries],"gt_entities":gt_entities[:num_of_entries],"gt_key_value":gt_key_value[:num_of_entries],"sro":sro[:num_of_entries],"domain":domain[:num_of_entries],"difficulty":difficulty[:num_of_entries]}
    else:
        return {"texts":texts[:num_of_entries],"gt_entities":gt_entities[:num_of_entries],"gt_key_value":gt_key_value[:num_of_entries],"sro":None,"domain":domain[:num_of_entries],"difficulty":difficulty[:num_of_entries]}

def extract_schema(text):
    """
    Extracts the JSON/Python schema definition (as a list of tables) from the input text.
    Uses only regex (no bracket balancing).
    return string
    """
    patterns = [
        r"```(?:json|python)?\s*schema\s*=\s*(\[[\s\S]*\])\s*```",
        r"```(?:json|python)?\s*(\[[\s\S]*\])\s*```",
        r"schema\s*=\s*(\[[\s\S]*\])",
        r"[Ss]chema\s*[:=]\s*(\[[\s\S]*?\])",
        r"[Ss]chema\s*:\s*{\s*\"schema\"\s*:\s*(\[[\s\S]*?\])\s*}"

    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            match.group(1).replace("Schema: ", "")
            return match.group(1)
    return None

def extract_schema_with_repair(text):
    """NOT USED!!!"""
    good_json_string = json_repair.repair_json(text)
    if good_json_string=="":
        return None
    return good_json_string

def load_checkpoint(checkpoint_file):
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        else:
            return []

    # Function to save checkpoint
def save_checkpoint(checkpoint_file,data):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)


# Convert to JSON-serializable format
# (pickle can store Python objects that JSON cannot, like sets or custom objects)
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(elem) for elem in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # fallback to string conversion for unsupported types

def convert_pkl_to_json(json_file, pkl_file):
    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Convert to JSON-serializable format
    json_data = make_json_serializable(data)

    # Save to JSON file
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
