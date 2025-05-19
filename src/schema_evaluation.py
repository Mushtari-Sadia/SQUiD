
import sys
import os
import re

import warnings
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from utils import load_config, extract_schema
import argparse
import numpy as np
# from evaluation import compute_metrics

warnings.filterwarnings("ignore")


def compute_similarity_transformer(model, word1, word2):
    """Compute similarity using sentence-transformers model."""
    emb1 = model.encode(word1, convert_to_tensor=True)
    emb2 = model.encode(word2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def get_column_names(schema):
    """Extracts column names from the schema with table name prefixes."""
    column_names = []
    for table in schema:
        table_name = table["table_name"]
        for column in table["columns"]:
            prefixed_name = f"{table_name}_{column['name']}"
            column_names.append(prefixed_name)
        for column in table["columns"]:
            column_names.append(column['name'])
    return column_names



def compare_schema(schema, ground_truth_list):
    """
    Compares the predicted schema with the ground truth list of entities
    using semantic similarity to match column names.
    
    Returns a similarity score (percentage of well-matched entities).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Load a lightweight model
    column_names = get_column_names(schema)

    total_score = 0
    matched_count = 0

    predictions = []
    references = []

    for entity in ground_truth_list:
        if "id" in entity:
            continue  # Skip entities with 'id'

        similarities = [compute_similarity_transformer(model, entity, col) for col in column_names]
        best_match_score = max(similarities) if similarities else 0

        matched_count += best_match_score

    overall_score = (matched_count / len(ground_truth_list)) * 100

    # print(f"Predictions: {predictions}")
    # print(f"References: {references}")
    # # Now compute semantic metrics
    metrics = None
    # metrics = compute_metrics(predictions, references)
    return overall_score, metrics


def check_primary_keys(schema):
    """Checks if every table in the schema has a primary key."""
    tables_with_pk = 0

    for table in schema:
        has_primary_key = any(column.get("primary_key", False) for column in table["columns"])
        if has_primary_key:
            tables_with_pk += 1

    total_tables = len(schema)
    pk_coverage = (tables_with_pk / total_tables) * 100 if total_tables else 0

    return pk_coverage

# redefine
def check_foreign_keys(schema):
    """
    Checks if every declared foreign key in the schema points to a valid
    primary key in the referenced table.

    Returns:
    - Percentage of valid foreign key definitions.
    - Prints any invalid foreign key references.
    """
    fk_count = 0
    valid_fk_count = 0
    invalid = ""

    # Build a lookup for primary keys: {table_name: set of primary key column names}
    pk_lookup = {
        table["table_name"]: {
            col["name"] for col in table["columns"] if col.get("primary_key", False)
        }
        for table in schema
    }

    for table in schema:
        for column in table["columns"]:
            if column.get("foreign_key"):
                fk_count += 1
                fk_table = column.get("foreign_key_table")
                fk_column = column.get("foreign_key_column")

                if fk_table in pk_lookup and fk_column in pk_lookup[fk_table]:
                    valid_fk_count += 1
                else:
                    invalid += f"[Invalid FK] In table '{table['table_name']}', column '{column['name']}' references '{fk_table}.{fk_column}', which is not a valid primary key."


    fk_coverage = (valid_fk_count / fk_count) * 100 if fk_count else 0
    return fk_coverage, invalid



def compute_score(schema, entities_list):
    score = {
        'entity_coverage': 0,
        'primary_key_coverage': 0,
        'foreign_key_coverage': 0
    }
    # Convert schema string to Python object
    try:
        print(f"Schema: {schema}")
        schema = extract_schema(schema)
        print(f"Extracted Schema: {schema}")
        schema = eval(schema)
    except Exception as e:
        print(f"Error: {e}")
        print("Error: A valid schema was not generated.")
        return score, ""

    print(f"Extracted Schema: {schema}")
    print(f"Ground Truth Entities: {entities_list}")
    # Calculate scores
    score['entity_coverage'], semantic_metrics = compare_schema(schema, entities_list)
    # score.update({
    #     'entity_bleurt': semantic_metrics['bleurt'][-1],
    #     'entity_bertscore': semantic_metrics['bertscore'][-1],
    #     'entity_sacrebleu': semantic_metrics['sacrebleu'][-1],
    #     'entity_rouge': semantic_metrics['rouge'][-1],
    #     # 'entity_bartscore': semantic_metrics['bartscore'][-1],
    #     'entity_accuracy': semantic_metrics['accuracy'][-1],
    # })
    score['primary_key_coverage'] = check_primary_keys(schema)
    score['foreign_key_coverage'], invalid_foreign_keys = check_foreign_keys(schema)

    # Print scores .2f
    print("*"*40)
    print(f"Entity Coverage: {score['entity_coverage']:.2f}%")
    print(f"Primary Key Coverage: {score['primary_key_coverage']:.2f}%")
    print(f"Foreign Key Coverage: {score['foreign_key_coverage']:.2f}%")
    print("*"*40)
    return score, invalid_foreign_keys

    

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

    results_root = config['results_dir']
    evaluation_path = os.path.join(results_root, f"{config['datapath']}/{config['method']}_{config['prompt_type']}_{config['model_name']}.json")
    os.makedirs(os.path.join(results_root, f"{config['datapath']}"), exist_ok=True)

    with open(evaluation_path, "r") as f:
        data = json.load(f)
        f2 = open(f"logs/schema_results/{config['method']}_{config['prompt_type']}_{config['model_name']}_score.txt", 'w')
        sys.stdout = f2

        aggr_score = {
            'entity_coverage': [],
            'entity_bleurt': [],
            'entity_bertscore': [],
            'entity_sacrebleu': [],
            'entity_rouge': [],
            # 'entity_bartscore': [],
            'entity_accuracy': [],
            'primary_key_coverage': [],
            'foreign_key_coverage': []
        }
        domain = {}
        for item in data:
            
            schema = item['predicted_schema']
            
            if item['domain'] not in domain:
                domain[item['domain']] = schema
            else:
                continue

            entities_list = item['ground_truth_key_value'].keys()
            score, invalid_foreign_keys = compute_score(schema, entities_list)
            if invalid_foreign_keys!="":
                print(f"Invalid Foreign Keys: {invalid_foreign_keys} for schema: {schema}")
            if not score:
                continue
            aggr_score['entity_coverage'].append(score['entity_coverage'])
            aggr_score['primary_key_coverage'].append(score['primary_key_coverage'])
            aggr_score['foreign_key_coverage'].append(score['foreign_key_coverage'])

        avg_score = {
            'entity_coverage': np.mean(aggr_score['entity_coverage']),
            'primary_key_coverage': np.mean(aggr_score['primary_key_coverage']),
            'foreign_key_coverage': np.mean(aggr_score['foreign_key_coverage'])
        }
        print("\n\n\n")
        print(f"Average Entity Coverage: {avg_score['entity_coverage']:.2f}%")
        print(f"Average Primary Key Coverage: {avg_score['primary_key_coverage']:.2f}%")
        print(f"Average Foreign Key Coverage: {avg_score['foreign_key_coverage']:.2f}%")
        sys.stdout = sys.__stdout__
        f2.close()
        