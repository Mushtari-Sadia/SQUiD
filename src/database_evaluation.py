import sqlite3
import json
from sentence_transformers import SentenceTransformer, util
import json
import sys
import os
from utils import load_config
import re
# import evaluate
import argparse

# bleurt = evaluate.load("bleurt", checkpoint="bleurt-base-128")
# bertscore = evaluate.load("bertscore")
# sacrebleu = evaluate.load("sacrebleu")
# rouge = evaluate.load("rouge")

def is_number(s):
    """
    Detect if s is a number (optionally with $ and commas) and return the cleaned float value or None.
    
    Acceptable formats:
    - "$1,234.56"
    - "-$1,000"
    - "1234.56"
    - "1,000"
    - "$1000"
    - "   $12,345.67   "
    
    Returns:
        float or None
    """
    if not isinstance(s, str):
        return None
    
    s = s.strip()

    # Regex to match valid number formats (with optional $ and commas)
    pattern = re.compile(
        r"""^            # start of string
        -?\$?            # optional negative sign, then optional $
        (?:
            \d{1,3}(?:,\d{3})+  # 1-3 digits, then comma-separated groups
            |                  # or
            \d+                # plain digits
        )
        (?:\.\d+)?        # optional decimal part
        $                 # end of string
        """, re.VERBOSE
    )
    
    if pattern.match(s):
        # Remove $ and commas
        cleaned = s.replace('$', '').replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None

def print_joined_rows(joined_rows):
    print("\nJoined Rows:")
    for idx, row in enumerate(joined_rows, start=1):
        print(f"\nRow {idx}:")
        for key, value in row.items():
            print(f"  {key}: {value}")
    print("\nTotal rows:", len(joined_rows))

def get_id_key(joined_rows):
    for idx, row in enumerate(joined_rows, start=1):
        for key, value in row.items():
            return key
    return None


def get_join_rows(json_path,datapath,method):
    # Load the JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    cursor = None
    conn = None
    # Process each database entry
    for i in data:
        if "joined_rows" not in i:
            i["joined_rows"] = []
        else:
            if i["joined_rows"] != []:
                continue
        db_path = f"databases/{datapath}/{method}/{i['db_name']}.db"
        print(f"Processing {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            all_rows = []

            # Run each subquery and collect results
            if isinstance(i["join_query"], list):
                for j in i["join_query"]:
                    try:
                        # print(f"Executing query: {j}")
                        j = j.replace("```sql", "")
                        j = j.replace("```", "")
                        cursor.execute(j)
                        columns = [desc[0] for desc in cursor.description]
                        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                        all_rows.extend(rows)
                    except sqlite3.Error as e:
                        print(f"Query error in {db_path}: {e}")
                        continue

                # Save merged results
                i["joined_rows"] = all_rows
            else:
                j = i["join_query"]
                try:
                    # print(f"Executing query: {j}") 
                    j = j.replace("```sql", "")
                    j = j.replace("```", "")
                    cursor.execute(j)
                    columns = [desc[0] for desc in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    all_rows.extend(rows)
                except:
                    # print(f"Query error in {db_path}: {e}")
                    continue
                # Save merged results
                i["joined_rows"] = all_rows

        except sqlite3.Error as e:
            print(f"Error processing {db_path}: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        
    # Save updated JSON after processing all
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_similarity(model, word1, word2, fast=True):
    """Compute similarity, treating numbers differently. Returns (best_score, best_metric)."""
    n1 = is_number(word1)
    n2 = is_number(word2)
    
    if n1 is not None:
        if n2 is not None:
          return (1.0, "number_match") if abs(n1 - n2) < 1e-2 else (0.0, "number_mismatch")
        else:
          return (0.0, "number_mismatch")

    # Fast path: cosine similarity only
    if fast:
        emb1 = model.encode(word1, convert_to_tensor=True)
        emb2 = model.encode(word2, convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
        cosine_sim_norm = (cosine_sim + 1) / 2  # Normalize to [0, 1]
        return cosine_sim_norm, "cosine_similarity"

    # # Full evaluation: slower but thorough
    # emb1 = model.encode(word1, convert_to_tensor=True)
    # emb2 = model.encode(word2, convert_to_tensor=True)
    # cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()

    # bleurt_score = bleurt.compute(predictions=[word1], references=[word2])['scores'][0]
    # sacrebleu_score = sacrebleu.compute(predictions=[word1], references=[[word2]])['score'] / 100.0
    # rougeL_f1 = rouge.compute(predictions=[word1], references=[word2])['rougeL']

    # cosine_sim_norm = (cosine_sim + 1) / 2  # Normalize cosine to [0, 1]

    # scores = {
    #     'cosine_similarity': cosine_sim_norm,
    #     'bleurt': bleurt_score,
    #     'sacrebleu': sacrebleu_score,
    #     'rougeL': rougeL_f1
    # }
    # best_metric = max(scores, key=scores.get)
    # best_score = scores[best_metric]
    
    # return best_score, best_metric




def extract_first_table_id(schema):
    """Extract first table and its first column to create ID key name."""
    first_table = schema[0]['table_name']
    first_column = schema[0]['columns'][0]['name']
    return f"{first_table}_{first_column}"

def extract_id_key_from_schema(schema):
    """Find the first column ending with 'id' across all tables and return its fully qualified alias name."""
    for table in schema:
        table_name = table["table_name"]
        for column in table["columns"]:
            if column["name"].lower().endswith("id"):
                return f"{table_name}_{column['name']}"
    raise ValueError("No column ending with 'id' found in schema.")


def match_ground_truth_with_joined_rows(data_entry, model, similarity_threshold=0.8, row_similarity_threshold=0.5):
    """
    Compare ground_truth_key_value with joined_rows using semantic similarity.
    Returns overall match accuracy score.
    """
    gt = data_entry.get("ground_truth_key_value", {})
    joined = data_entry.get("joined_rows", [])
    if not joined:
        return -1,-1,-1

    id_key = get_id_key(joined)

    print("~" * 40)
    print_joined_rows(joined)
    print("~" * 40)
    print("ID key:", id_key)
    print("~" * 40)
    # Index rows by ID value for faster lookup
    # Index rows by ID value for faster lookup, with lists of rows for duplicate IDs
    id_to_row = {}

    for row in joined:
        row_id = str(row.get(id_key))  # Extract the ID value as a string
        if row_id in id_to_row:
            id_to_row[row_id].append(row)  # Append the row to the list for this ID
        else:
            id_to_row[row_id] = [row]  # Create a new list for this ID

    # Now id_to_row contains lists of rows for each unique ID


    total_matches = 0
    total_match_in_right_column = 0
    total_items = 0
    total_row_accuracy_sum = 0
    total_row_accuracy_count = 0
    
    for attribute, kv_list in gt.items():
        row_ids = [str(id_val) for id_val, _ in kv_list]
        unique_row_ids = set(row_ids)
        rows_found = 0
        best_sim = None
        best_row_val = None
        found_row_key = None

        for id_val, gt_val in kv_list:
            id_val_str = str(id_val)
            total_items += 1
            match_found = False
            match_found_right_column = False
            
            rows = id_to_row.get(id_val_str) # get the list of rows for the same ID in ground truth
            if rows is None:
                rows = []
            if len(rows) == 0:
                idx = int(id_val)-1 # if no matching id was found, get the row with the same index as the id
                while idx < len(joined):
                    rows.append(joined[idx])
                    idx += 5  #only for ensemble methods (since there will be more than 5 rows), but will not cause any issues for other methods - our dataset contains 5 rows per entry, if more, change this
            if len(rows) == 0:
                print(f"No match found for GT: {gt_val} with ID: {id_val}")
                continue

            rows_found += 1
            for row in rows:
                for row_key, row_val in row.items():
                    if not isinstance(row_val, (str, int, float)):
                        continue
                    sim, metric = compute_similarity(model, str(gt_val), str(row_val))
                    if sim < similarity_threshold:
                        continue
                    best_sim = sim
                    best_row_val = row_val
                    found_row_key = row_key
                    match_found = True
                    sim2, metric2 = compute_similarity(model, attribute, str(row_key))
                    if sim2 >= row_similarity_threshold:
                        match_found_right_column = True
                        print(f"ID key: {id_val} Match found: {gt_val} (GT) vs {row_val} (Row) with {metric} {best_sim:.4f}")
                        print(f"ID key: {id_val} Match found in the same column: {attribute} (GT) vs {row_key} (Row) with {metric2} {sim2:.4f}")
                        break  # Found in correct column

            if match_found:
                total_matches += 1
                if match_found_right_column:
                    total_match_in_right_column += 1
                else:
                    print(f"ID key: {id_val} Match found: {gt_val} (GT) vs {best_row_val} (Row) with {metric} {best_sim:.4f}")
                    print(f"ID key: {id_val} Match found in different column: {attribute} (GT) vs {found_row_key} (Row) with {metric2} {sim2:.4f}")
            else:
                print(f"No match found for GT: {gt_val} with ID: {id_val}")
            

        rows_in_gt = len(unique_row_ids)
        row_accuracy = rows_found / rows_in_gt if rows_in_gt else 1
        total_row_accuracy_sum += row_accuracy
        total_row_accuracy_count += 1
        

    match_accuracy = total_matches / total_items if total_items else 1
    match_in_right_column_accuracy = total_match_in_right_column / total_items if total_items else 0
    avg_row_accuracy = total_row_accuracy_sum / total_row_accuracy_count if total_row_accuracy_count else 1


    return avg_row_accuracy, match_accuracy, match_in_right_column_accuracy




# def match_ground_truth_with_joined_rows(data_entry, model, similarity_threshold=0.6):
#     """
#     Compare ground_truth_key_value values with all values in joined_rows using semantic similarity.
#     Ignores ID-based matching.
#     """
#     gt = data_entry.get("ground_truth_key_value", {})
#     joined = data_entry.get("joined_rows", [])

#     total_matches = 0
#     total_items = 0

#     for attribute, kv_list in gt.items():
#         for _, gt_val in kv_list:
#             total_items += 1
#             match_found = False

#             for row in joined:
#                 for row_val in row.values():
#                     if isinstance(row_val, (str, int, float)):
#                         sim = compute_similarity(model, str(gt_val), str(row_val))
#                         if sim >= similarity_threshold:
#                             match_found = True
#                             print(f"Match found: {gt_val} (GT) vs {row_val} (Row) with similarity {sim:.4f}")
#                             break
#                 if match_found:
#                     break

#             if match_found:
#                 total_matches += 1
#             else:
#                 print(f"No match found for GT: {gt_val}")

#     return total_matches / total_items if total_items > 0 else 0



if __name__ == "__main__":
    experiments = load_config("configs/config.yaml")
    config = experiments["database_generation"]
    datapath = config['datapath']
    method = config['method']
    model_name = config['model_name']
    base_data_path = config['base_data_path']
    results_dir = config['results_dir']
    logs_dir = config['logs_dir']

    parser = argparse.ArgumentParser(description="Evaluate database generation.")
    parser.add_argument("--method", type=str, required=False, help="Method used for database generation.")
    parser.add_argument("--json_path", type=str, required=False, help="Path to the JSON file.")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the model to use.")
    parser.add_argument("--dataset", type=str, required=False, help="bird")
    args = parser.parse_args()
    if args.method:
        method = args.method
    if args.model_name:
        model_name = args.model_name
    if args.dataset == "bird":
        datapath = config['datapath'].replace("SyntheticText/merged_dataset2","BIRD/bird_dataset")
    datapath = datapath.replace("<<model_name>>", model_name)


    json_path = f"{results_dir}{method}/{datapath}.json"
    score_path = f"{logs_dir}{method}/{datapath}/"
    if args.json_path:
        json_path = args.json_path
        score_path = f"{logs_dir}unspecified/{json_path}/"
    
    if "ensemble" not in method:
        get_join_rows(json_path,datapath,method)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    os.makedirs(score_path, exist_ok=True)
    f = open(f"{score_path}/score.txt", "w")
    sys.stdout = f

    with open(json_path) as f:
        data = json.load(f)

    total_avg_row_accuracy = 0
    total_match_accuracy = 0
    total_match_in_right_column_accuracy = 0
    total_items = 0
    

    s = f"{results_dir}{method}/{datapath}"
    results_dict_path = f"{s}/evaluation/results.json"
    os.makedirs(s+"/evaluation/", exist_ok=True)
    
    # load the existing results if available
    # if os.path.exists(results_dict_path):
    #     # if "ensemble" not in method:
    #     with open(results_dict_path) as f:
    #         results_dict = json.load(f)
    #     # else:
    #     #     results_dict = {}
    # else:
    #     results_dict = {}
    results_dict = {}
    

    for item in data:
        if item['db_name'] in results_dict:
            continue
        
        avg_row_accuracy,match_accuracy,match_in_right_column_accuracy = match_ground_truth_with_joined_rows(item, model)
        if match_accuracy == -1:
            results_dict[item['db_name']] = {
            "avg_row_accuracy": 0,
            "match_accuracy": 0,
            "match_in_right_column_accuracy": 0
        }
            continue
        total_avg_row_accuracy += avg_row_accuracy
        total_match_accuracy += match_accuracy
        total_match_in_right_column_accuracy += match_in_right_column_accuracy
        total_items += 1

        print(f"*"*40)
        print(f"DB Name: {item['db_name']}")
        print(f"Avg Row Accuracy: {avg_row_accuracy:.4f}")
        print(f"Match Accuracy: {match_accuracy:.4f}")
        print(f"Match in Right Column Accuracy: {match_in_right_column_accuracy:.4f}")
        
        print(f"*"*40)
        
        results_dict[item['db_name']] = {
            "avg_row_accuracy": avg_row_accuracy,
            "match_accuracy": match_accuracy,
            "match_in_right_column_accuracy": match_in_right_column_accuracy
        }

        
    # Calculate overall accuracy
    if total_items > 0:
        overall_accuracy = total_match_accuracy / total_items
        avg_row_accuracy = total_avg_row_accuracy / total_items
        match_in_right_column_accuracy = total_match_in_right_column_accuracy / total_items
    else:
        overall_accuracy = 0
        avg_row_accuracy = 0
        match_in_right_column_accuracy = 0

    # Save the results to a JSON file
    with open(results_dict_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Overall Avg Row Accuracy: {avg_row_accuracy:.4f}")
    print(f"Overall Match Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Match in Right Column Accuracy: {match_in_right_column_accuracy:.4f}")
    print(f"Total Items Processed: {total_items}")
    print(f"Total Items: {len(data)}")

        
    
    f.close()
    sys.stdout = sys.__stdout__
    print("Evaluation complete. Results saved to:", results_dict_path)