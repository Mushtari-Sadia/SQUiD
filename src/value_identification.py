import sys
import re
import stanza
from stanza.server import CoreNLPClient
from sentence_transformers import SentenceTransformer, util
import numpy as np
import math
import traceback
import ast
try:
    stanza.install_corenlp()
except:
    pass

from utils import load_data, extract_schema, load_config, load_checkpoint, save_checkpoint
from model import *
from prompt import *
import argparse
import pickle
import logging


def save_to_json(original_json,triplet_file,output_file):

    original_data = []
    triplet_data = []
    combined_data = []

    with open(original_json, 'r') as f:
        original_data = json.load(f)

    with open(triplet_file, 'rb') as file:
        # Load the data from the pickle file
        triplet_data = pickle.load(file)

    for i in range(len(original_data)):
        original_entry = original_data[i]
        try:
            triplet_entry = triplet_data[i]
        except:
            break
        
        # Combine the entries
        combined_entry = {
            "text": original_entry["text"],
            "ground_truth_entities": original_entry["ground_truth_entities"],
            "ground_truth_key_value": original_entry["ground_truth_key_value"],
            "domain": original_entry["domain"],
            "difficulty": original_entry["difficulty"],
            "predicted_schema": original_entry["predicted_schema"],
            "identified_values": triplet_entry
        }
        
        combined_data.append(combined_entry)

    # Save the combined data to a new JSON file
    # output_dir = "/home/mushtari/nl2db/nl2db-main/results/value_identification/symbolic/SyntheticText/merged_dataset/"
    # output_file = f"{output_dir}/text_cot_llama.json"
    # check if path exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

# Extract braces function
def extract_braces(text):
    # Extract content from each brace set
    matches = re.findall(r"\{([^{}]+)\}", text)  # Non-greedy match to extract everything between braces
    return matches

# Convert braces to dictionary (subject, relation, object)
def braces_to_dict(braces):
    triplets = []
    
    for brace in braces:
        # Split the brace content by commas
        components = [item.strip() for item in brace.split(',')]
        
        # Create a dictionary for each triplet
        if len(components) >= 3:
            triplet = {
                "subject": components[0],
                "relation": components[1],
                "object": " ".join(components[2:])
            }
            triplets.append(triplet)
    # print(triplets)
    return triplets

def get_superkey(schema):
        first_table = schema[0]
        for column in first_table["columns"]:
            if column.get("primary_key"):
                table_name, column_name = first_table["table_name"], column["name"]
                return f"{table_name}.{column_name}"
        return None

class ValueIdentificationPipeline:
    def __init__(self, config):
        self.model = Model(config['model_name'])
        self.model_name = config['model_name']
        self.datapath = config['datapath']
        self.base_data_path = config['base_data_path']
        self.method = config['method']
        self.load_data_path = config['schema_path']
        self.results_path = config['results_dir'] + self.method + "/"
        # Create the directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        self.logs_path = config['logs_dir']
        os.makedirs(self.logs_path, exist_ok=True)
        self.data = load_data(self.load_data_path)
        # self.num_of_entries = math.ceil(config['num_of_entries']*len(self.data["texts"]))
        self.num_of_entries = config['num_of_entries'] #remove later
        self.nlp = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'openie'], 
                                 timeout=30000, memory='4G')
        logging.basicConfig(
            filename=f'{self.logs_path}triplet_generation.log',
            filemode='w',  # or 'a' to append
            level=logging.INFO,
            format='%(message)s'  # Only show the actual log message
        )
        self.logger = logging.getLogger(__name__)


    def get_each_users_data_from_text(self,text):
        each_users_data = text.split('\n')
        return each_users_data

    
    def get_each_users_data_and_superkeys(self,text):
        """
        Extract paragraphs and their associated superkeys from the text.
        Args:
            text (str): The input text containing paragraphs and superkeys.
        Returns:
            list: A list of dictionaries, each containing a "paragraph" and its associated "superkey".
        """
        # Updated regex: capture any characters (non-greedy) as the superkey
        entries = re.findall(r"paragraph \d+: (.*?)\nassociated superkey: ([^\n]+)", text, re.DOTALL)
        
        # Build list of dictionaries with superkey as a string
        result = [{"paragraph": paragraph.strip(), "superkey": superkey.strip()} for paragraph, superkey in entries]
        
        if len(result) == 0:
            # If no matches found, try the alternative regex
            pattern = r"(?:\*\*)?[Pp]aragraph\s+\d+:(?:\*\*)?\s*\n(.*?)\nassociated superkey:\s*([^\n]+)"
            entries = re.findall(pattern, text, re.DOTALL)
            result = [
                {
                    "paragraph": paragraph.strip(),
                    "superkey": superkey.strip()
                }
                for paragraph, superkey in entries
            ]
        # If still no matches, return an empty list
        return result

    
    def associate_superkeys_with_each_users_data(self, superkey, each_users_data, idx):
        text = self.data["texts"][idx]
        schema = self.data["schema"][idx]
        system_prompt, user_prompt = get_superkey_association_prompt(text, schema, superkey, each_users_data)
        print("System Prompt:",system_prompt)
        print("User Prompt:",user_prompt)
        associations = self.model.predict(system_prompt,user_prompt)
        print("Output:",associations)
        # Extract sentences and their associated superkeys
        associations_list = self.get_each_users_data_and_superkeys(associations)
        if len(associations_list) == 0:
            for j,item in enumerate(each_users_data):
                associations_list.append({"paragraph": item, "superkey": str(j+1)})
        # Print the results
        print("Associations:", associations_list)
        return associations_list
    
    def get_missing_nouns_in_triplets(self,sentence):
        openie_triples = sentence.openieTriple
        noun_pronoun_list = [
                token.word for token in sentence.token 
                if token.pos in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'CD', 'PRP$']
            ]
        self.logger.debug(f"Nouns/pronouns in sentence: {noun_pronoun_list}")
        # Flatten the list of all triplet parts (subject, relation, object)
        openie_triplets_str = str(openie_triples)

        # Find which nouns/pronouns/numbers are missing from the triplet parts
        missing_items = [noun for noun in noun_pronoun_list if noun not in openie_triplets_str]
        
        return missing_items
    
    def extract_openie_triplets(self, sentence):
        triplets = []
        for triple in sentence.openieTriple:
            triplets.append({
                "subject": triple.subject,
                "relation": triple.relation,
                "object": triple.object
            })
        return triplets
    
    def get_additional_triplets(self, sentence_text, existing_triplets, missing_items):
        model = self.model
        validation_prompt = (
            f"sentence:\n{sentence_text}\n"
            f"These are the {{subject, relation, object}} triplets extracted from the sentence:\n"
        )
        for trip in existing_triplets:
            validation_prompt += f"{{{trip['subject']},{trip['relation']},{trip['object']}}}\n"

        validation_prompt += (
            f"These are the information that are missing in the triplets: {', '.join(missing_items)}\n"
            "Create additional triplets in the format {subject, relation, object} that incorporate the missing information. Additional triplets:"
        )

        self.logger.debug(f"Validation prompt for missing items:\n{validation_prompt}")

        try:
            output = model.predict(validation_prompt)
            self.logger.debug(f"Raw model output: {output}")
            extracted = extract_braces(output)
            structured = braces_to_dict(extracted)
            return structured
        except Exception as e:
            self.logger.error(f"Error extracting additional triplets: {e}")
            # traceback.print_exc()
            return []
        
    
    def generate_triplets(self, associations_list, text):
        nlp = self.nlp
        sentence_idx = 0
        for associations in associations_list:
            paragraph = associations["paragraph"]
            associations["triplets"] = []
            ann = nlp.annotate(paragraph)
            superkey = associations["superkey"]
            sentence_idx = 0
            for sentence in ann.sentence:
                sentence_text = " ".join([token.word for token in sentence.token])
                self.logger.info(f"\nProcessing sentence: {sentence_text}")
                self.logger.info(f"\nSuperkey of sentence: {superkey}\n")
                associations["triplets"].extend(self.extract_openie_triplets(sentence))
                # logging begin
                formatted_triplets = "\n".join(
                [f"{{{t['subject']}, {t['relation']}, {t['object']}}}" for t in associations["triplets"]]
                )
                self.logger.info(f"*******\nExtracted OpenIE triplets:\n{formatted_triplets}*******")
                # logging end

                missing_items = self.get_missing_nouns_in_triplets(sentence)
                self.logger.info(f"Missing items from triplets: {missing_items}")

                if missing_items:
                    additional_triplets = self.get_additional_triplets(sentence_text, associations["triplets"], missing_items)
                    # logging begin
                    formatted_additional = "\n".join(
                        [f"{{{t['subject']}, {t['relation']}, {t['object']}}}" for t in additional_triplets]
                    )
                    self.logger.info(f"*******\nLLM-generated additional triplets:\n{formatted_additional}\n*******")
                    # logging end
                    if additional_triplets:
                        associations["triplets"].extend(additional_triplets)
                
                sentence_idx += 1

        return associations_list
    
    

    def extract_triplets_as_dicts(self,llm_output):
        """
        Extracts triplets from LLM output and returns them as a list of dictionaries:
        [{"table_name": ..., "column_name": ..., "value": ...}, ...]
        """
        print(llm_output)
        try:
            # Extract the relevant part of the output
            output = llm_output.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
        except:
            output = llm_output
        matches = re.findall(r'\[\s*{.*?}\s*\]', output, flags=re.DOTALL)
        output = matches[-1] if matches else output

        try:
            if output[-1] == "]":
                output = ast.literal_eval(output)
            else:
                output = ast.literal_eval(output+ "]")
        except:
            print("Error in parsing the output")
            output = []
        print(output)
        return output

    def get_additional_triplets_llm(self, paragraph, existing_triplets, missing_items, schema):
        """
        Uses the LLM to generate additional triplets for missing items in a paragraph,
        guided by the database schema. The prompt is divided into system and user messages.
        """
        model = self.model

        system_prompt = (
            "You are an expert in Open Information Extraction and relational databases. "
            "You will be given a database schema, a paragraph, and a list of already extracted triplets. "
            "Your task is to extract additional triplets to cover missing information. "
            "Each triplet must be formatted as a Python dictionary with the keys: 'table_name', 'column_name', and 'value'. "
            "The final output must be a Python list of such dictionaries. "
            "Only extract values that explicitly appear in the input sentence. The table_name and column_name must match the schema. "
            "Do NOT repeat existing triplets or invent values. DO NOT generate code, do the task yourself. "
        )

        # Example section (flight domain)
        example_schema = '''
    Schema:
    [
        {
            "table_name": "flight",
            "columns": [
                {"name": "flight_id", "primary_key": true},
                {"name": "origin"},
                {"name": "destination"},
                {"name": "departure_time"},
                {"name": "arrival_time"}
            ]
        },
        {
            "table_name": "airline",
            "columns": [
                {"name": "airline_id", "primary_key": true},
                {"name": "name"},
                {"name": "country"}
            ]
        }
    ]
        '''

        example_paragraph = "The Delta flight from New York to Los Angeles departs at 9 AM and arrives at noon."

        example_existing = [
            {"table_name": "flight", "column_name": "origin", "value": "New York"},
            {"table_name": "flight", "column_name": "destination", "value": "Los Angeles"}
        ]

        example_missing = "departure_time, arrival_time, airline name"

        example_output = '''
    [
        {"table_name": "flight", "column_name": "departure_time", "value": "9 AM"},
        {"table_name": "flight", "column_name": "arrival_time", "value": "noon"},
        {"table_name": "airline", "column_name": "name", "value": "Delta"}
    ]
        '''

        user_prompt = f"""
    You will be given a schema, a paragraph, existing triplets, and a list of missing items. 
    Extract *only the new triplets* for the missing items. Only provide the missing triplets, do not output anything else. Do not provide any explanations.

    {example_schema}
    Paragraph: {example_paragraph}
    Existing Triplets: {example_existing}
    Missing Information: {example_missing}
    Output: {example_output}

    Now for the real input:

    Schema:
    {schema}

    Paragraph:
    {paragraph}

    Existing Triplets:
    {existing_triplets}

    Missing Information: {', '.join(missing_items)}

    Output:
    """

        self.logger.debug(f"System prompt:\n{system_prompt}")
        self.logger.debug(f"User prompt:\n{user_prompt}")

        try:
            output = model.predict(system_prompt, user_prompt)
            self.logger.debug(f"Raw model output: {output}")
            extracted = self.extract_triplets_as_dicts(output)
            return extracted
        except Exception as e:
            self.logger.error(f"Error extracting additional triplets: {e}")
            return []



    def get_missing_nouns_in_triplets_llm(self,paragraph,values):
        nlp = self.nlp
        ann = nlp.annotate(paragraph)
        missing_items = []
        for sentence in ann.sentence:
            noun_pronoun_list = [
                token.word for token in sentence.token 
                if token.pos in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'CD', 'PRP$']
            ]
            self.logger.debug(f"Nouns/pronouns in sentence: {noun_pronoun_list}")
            missing_items.extend([noun for noun in noun_pronoun_list if noun not in values])
        
        return missing_items
    
    def generate_triplets_llm(self, associations_list, text, schema):
        model = self.model
        
        for associations in associations_list:
            paragraph = associations["paragraph"]
            associations["triplets"] = []
            superkey = associations["superkey"]
            self.logger.info(f"\nProcessing paragraph: {paragraph}")
            self.logger.info(f"\nSuperkey of sentence: {superkey}\n")
            system_prompt, user_prompt = get_triplets_extraction_with_llm_prompt(paragraph,schema)
            triplets = model.predict(system_prompt, user_prompt)
            triplets = self.extract_triplets_as_dicts(triplets)
            associations["triplets"].extend(triplets)
            # logging begin
            self.logger.info(f"*******\nExtracted LLM triplets:\n{associations['triplets']}*******")
            # logging end
            values = [item['value'] for item in associations["triplets"]]
            missing_items = self.get_missing_nouns_in_triplets_llm(paragraph,values)
            self.logger.info(f"Missing items from triplets: {missing_items}")
            if missing_items:
                additional_triplets = self.get_additional_triplets_llm(paragraph, associations["triplets"], missing_items, schema)
                # logging begin
                self.logger.info(f"*******\nLLM-generated additional triplets:\n{additional_triplets}\n*******")
                # logging end
                if additional_triplets:
                    associations["triplets"].extend(additional_triplets)
            

        return associations_list
    
    def redundancy_remover_x(self, triplets_list, threshold=0.97):
        """
        Remove semantically redundant triplets from each sentence using SentenceTransformer similarity.
        Args:
            triplets_list: List of sentence-level triplet dictionaries (from generate_triplets).
            threshold: Cosine similarity threshold for considering two triplets redundant.
        Returns:
            Deduplicated triplets_list with proper {'subject', 'relation', 'object'} structure.
        """
        model = SentenceTransformer("sentence-transformers/sentence-t5-base")

        for paragraph_entry in triplets_list:
            self.logger.info(f"\nProcessing sentence: {paragraph_entry.get('paragraph', '')}")
            self.logger.info(f"\nSuperkey of sentence: {paragraph_entry.get('superkey', '')}\n")

            triplets = paragraph_entry.get("triplets", [])
            if not triplets:
                continue

            # Convert to string form for deduplication
            triplet_strings = [
                f"{t['subject']} | {t['relation']} | {t['object']}" for t in triplets
            ]

            embeddings = model.encode(triplet_strings, convert_to_tensor=True)
            similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

            to_remove = set()
            for i in range(len(triplet_strings)):
                for j in range(i + 1, len(triplet_strings)):
                    if similarity_matrix[i][j] > threshold:
                        to_remove.add(j)

            # Rebuild the deduplicated triplets as dicts
            unique_triplets = []
            for i in range(len(triplet_strings)):
                if i not in to_remove:
                    try:
                        parts = [p.strip() for p in triplet_strings[i].split('|')]
                        triplet_dict = {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2]
                        }
                        unique_triplets.append(triplet_dict)
                    except Exception as e:
                        self.logger.error(f"Failed to parse triplet: {triplet_strings[i]}")
                        self.logger.error(e)

            self.logger.info(
                f"Triplets reduced: {len(triplets)} → {len(unique_triplets)} (removed {len(to_remove)})"
            )
            formatted_reduced = "\n".join(
                [f"{{{t['subject']}, {t['relation']}, {t['object']}}}" for t in unique_triplets]
            )
            self.logger.info(f"*******\nReduced triplets:\n{formatted_reduced}\n*******")

            paragraph_entry["triplets"] = unique_triplets

        return triplets_list
    
    def redundancy_remover_y(self, triplets_list, threshold=0.99):
        """
        Remove semantically redundant triplets from each sentence using SentenceTransformer similarity.
        Args:
            triplets_list: List of sentence-level triplet dictionaries (from generate_triplets).
            threshold: Cosine similarity threshold for considering two triplets redundant.
        Returns:
            Deduplicated triplets_list with proper {'subject', 'relation', 'object'} structure.
        """
        model = SentenceTransformer("sentence-transformers/sentence-t5-base")

        for paragraph_entry in triplets_list:
            self.logger.info(f"\nProcessing sentence: {paragraph_entry.get('paragraph', '')}")
            self.logger.info(f"\nSuperkey of sentence: {paragraph_entry.get('superkey', '')}\n")

            triplets = paragraph_entry.get("triplets", [])
            if not triplets:
                continue

            # Convert to string form for deduplication
            triplet_strings = [
                f"{t['table_name']} | {t['column_name']} | {t['value']}" for t in triplets
            ]

            embeddings = model.encode(triplet_strings, convert_to_tensor=True)
            similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

            to_remove = set()
            for i in range(len(triplet_strings)):
                for j in range(i + 1, len(triplet_strings)):
                    if similarity_matrix[i][j] > threshold:
                        to_remove.add(j)

            # Rebuild the deduplicated triplets as dicts
            unique_triplets = []
            for i in range(len(triplet_strings)):
                if i not in to_remove:
                    try:
                        parts = [p.strip() for p in triplet_strings[i].split('|')]
                        triplet_dict = {
                            "table_name": parts[0],
                            "column_name": parts[1],
                            "value": parts[2]
                        }
                        unique_triplets.append(triplet_dict)
                    except Exception as e:
                        self.logger.error(f"Failed to parse triplet: {triplet_strings[i]}")
                        self.logger.error(e)

            self.logger.info(
                f"Triplets reduced: {len(triplets)} → {len(unique_triplets)} (removed {len(to_remove)})"
            )
            formatted_reduced = "\n".join(
                [f"{{{t['table_name']}, {t['column_name']}, {t['value']}}}" for t in unique_triplets]
            )
            self.logger.info(f"*******\nReduced triplets:\n{formatted_reduced}\n*******")

            paragraph_entry["triplets"] = unique_triplets

        return triplets_list

    def value_identification_symbolic(self):
        model = self.model
        data = self.data
        num_of_entries = self.num_of_entries
        data = {"texts":data["texts"][:num_of_entries],"predicted_schema":data["schema"][:num_of_entries]}

        os.makedirs("logs/value_identification_results/symbolic", exist_ok=True)
        f = open("logs/value_identification_results/symbolic/associations.txt","w")
        sys.stdout = f

        results_path = self.results_path
        datapath = self.datapath

        s = f"{results_path}{datapath}.json"
        head, tail = s.rsplit("/", 1)
        os.makedirs(head, exist_ok=True)

        SRO_list_for_all_texts = load_checkpoint(f"{head}/checkpoint.pkl")
        processed_idx = len(SRO_list_for_all_texts)

        for idx, text in enumerate(data["texts"]):
            # was_empty = False
            # if len(SRO_list_for_all_texts[idx]) == 0:
            #     was_empty = True
            # if idx < processed_idx and not was_empty:
            if idx < processed_idx:
                print(f"Skipping already processed index {idx}, while the total processed index is {processed_idx}")
                continue
            schema = data["predicted_schema"][idx]
            schema = extract_schema(schema)
            try:
                schema = eval(schema)
            except:
                print("schema could not be extracted")
                continue
            superkey = get_superkey(schema)
            each_users_data = self.get_each_users_data_from_text(text)
            associations_list = self.associate_superkeys_with_each_users_data(superkey, each_users_data, idx) #processes all sentences together

            triplets_list = self.generate_triplets(associations_list,text) # processes sentence by sentence
            reduced_triplets_list = self.redundancy_remover_x(triplets_list) # processes sentence by sentence
            # if idx < processed_idx:
            #     SRO_list_for_all_texts[idx] = reduced_triplets_list
            # else:
            SRO_list_for_all_texts.append(reduced_triplets_list)

            # Save the checkpoint
            save_checkpoint(f"{head}/checkpoint.pkl", SRO_list_for_all_texts)

            output_file = s
            save_to_json(self.load_data_path, f"{head}/checkpoint.pkl", output_file)

        sys.stdout = sys.__stdout__           
        f.close()

    def value_identification_llm(self):
        model = self.model
        data = self.data
        num_of_entries = self.num_of_entries
        data = {"texts":data["texts"][:num_of_entries],"predicted_schema":data["schema"][:num_of_entries]}

        os.makedirs("logs/value_identification_results/llm", exist_ok=True)
        f = open("logs/value_identification_results/llm/associations.txt","w")
        sys.stdout = f
        results_path = self.results_path
        datapath = self.datapath
        s = f"{results_path}{datapath}.json"
        head, tail = s.rsplit("/", 1)
        # print(head)
        os.makedirs(head, exist_ok=True)

        checkpoint_name = f"{head}/{self.model_name}_checkpoint.pkl"

        SRO_list_for_all_texts = load_checkpoint(checkpoint_name)
        processed_idx = len(SRO_list_for_all_texts)


        for idx, text in enumerate(data["texts"]):
            if idx < processed_idx:
                print(f"Skipping already processed index {idx}, while the total processed index is {processed_idx}")
                continue
            schema = data["predicted_schema"][idx]
            schema = extract_schema(schema)
            try:
                schema = eval(schema)
            except:
                print("schema could not be extracted")
                continue
            superkey = get_superkey(schema)
            each_users_data = self.get_each_users_data_from_text(text)
            associations_list = self.associate_superkeys_with_each_users_data(superkey, each_users_data, idx) #processes all sentences together
            triplets_list = self.generate_triplets_llm(associations_list,text,schema) # processes sentence by sentence
            # reduced_triplets_list = self.redundancy_remover_y(triplets_list) # processes sentence by sentence

            SRO_list_for_all_texts.append(triplets_list)

            # Save the checkpoint
            save_checkpoint(checkpoint_name, SRO_list_for_all_texts)

        
            output_file = f"{self.results_path}{self.datapath}.json"
            save_to_json(self.load_data_path, checkpoint_name, output_file)


        sys.stdout = sys.__stdout__           
        f.close()

    
            
if __name__ == "__main__":
    experiments = load_config("configs/config.yaml")
    config = experiments['value_identification']
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=False, help="llama or openai or deepseek")
    parser.add_argument("--method", required=False, help="symbolic or llm")
    parser.add_argument("--num_of_entries", type=int, default=
    config['num_of_entries'], help="number of entries to process")
    parser.add_argument("--dataset", required=False, help="bird")
    args = parser.parse_args()

    if args.model_name:
        config['model_name'] = args.model_name
    if args.method:
        config['method'] = args.method
    if args.dataset=="bird":
        config['datapath'] = config['datapath'].replace("SyntheticText/merged_dataset2","BIRD/bird_dataset")
        config['schema_path'] = config['schema_path'].replace("SyntheticText/merged_dataset2","BIRD/bird_dataset")

    config['num_of_entries'] = args.num_of_entries
    config['datapath'] = config['datapath'].replace("<<model_name>>",config['model_name'])

    vip = ValueIdentificationPipeline(config)

    if config['method'] == "llm":
        vip.value_identification_llm()
    elif config['method'] == "symbolic":
        vip.value_identification_symbolic()
        vip.nlp.stop()

