import ast
import json
import os
import re
from utils import load_data, load_config, load_checkpoint, save_checkpoint, extract_from_output
import sqlite3
import argparse
from collections import defaultdict
from utils import extract_schema

import os
import sqlite3
import re

def create_database(db_name, sql_statements):
    print(f"Creating database '{db_name}.db'...")

    # Delete the DB file if it already exists
    if os.path.exists(f"{db_name}.db"):
        os.remove(f"{db_name}.db")

    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.cursor()

    # Preprocess SQL string
    sql = sql_statements
    sql = re.sub(r'\border\b', 'order1', sql)
    sql = re.sub(r"'#'", "NULL", sql)   # Replace quoted '#'
    sql = re.sub(r"\b#\b", "NULL", sql)  # Replace bare #
    sql = re.sub(r",\s*\)", ")", sql)   # Clean trailing commas

    # Split into individual statements (assuming they are separated by semicolons)
    statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]

    for stmt in statements:
        try:
            cursor.execute(stmt)
        except sqlite3.Error as e:
            print(f"Error executing SQL:\n{stmt}\nError: {e}")

    conn.commit()
    conn.close()
    print(f"Database '{db_name}.db' created and populated successfully.")


def generate_mysql_from_schema_and_values_baseline(schema, values):
    sql_statements = []

    # SQL type mapping for base types
    type_mapping = {
        "INTEGER": "INT",
        "TEXT": "VARCHAR(255)",
        "REAL": "FLOAT",
        "DATE": "DATE",
        "TIME": "TIME",
        "BOOLEAN": "BOOLEAN"
    }

    def resolve_type(col_type):
        if re.match(r"^[A-Z]+\s*(\(\d+(,\s*\d+)?\))?$", col_type.strip(), re.IGNORECASE):
            return col_type.upper()
        return type_mapping.get(col_type.upper(), col_type.upper())

    reserved_keywords = ["transaction", "order", "group", "select", "insert", "update", "delete", "from", "where", "join"]

    for table in schema:
        table_name = table['table_name']
        if table_name in reserved_keywords:
            table_name = f"{table_name}1"
        columns = table['columns']

        # CREATE TABLE
        create_stmt = f"CREATE TABLE `{table_name}` (\n"
        col_defs = []
        for col in columns:
            sql_type = resolve_type(col['type'])
            col_line = f"  `{col['name']}` {sql_type}"
            if col.get('primary_key'):
                col_line += " PRIMARY KEY"
            if col.get('foreign_key'):
                foreign_table = col.get('foreign_key_table')
                foreign_column = col.get('foreign_key_column')
                if foreign_table and foreign_column:
                    col_line += f" REFERENCES `{foreign_table}`(`{foreign_column}`)"
            col_defs.append(col_line)
        create_stmt += ",\n".join(col_defs) + "\n);"
        sql_statements.append(create_stmt)

        # INSERT INTO
        # print(values)
                # INSERT INTO
        rows = values.get(table_name, [])

        if not rows:
            continue
        col_names = rows[0].keys()
        for row in rows:
            values_list = []
            for col in col_names:
                val = row[col]
                if isinstance(val, str):
                    val = val.replace("'", "''")
                    values_list.append(f"'{val}'")
                elif val is None:
                    values_list.append("NULL")
                else:
                    values_list.append(str(val))
            insert_stmt = f"INSERT INTO `{table_name}` ({', '.join(f'`{col}`' for col in col_names)}) VALUES ({', '.join(values_list)});"
            sql_statements.append(insert_stmt)


    return "\n\n".join(sql_statements)



def parse_values(idx,values):
    """
    Safely parses a list of strings into Python dictionaries and returns a list of dictionaries.
    """
    parsed_values = []
    
    for value in values:
        try:
            # Extract content between <|start_header_id|> and <|end_header_id|>
            clean_value = value.replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")
            clean_value = clean_value.replace("\'", "\"")
            clean_value = re.sub(r'\bNone\b', 'null', clean_value)
            clean_value = clean_value.replace('": ""', '": null')
            clean_value = re.sub(r'(?<=\s|:|,)(True|False)(?=\s|,|}|\])', lambda m: 'true' if m.group(0) == 'True' else 'false', clean_value)

            
            
            parsed_values.append(json.loads(clean_value))
        except json.JSONDecodeError as e:
            # print(f"\n\nError decoding JSON: {e}")
            # print(f"Clean value: {clean_value}\n")
            # print(f"Invalid JSON: {idx}\n\n")
            pass
            
        
    # Check if parsed_values is empty
    # if len(parsed_values) == 0:
    #     print(f"Error: No valid JSON found in the provided values for index {idx}.\n\n")

    return parsed_values

def generate_mysql_from_schema_and_values(schema, values):
    sql_statements = []

    # SQL type mapping for base types
    type_mapping = {
        "INTEGER": "INT",
        "TEXT": "VARCHAR(255)",
        "REAL": "FLOAT",
        "DATE": "DATE",
        "TIME": "TIME",
        "BOOLEAN": "BOOLEAN"
    }

    def resolve_type(col_type):
        """Resolve column types for MySQL based on schema type"""
        if re.match(r"^[A-Z]+\s*(\(\d+(,\s*\d+)?\))?$", col_type.strip(), re.IGNORECASE):
            return col_type.upper()
        return type_mapping.get(col_type.upper(), col_type.upper())

    # Group values by table name
    values_by_table = defaultdict(list)
    for row in values:
        for table_name, table_data in row.items():
            values_by_table[table_name].extend(table_data)

    # Process each table schema
    for table in schema:
        table_name = table['table_name']
        columns = table['columns']

        # CREATE TABLE
        create_stmt = f"CREATE TABLE `{table_name}` (\n"
        col_defs = []
        primary_keys = [col['name'] for col in columns if col.get('primary_key')]
        for col in columns:
            sql_type = resolve_type(col['type'])
            col_line = f"  `{col['name']}` {sql_type}"
            if col.get('primary_key'):
                col_line += " PRIMARY KEY"
            if col.get('foreign_key'):
                foreign_table = col.get('foreign_key_table')
                foreign_column = col.get('foreign_key_column')
                if foreign_table and foreign_column:
                    col_line += f" REFERENCES `{foreign_table}`(`{foreign_column}`)"
            col_defs.append(col_line)
        create_stmt += ",\n".join(col_defs) + "\n);"
        sql_statements.append(create_stmt)
    
    for table in schema:
        table_name = table['table_name']
        columns = table['columns']

        # INSERT INTO (merged by primary key)
        rows = values_by_table.get(table_name, [])

        if not rows:
            continue

        # Merge rows based on primary key(s)
        merged_rows = {}
        for row in rows:
            key = tuple(row.get(pk) for pk in primary_keys)
            if key not in merged_rows:
                merged_rows[key] = row.copy()
            else:
                # Merge: if a field is missing or None, fill it from the new row
                for col_name in row:
                    if merged_rows[key].get(col_name) is None and row.get(col_name) is not None:
                        merged_rows[key][col_name] = row[col_name]

        # Now generate INSERTs
        value_rows = []
        insert_header = None
        for merged_row in merged_rows.values():
            col_names = merged_row.keys()
            if not insert_header:
                insert_header = f"INSERT INTO `{table_name}` ({', '.join(f'`{col}`' for col in col_names)}) VALUES"
            values_list = []
            for col in col_names:
                val = merged_row[col]
                if isinstance(val, str):
                    val = val.replace("'", "''")
                    values_list.append(f"'{val}'")
                elif val is None:
                    values_list.append("NULL")
                else:
                    values_list.append(str(val))
            value_rows.append(f"({', '.join(values_list)})")

        if value_rows:
            insert_stmt = insert_header + "\n" + ",\n".join(value_rows) + ";"
            sql_statements.append(insert_stmt)

    return "\n\n".join(sql_statements)

def parse_values_baseline(values_str):
    """
    Safely parses the 'values' string into a Python dictionary.
    Handles both single string and list of assistant-generated chunks.
    """
    def clean_chunk(chunk):
        if isinstance(chunk, str):
            clean_value = chunk.replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")
            clean_value = clean_value.replace("\'", "\"")
            clean_value = re.sub(r'\bNone\b', 'null', clean_value)
            clean_value = clean_value.replace('": ""', '": null')
            clean_value = re.sub(r'(?<=\s|:|,)(True|False)(?=\s|,|}|\])', lambda m: 'true' if m.group(0) == 'True' else 'false', clean_value)

            return clean_value
        return str(chunk)

    # Handle list of string chunks or a single string
    if isinstance(values_str, list):
        # Clean and join all assistant responses
        values_clean = ""
        for item in values_str:
            cleaned = clean_chunk(item)
            # remove any trailing commas from partial joins
            values_clean += cleaned.strip().rstrip(',') + ","
        values_clean = values_clean.rstrip(',')  # Final cleanup
    elif isinstance(values_str, str):
        values_clean = clean_chunk(values_str).strip()
    else:
        raise TypeError("Unsupported input type. Expected str or list of str.")

    # Attempt parsing into a dict
    try:
        return ast.literal_eval(values_clean)
    except Exception as e:
        print(f"Could not parse values string: {e}\nCleaned string:\n{values_clean}")
        return {}

def parse_values_baseline_tooluse(values_str, schema):
    return extract_from_output(values_str, schema)
    


def generate_intermediate_queries(schema, domain, method):
    query = ""
    if domain == "tourism":
        query = """SELECT
  traveler.id AS traveler_id,
  traveler.name AS traveler_name,
  traveler.age AS traveler_age,
  traveler.gender AS traveler_gender,
  traveler.nationality AS traveler_nationality,
  
  destination.id AS destination_id,
  destination.city AS destination_city,
  destination.country AS destination_country,

  trip.id AS trip_id,
  trip.start_date,
  trip.end_date,
  trip.duration_days,

  accommodation.id AS accommodation_id,
  accommodation.type AS accommodation_type,
  accommodation.cost AS accommodation_cost,

  transportation.id AS transportation_id,
  transportation.mode AS transportation_mode,
  transportation.cost AS transportation_cost

FROM trip
JOIN traveler ON trip.traveler_id = traveler.id
JOIN destination ON trip.destination_id = destination.id
LEFT JOIN accommodation ON accommodation.trip_id = trip.id
LEFT JOIN transportation ON transportation.trip_id = trip.id
LIMIT 50;
"""
    elif domain == "finance":
        query = """SELECT
  customer.id AS customer_id,
  customer.name AS customer_name,
  customer.age,
  customer.location_type,
  customer.income_level,

  payment_method.id AS payment_method_id,
  payment_method.method_name AS payment_method_name,

  transaction_summary.id AS transaction_id,
  transaction_summary.transaction_count,
  transaction_summary.total_spend,
  transaction_summary.active_days,
  transaction_summary.loyalty_points,
  transaction_summary.cashback_amount,
  transaction_summary.satisfaction_score

FROM transaction_summary
JOIN customer ON transaction_summary.customer_id = customer.id
JOIN payment_method ON transaction_summary.payment_method_id = payment_method.id
LIMIT 50;
"""
    elif domain == "healthcare":
        query = """SELECT
  medication.id AS medication_id,
  medication.proprietary_name,
  medication.generic_name,
  medication.formulary_code,
  medication.gsn,
  medication.ndc,
  medication.product_strength,
  medication.form_unit AS medication_form_unit,

  medication_period.id AS medication_period_id,
  medication_period.start_date,
  medication_period.end_date,
  medication_period.dose_value,
  medication_period.dose_unit,
  medication_period.form_value,
  medication_period.form_unit AS period_form_unit,

  medication_type.id AS medication_type_id,
  medication_type.type_name,

  medication_administration_route.id AS route_id,
  medication_administration_route.route_name

FROM medication_period
JOIN medication ON medication_period.medication_id = medication.id
JOIN medication_type ON medication_period.medication_type_id = medication_type.id
JOIN medication_administration_route ON medication_period.route_id = medication_administration_route.id
LIMIT 50;
"""
    elif domain == "education":
        query = """SELECT
  person.id AS person_id,
  person.name AS person_name,
  person.age,
  person.race,
  person.country,

  education.id AS education_id,
  education.degree,
  education.completed,

  marital_status.id AS marital_status_id,
  marital_status.status,
  marital_status.spouse_name,

  occupation.id AS occupation_id,
  occupation.title AS occupation_title,
  occupation.hours_per_week,
  occupation.salary,
  occupation.salary_range

FROM person
LEFT JOIN education ON education.person_id = person.id
LEFT JOIN marital_status ON marital_status.person_id = person.id
LEFT JOIN occupation ON occupation.person_id = person.id
LIMIT 50;
"""
  

    elif domain == "california_schools":
        query = """SELECT
    school.id AS school_id,
    school.name AS school_name,
    school.code AS school_code,
    school.cdscode AS school_cdscode,
    school.address AS school_address,
    school.city AS school_city,
    school.zip_code AS school_zip_code,
    school.state AS school_state,
    school.phone AS school_phone,
    school.mail_address AS school_mail_address,
    school.mail_city AS school_mail_city,
    school.mail_zip_code AS school_mail_zip_code,
    school.mail_state AS school_mail_state,
    school.open_date AS school_open_date,
    school.school_type AS school_school_type,
    school.educational_option_type AS school_educational_option_type,
    district.id AS district_id,
    district.name AS district_name,
    district.code AS district_code,
    district.district_type AS district_district_type,
    county.id AS county_id,
    county.name AS county_name,
    county.code AS county_code
FROM school
JOIN district ON school.district_id = district.id
JOIN county ON district.county_id = county.id
LIMIT 50;
"""
    elif domain == "financial":
        query = """SELECT
    transaction1.transaction_id AS transaction_transaction_id,
    transaction1.transaction_date AS transaction_transaction_date,
    transaction1.amount AS transaction_amount,
    transaction1.currency AS transaction_currency,
    transaction1.balance_after AS transaction_balance_after,
    transaction1.issued_date AS transaction_issued_date,
    account.account_id AS account_account_id,
    account.account_number AS account_account_number,
    bank.bank_id AS bank_bank_id,
    bank.bank_name AS bank_bank_name,
    person.person_id AS person_person_id,
    person.birth_date AS person_birth_date,
    person.gender AS person_gender,
    transaction_type.transaction_type_id AS transaction_type_transaction_type_id,
    transaction_type.type_name AS transaction_type_type_name,
    transaction_type.type_code AS transaction_type_type_code,
    payment_frequency.frequency_id AS payment_frequency_frequency_id,
    payment_frequency.frequency_name AS payment_frequency_frequency_name
FROM transaction1
JOIN account ON transaction1.account_id = account.account_id
JOIN person ON account.person_id = person.person_id
JOIN bank ON account.bank_id = bank.bank_id
JOIN transaction_type ON transaction1.transaction_type_id = transaction_type.transaction_type_id
JOIN payment_frequency ON transaction1.frequency_id = payment_frequency.frequency_id
LIMIT 50;
"""
    elif domain == "superhero":
        query = """SELECT
    superhero.id AS superhero_id,
    superhero.name AS superhero_name,
    superhero.alias AS superhero_alias,
    superhero.height_cm AS superhero_height_cm,
    superhero.weight_kg AS superhero_weight_kg,
    superhero.gender AS superhero_gender,
    superhero.hair_color AS superhero_hair_color,
    superhero.alignment AS superhero_alignment,
    universe.id AS universe_id,
    universe.name AS universe_name,
    superhero_species.id AS superhero_species_id,
    species.id AS species_id,
    species.name AS species_name
FROM superhero
JOIN universe ON superhero.universe_id = universe.id
JOIN superhero_species ON superhero.id = superhero_species.superhero_id
JOIN species ON superhero_species.species_id = species.id
LIMIT 50;
"""
    elif domain == "law_episode":
        query = """SELECT
    episode.id AS episode_id,
    episode.series_id AS episode_series_id,
    episode.title AS episode_title,
    episode.season_number AS episode_season_number,
    episode.episode_number AS episode_episode_number,
    episode.overall_episode_number AS episode_overall_episode_number,
    episode.air_date AS episode_air_date,
    episode.rating AS episode_rating,
    episode.vote_count AS episode_vote_count,
    tv_series.id AS tv_series_id,
    tv_series.title AS tv_series_title,
    
    episode_crew.id AS episode_crew_id,
    person_crew.id AS person_crew_id,
    person_crew.name AS person_crew_name,
    role.id AS role_id,
    role.name AS role_name,
    episode_crew.is_credited AS episode_crew_is_credited,

    award_nomination.id AS award_nomination_id,
    award_nomination.year AS award_nomination_year,
    award_nomination.is_winner AS award_nomination_is_winner,
    award_category.id AS award_category_id,
    award_category.name AS award_category_name,

    episode_cast.id AS episode_cast_id,
    person_cast.id AS person_cast_id,
    person_cast.name AS person_cast_name,
    character.id AS character_id,
    character.name AS character_name,
    character.description AS character_description
FROM episode
JOIN tv_series ON episode.series_id = tv_series.id
LEFT JOIN episode_crew ON episode.id = episode_crew.episode_id
LEFT JOIN person AS person_crew ON episode_crew.person_id = person_crew.id
LEFT JOIN role ON episode_crew.role_id = role.id
LEFT JOIN award_nomination ON episode_crew.id = award_nomination.episode_crew_id
LEFT JOIN award_category ON award_nomination.award_category_id = award_category.id
LEFT JOIN episode_cast ON episode.id = episode_cast.episode_id
LEFT JOIN person AS person_cast ON episode_cast.person_id = person_cast.id
LEFT JOIN character ON episode_cast.character_id = character.id
LIMIT 50;
"""
    elif domain == "books":
        query = """SELECT
    book.id AS book_id,
    book.isbn AS book_isbn,
    book.title AS book_title,
    book.page_count AS book_page_count,
    book.publication_date AS book_publication_date,
    book.language AS book_language,
    book_author.id AS book_author_id,
    author.id AS author_id,
    author.name AS author_name
FROM book
JOIN book_author ON book.id = book_author.book_id
JOIN author ON book_author.author_id = author.id
LIMIT 50;"""
    elif domain == "computer_student":
        query = """SELECT
    professor.id AS professor_id,
    professor.name AS professor_name,
    professor.experience_years AS professor_experience_years,
    professor.faculty_status AS professor_faculty_status,
    professor.phase AS professor_phase,
    
    student.id AS student_id,
    student.name AS student_name,
    student.experience_years AS student_experience_years,
    student.phase AS student_phase,

    course.id AS course_id,
    course.level AS course_level,
    course.title AS course_title,

    teaching_assignment.id AS teaching_assignment_id,
    teaching_assignment.professor_id AS teaching_assignment_professor_id,
    teaching_assignment.course_id AS teaching_assignment_course_id,

    enrollment.id AS enrollment_id,
    enrollment.student_id AS enrollment_student_id,
    enrollment.course_id AS enrollment_course_id

FROM enrollment
JOIN student ON enrollment.student_id = student.id
JOIN course ON enrollment.course_id = course.id
LEFT JOIN teaching_assignment ON teaching_assignment.course_id = course.id
LEFT JOIN professor ON teaching_assignment.professor_id = professor.id
LIMIT 50;
"""
    elif domain == "mental_health_survey":
        query = """SELECT
    person.id AS person_id,
    person.name AS person_name,
    age_record.id AS age_record_id,
    age_record.person_id AS age_record_person_id,
    age_record.age AS age_record_age,
    age_record.timestamp AS age_record_timestamp

FROM age_record
JOIN person ON age_record.person_id = person.id
LIMIT 50;"""
    elif domain == "authors":
        query = """SELECT
    researcher.id AS researcher_id,
    researcher.full_name AS researcher_full_name,
    researcher.short_name AS researcher_short_name,
    researcher.homepage_url AS researcher_homepage_url,

    institution.id AS institution_id,
    institution.name AS institution_name,
    institution.department AS institution_department,
    institution.address AS institution_address,

    researcher_institution.id AS researcher_institution_id,
    researcher_institution.researcher_id AS researcher_institution_researcher_id,
    researcher_institution.institution_id AS researcher_institution_institution_id,

    publication.id AS publication_id,
    publication.title AS publication_title,
    publication.short_name AS publication_short_name,
    publication.homepage_url AS publication_homepage_url,

    researcher_publication.id AS researcher_publication_id,
    researcher_publication.researcher_id AS researcher_publication_researcher_id,
    researcher_publication.publication_id AS researcher_publication_publication_id

FROM researcher
LEFT JOIN researcher_institution ON researcher.id = researcher_institution.researcher_id
LEFT JOIN institution ON researcher_institution.institution_id = institution.id
LEFT JOIN researcher_publication ON researcher.id = researcher_publication.researcher_id
LEFT JOIN publication ON researcher_publication.publication_id = publication.id
LIMIT 50;"""


    return [query]


def generate_mysql_for_all_entries(config):
    
    method = config['method']
    base_data_path = config['base_data_path']
    results_dir = config['results_dir']
    logs_path = config['logs_dir']
    num_of_entries = config['num_of_entries']
    model_name = config['model_name']
    dataset = "SyntheticText/merged_dataset2"

    parser = argparse.ArgumentParser(description="Evaluate database generation.")
    parser.add_argument("--method", type=str, required=False, help="Method used for database generation.")
    parser.add_argument("--model_name", type=str, required=False, help="Model name used for database generation.")
    parser.add_argument("--dataset", type=str, required=False, help="bird")
    args = parser.parse_args()
    if args.method:
        method = args.method
    if args.model_name:
        model_name = args.model_name
    if args.dataset == "bird":
        config['datapath'] = config['datapath'].replace("SyntheticText/merged_dataset2","BIRD/bird_dataset")
        dataset = "BIRD/bird_dataset"
    
    config['datapath'] = config['datapath'].replace("<<model_name>>",model_name)
    datapath = config['datapath']

    schema_path = f"schema/{dataset}/schema_mapping.json"
    schema_dict = json.load(open(schema_path))

    path = f"{base_data_path}{method}/{datapath}.json"
    print(f"Loading data from {path}")
    data = load_data(path)
    print(f"Loaded {len(data['texts'])} entries from {path}")
    # for i in data:
    #     print(i)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}{method}/{datapath}/", exist_ok=True)
    updated_json = load_checkpoint(f"{results_dir}{method}/{datapath}/checkpoint.pkl")
    updated_json = []
    processed_idx = len(updated_json)
    for idx,text in enumerate(data['texts']):
        # if idx < processed_idx:
        #     continue
        if num_of_entries != -1 and idx >= num_of_entries:
            break
        entry = {}
        entry['text'] = text
        entry['schema'] = data['schema'][idx] 
        entry['domain'] = data['domain'][idx]
        entry['difficulty'] = data['difficulty'][idx]
        entry['ground_truth_entities'] = data['ground_truth_entities'][idx]
        entry['ground_truth_key_value'] = data['ground_truth_key_value'][idx]
        entry['identified_values'] = data['identified_values'][idx]
        entry['output'] = data['output'][idx]

        # schema = schema_dict[entry['domain']]
        # schema = extract_schema(schema)
        # try:
        #     schema = eval(schema)
        # except:
        #     pass
        schema = entry['schema']

        # if method == "baseline":
        #     # values = parse_values_baseline(entry['output'])
        #     values = parse_values_baseline_tooluse(entry['output'], schema)
        #     try:
        #         values = values[0]
        #     except:
        #         pass
        #     entry['sql_statements'] = generate_mysql_from_schema_and_values_baseline(schema, values)
            
        # else:
        #     values_all = []
        #     for item in entry['output']:
        #         print(f"Item: {item}")
        #         values = parse_values_baseline_tooluse(item, schema)
        #         values_all.append(values)
        #         print(f"Parsed values: {values}\n\n\n\n")
        #     entry['sql_statements'] = generate_mysql_from_schema_and_values(schema, values_all)
            
        values = parse_values_baseline_tooluse(entry['output'], schema)
        # print(f"Parsed values: {values}\n\n\n\n")
        entry['sql_statements'] = generate_mysql_from_schema_and_values_baseline(schema, values)

        os.makedirs(f"databases/{datapath}/{method}/", exist_ok=True)
        try:
            # if database already exists, skip
            if os.path.exists(f"databases/{datapath}/{method}/{entry['domain']}_{idx}.db"):
                pass
            else:
                create_database(f"databases/{datapath}/{method}/{entry['domain']}_{idx}", entry['sql_statements'])
        except Exception as e:
            print(f"Error creating database for entry {idx}: {e}")
            continue
        entry['join_query'] = generate_intermediate_queries(schema, entry['domain'], method)
        entry['db_name'] = f"{entry['domain']}_{idx}"
        updated_json.append(entry)


        save_checkpoint(f"{results_dir}{method}/{datapath}/checkpoint.pkl",updated_json)
        # print(updated_json)

        with open(f"{results_dir}{method}/{datapath}.json", 'w') as f:
            json.dump(updated_json, f, indent=2)


if __name__ == "__main__":
    experiments = load_config("configs/config.yaml")
    config = experiments["database_generation"]
    generate_mysql_for_all_entries(config)

    