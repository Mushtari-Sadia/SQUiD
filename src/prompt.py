import warnings
warnings.filterwarnings('ignore')
from utils import *


def get_schema_prompt_with_cot_and_triplets(triplets):
    system_prompt = """You are an expert at creating database schemas from <subject, relation, object> triplets. I have extracted structured triplets from text in the form of <subject, relation, object>.  
Using these triplets, generate a **normalized relational database schema** in Python dictionary format.  

---

### **Step-by-Step Guide for Schema Creation (Follow This Chain of Thought)**  

**Requirements Analysis**:  
- Identify all distinct entities and attributes from the triplets.  
- Determine necessary tables and their columns.  

**Entity-Relationship (ER) Modeling**:  
- Identify entity relationships (One-to-One, One-to-Many, Many-to-Many).  
- If applicable, use associative tables for Many-to-Many relationships.  

**Define Tables and Columns**:  
- Convert entities into relational tables with appropriate **data types**.  

**Establish Primary Keys**:  
- Assign a **Primary Key (PK)** for each table to uniquely identify records.  

**Define Relationships and Foreign Keys**:  
- Use **Foreign Keys (FK)** to enforce referential integrity between tables.  
- Apply **ON DELETE CASCADE** if necessary to maintain consistency.  

**Normalization (1NF → 2NF → 3NF)**:  
- Ensure atomic values (1NF).  
- Remove partial dependencies (2NF).  
- Eliminate transitive dependencies (3NF).  

**Define Constraints**:  
- Apply **NOT NULL**, **UNIQUE**, **CHECK**, and other constraints as needed.  

**Indexing for Performance**:  
- Create indexes on frequently queried columns (e.g., search fields).   

**Column and Table name restriction**:
reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or",
        "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
- Ensure that the table names and column names do not contain any SQL reserved keywords.

---
### **Task Instructions:**  
- **Step through the schema creation process using the above guide**.  
- **Generate a well-structured, normalized relational database schema**.  
- **Output only the final schema** in Python dictionary format (NO explanations). 
- **Column and Table name restriction**:
reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or",
        "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
- Ensure that the table names and column names do not contain any SQL reserved keywords. 

"""
    user_prompt = f"""
--- 
### **Triplets Extracted from Text:**
{triplets}

### **Expected Output Format (Strictly Follow This Example Structure)**:
schema = [
    {{
        "table_name": "student",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "name", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "course",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "title", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "enrollment",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "student_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "student", "foreign_key_column": "id"}},
            {{"name": "course_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "course", "foreign_key_column": "id"}}
        ]
    }}
]

Now output the schema as per the system instructions.
### Output:
"""
    return system_prompt, user_prompt

def get_schema_prompt_with_direct_and_triplets(triplets):
    system_prompt = """You are an expert at creating database schemas from <subject, relation, object> triplets. I have extracted structured triplets from text in the form of <subject, relation, object>. 
Using these triplets, your task is to generate a relational database schema in JSON format. 

### **Task:**
1. **Extract Entities & Relationships**: Identify unique entity types and relationships.
2. **Determine Attributes**: Define necessary columns for each table.
3. **Normalize the Schema**: Ensure proper **primary keys, foreign keys, and normalization (3NF)**.
4. **Generate Output in JSON Format**.
5. **Column and Table name restriction**:
reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or",
        "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
- Ensure that the table names and column names do not contain any SQL reserved keywords.

"""
    user_prompt = f"""
### **Triplets:**
{triplets}

### **Expected Example Output Format (Strictly Follow This Structure)**:
schema = [
    {{
        "table_name": "student",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "name", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "course",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "title", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "enrollment",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "student_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "student", "foreign_key_column": "id"}},
            {{"name": "course_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "course", "foreign_key_column": "id"}}
        ]
    }}
]



Now output the schema as per the system instructions.
### Output:
"""
    return system_prompt, user_prompt



def get_schema_prompt_with_cot_and_text(text):
    system_prompt = """You are an expert at formulating database schemas from textual data. I have given you a paragraph of text. 
Using this text, your task is to generate a relational database schema in JSON format.   

---

### **Step-by-Step Guide for Schema Creation (Follow This Chain of Thought)**  

**Requirements Analysis**:  
- Identify all distinct entities and attributes from the text.  
- Determine necessary tables and their columns.  

**Entity-Relationship (ER) Modeling**:  
- Identify entity relationships (One-to-One, One-to-Many, Many-to-Many).  
- If applicable, use associative tables for Many-to-Many relationships.  

**Define Tables and Columns**:  
- Convert entities into relational tables with appropriate **data types**.  

**Establish Primary Keys**:  
- Assign a **Primary Key (PK)** for each table to uniquely identify records.  

**Define Relationships and Foreign Keys**:  
- Use **Foreign Keys (FK)** to enforce referential integrity between tables.  
- Ensure that it is possible to join all tables to create one flat table using the foreign keys.
- Apply **ON DELETE CASCADE** if necessary to maintain consistency.  

**Normalization (1NF → 2NF → 3NF)**:  
- Ensure atomic values (1NF).  
- Remove partial dependencies (2NF).  
- Eliminate transitive dependencies (3NF).  

**Define Constraints**:  
- Apply **NOT NULL**, **UNIQUE**, **CHECK**, and other constraints as needed.  

**Indexing for Performance**:  
- Create indexes on frequently queried columns (e.g., search fields).   

- **Column and Table name restriction**:
reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or",
        "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
- Ensure that the table names and column names do not contain any SQL reserved keywords. 

---
### **Task Instructions:**  
- **Step through the schema creation process using the above guide**.  
- **Generate a well-structured, normalized relational database schema**.  
- **Output only the final schema** in Python dictionary format (NO explanations).
- **Column and Table name restriction**:
reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or",
        "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
- Ensure that the table names and column names do not contain any SQL reserved keywords. 

"""
    user_prompt = f"""
### **Text:**
{text}

### **Expected Example Output Format (Strictly Follow This Structure while modifying the table_names, column_names to match the given text)**:
schema = [
    {{
        "table_name": "student",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "name", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "course",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "title", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "enrollment",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "student_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "student", "foreign_key_column": "id"}},
            {{"name": "course_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "course", "foreign_key_column": "id"}}
        ]
    }}
]

Now output the schema as per the system instructions.
### Output:
"""
    return system_prompt, user_prompt

def get_schema_prompt_with_direct_and_text(text):
    system_prompt = """You are an expert at formulating database schemas from textual data. I have given you a paragraph of text. 
Using this text, your task is to generate a relational database schema in JSON format. 

### **Task:**
1. **Extract Entities & Relationships**: Identify unique entity types and relationships.
2. **Determine Attributes**: Define necessary columns for each table.
3. **Normalize the Schema**: Ensure proper **primary keys, foreign keys, and normalization (3NF)**.
4. **Generate Output in JSON Format**.
5. **Column and Table name restriction**:
reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or",
        "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
- Ensure that the table names and column names do not contain any SQL reserved keywords. 
"""
    user_prompt = f"""
### **Text:**
{text}

### **Expected Example Output Format (Strictly Follow This Structure while modifying the table_names, column_names to match the given text)**:
schema = [
    {{
        "table_name": "student",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "name", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "course",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "title", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "enrollment",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "student_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "student", "foreign_key_column": "id"}},
            {{"name": "course_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "course", "foreign_key_column": "id"}}
        ]
    }}
]


Now output the schema as per the system instructions.
### Output:
"""
    return system_prompt, user_prompt


def schema_validation_prompt(schema):
    system_prompt = """  
You are a database normalization expert. Your task is to analyze a relational database schema and determine whether it adheres to First Normal Form (1NF), Second Normal Form (2NF), and Third Normal Form (3NF).  

Follow these structured steps:  

### **Step 1: First Normal Form (1NF) Check**  
- Ensure all columns have atomic (indivisible) values.  
- Verify that each column contains values of a **single type** (no repeating groups or arrays).  

### **Step 2: Second Normal Form (2NF) Check**  
- Confirm the schema is in 1NF.  
- Ensure **no partial dependency exists**, meaning all **non-key attributes must depend on the entire primary key**, not just part of it. (Applies to composite primary keys.)  

### **Step 3: Third Normal Form (3NF) Check**  
- Confirm the schema is in 2NF.  
- Identify **functional dependencies (FDs)** where a **non-key attribute depends on another non-key attribute** instead of the primary key (transitive dependency).  
- Ensure that **all non-key attributes depend ONLY on the primary key** and not on other non-key attributes.  

Your response should follow this structured output:  
- **1NF Status:** ✅ / ❌ (With explanation)  
- **2NF Status:** ✅ / ❌ (With explanation)  
- **3NF Status:** ✅ / ❌ (With explanation)  
- **Final Verdict:** (Is the schema normalized? If not, what changes are required?)  

---

### **In-Context Example: Checking a Schema for 1NF, 2NF, and 3NF**  

#### **Input Schema:**
{
    "tables": [
        {
            "name": "employees",
            "columns": [
                {"name": "emp_id", "type": "INTEGER", "primary_key": true},
                {"name": "name", "type": "TEXT"},
                {"name": "projects", "type": "TEXT"},  
                {"name": "department_id", "type": "INTEGER", "foreign_key": "departments(dept_id)"},
                {"name": "department_name", "type": "TEXT"}
            ]
        },
        {
            "name": "departments",
            "columns": [
                {"name": "dept_id", "type": "INTEGER", "primary_key": true},
                {"name": "dept_name", "type": "TEXT"},
                {"name": "location", "type": "TEXT"}
            ]
        }
    ]
}
Expected Analysis:
✅ 1NF Check: ❌ Fails 1NF because projects contains multiple values (not atomic). First Normal Form (1NF) requires that each column should contain atomic values, meaning each cell should hold a single piece of data, not a list or multiple values.

✅ 2NF Check: ❌ Fails 2NF because department_name is only dependent on department_id, not emp_id. Second Normal Form (2NF) requires that all non-key attributes depend on the entire primary key, not just part of it. Here, department_name only depends on department_id, not emp_id, violating 2NF.

✅ 3NF Check: ❌ Fails 3NF because department_name depends on department_id, which is not the primary key of the employees table (transitive dependency). Third Normal Form (3NF) requires that non-key attributes depend only on the primary key and not on other non-key attributes. Here, department_name depends on department_id, making it a transitive dependency.

Now, analyze the following schema:
"""
    user_prompt = f"""
Analyze the following relational database schema and determine whether it follows 1NF, 2NF, and 3NF.
{schema}
Expected Output Format:
1NF Check: ✅ / ❌ Explanation
2NF Check: ✅ / ❌ Explanation
3NF Check: ✅ / ❌ Explanation
Final Verdict:
Is the schema normalized?
If not, what tables need restructuring?
"""
    return system_prompt, user_prompt

def get_superkey_association_prompt(text, schema, superkey, paragraphs):
    system_prompt = f"""
You are a helpful assistant that who assists a user with information extraction tasks. 
Your job is to associate a unique superkey value with each paragraph in the text. 
You will be given multiple paragraphs of text, a database schema, and a superkey. 
Your task is to associate the superkey value with each paragraph in the text. Each paragraph MUST be associated with a superkey value. No two superkey values should be the same.
Fill in the <FILL IN WITH APPROPRIATE VALUE OF {superkey}> with the value. You will not provide code or SQL, you will do the task yourself."""
    user_prompt = f"""
**Text**:
{text}

**Schema**:
{schema}

**Superkey**:
{superkey}

**Paragraphs**:
"""
    user_prompt += "\n--\n"
    for j in range(len(paragraphs)):
        user_prompt += f"paragraph {j}: {paragraphs[j]}\n"
        user_prompt += f"associated superkey: <FILL IN WITH APPROPRIATE VALUE OF {superkey}>\n"
        user_prompt += "\n--\n"

    return system_prompt, user_prompt




def get_triplet_superkey_prompt(text, schema, triplets):
    system_prompt = """You are an expert in relational databases and information extraction. Your task is to augment (subject, relation, object) triplets with unique entity context using a database schema.

Given:
- A paragraph of text
- A relational database schema
- Triplets extracted from the text in subject, relation, object format

Follow this reasoning chain:
1. Identify a **super key** (a set of attributes that uniquely identify an entity) for each relevant table in the schema.
2. Use the **text** and **schema** to extract values that can serve as the super key for disambiguation.
3. **Append** the super key values to each triplet using the format:
   `(subject, relation, object, superkey={...})`


"""

    user_prompt = f"""
### Text:
{text}

### Schema:
{schema}

### Extracted Triplets:
{triplets}

### Chain-of-Thought and Expected Output Format:

1. Identify superkeys (from schema and/or context in the text).
2. Extract superkey values from the text.
3. Append superkey values to each triplet like this:
   {{"subject":subject, "relation":relation, "object":object, superkey={{"attribute1": "value1", "attribute2": "value2"}}}})

### Output:
"""
    return system_prompt, user_prompt



def get_unique_users_prompt(text):
    system_prompt = """You are an expert in extracting primary users from textual paragraphs of their data. Your task is to identify and list all unique primary users mentioned in this given paragraph of text about their data."""

    user_prompt = """
    ### **Text:**
    {text}
    ### **Output:**
    """
    return system_prompt, user_prompt

def identify_candidate_key(text, schema):
    system_prompt = """You are an expert in identifying keys from a database schema that can uniquely identify an individ from textual paragraphs of their data. Your task is to identify and list all unique primary users mentioned in this given paragraph of text about their data."""

    user_prompt = """
    ### **Text:**
    {text}
    ### **Output:**
    """
    return system_prompt, user_prompt

# helper for get_value_population_prompt()
def generate_empty_data_template(schema, rows_per_table=3):
    data = {}

    for table in schema:
        table_name = table["table_name"]
        columns = [col["name"] for col in table["columns"]]

        # Build row templates (with empty values)
        rows = []
        for i in range(rows_per_table):
            row = {col: "#" for col in columns}
            rows.append(row)

        data[table_name] = rows

    return data


def generate_empty_data_template_tooluse(schema,triplets=False,output=[]):
    """
    Generate tool use instructions for extracting data from the database.
    """
    sqltype_to_python_type = {
        "INTEGER": "int",
        "TEXT": "str",
        "REAL": "float",
        "BOOLEAN": "bool",
        "DATE": "str",
        "DATETIME": "str",
        'DECIMAL(10,2)': "float",
        'FLOAT': "float",
    }

    table_instructions = []
    for table in schema:
        tool_name = table["table_name"]
        table_instruction = f" - {tool_name}: "
        for col in table["columns"]:
            try:
                col_type = sqltype_to_python_type[col["type"]]
            except:
                col_type = 'str'
            
            # Annotate primary key
            pk = " [PK]" if col.get("primary_key") else ""
            
            # Annotate foreign key
            if col.get("foreign_key"):
                fk_table = col.get("foreign_key_table")
                fk_col = col.get("foreign_key_column")
                fk = f" [FK → {fk_table}({fk_col})]"
            else:
                fk = ""
            
            table_instruction += f"\"{col['name']}: {col_type}{pk}{fk}\", "
        
        table_instructions.append(table_instruction.rstrip(", "))  # Remove trailing comma

        # print(table_instruction)
      
    table_instruction_str = ""
    for table_instruction in table_instructions:
        table_instruction_str += table_instruction + "\n"
    # print(table_instruction_str)
    if triplets:
        base_tool_use_instructions = f"""
Extract information using the extraction tool -- "extract".

Example:
Python table schema: 
    - person: "id: int [PK]", "name: string", "age: int", "location: string", "married: boolean"
    - job: "id: int [PK]", "person_id: int [FK → person(id)]", "title: string", "salary: int", "department: string"


Example triplets:
    < 1 , person, name, John >
    < 1 , person, age, 30 >
    < 1 , person, location, Los Angeles >
    < 1 , person, married, false >
    < 1 , job, title, Software Engineer >
    < 1 , job, salary, 100000 >
    < 1 , job, department, Engineering >
    
Example usage: 
    extract person: "id": 1; "name": "John"; "age" 30; "location": "Los Angeles"; "married": false
    extract job: "id": 1; "person_id": 1; "title": "Software Engineer"; "salary": 100000; "department": "Engineering"

Tool guidelines:
- "id" should always be present in the extraction. It is the primary key of the table. You must assign an appropriate value to it. Do not leave it empty.
- "xxx_id" is the foreign key of the table and references the primary key of a row in another table. It must also be present in the extraction, you must assign an appropriate value to it, and ensure that the value matches the id of the row in it's foreign key parent table. Do not leave it empty.
- You must follow the order of the columns per table, defined in the Python table schema.
- You must follow the data type for each column.
- You must use "extract" once per table of the schema to extract information from the whole paragraph and the set of triplets for each of the table.
- After reading the paragraph, the schema and the triplets, do all the extraction steps in one go, do not skip any extraction steps. Do not repeat extraction for the same id if there are no new information.
- Do not extract any information that is not present in the original paragraph.
- Other than "id" and "xxx_id", if no value is found for a column, use "?" to represent the value. If all columns are empty, do not use extract for that row.

Python table schema:
{table_instruction_str}
"""
        if len(output) > 0:
            base_tool_use_instructions += f"""
Already extracted information from previous paragraphs:
"""     
            for item in output:
                newline = ord('\n')
                base_tool_use_instructions += f"""
{item.replace(f"<|start_header_id|>assistant<|end_header_id|>{newline}{newline}", "")}
"""
    else:
        base_tool_use_instructions = f"""
Extract information using the extraction tool -- "extract".

Example:
Python table schema: 
    - person: "id: int [PK]", "name: string", "age: int", "location: string", "married: boolean"
    - job: "id: int [PK]", "person_id: int [FK → person(id)]", "title: string", "salary: int", "department: string"

Example usage: 
    extract person: "id": 1; "name": "John"; "age": 30; "location": "Los Angeles"; "married": false
    extract person: "id": 2; "name": "Jane"; "age": 25; "location": "New York"; "married": true
    extract job: "id": 1; "person_id": 1; "title": "Software Engineer"; "salary": 100000; "department": "Engineering"
    extract job: "id": 2; "person_id": 2; "title": "Data Scientist"; "salary": 90000; "department": "Engineering"

Tool guidelines:
- "id" should always be present in the extraction. It is the primary key of the table. You must assign an appropriate value to it. 
- "xxx_id" is the foreign key of the table and references the primary key of a row in another table. It must also be present in the extraction, you must assign an appropriate value to it, and ensure that the value matches the id of the row in it's foreign key parent table. 
- You must follow the order of the columns per table, defined in the Python table schema.
- You must follow the data type for each column.
- You must use "extract" once per table of the schema to extract information from the whole paragraph and the set of triplets for each of the table.
- Do not extract any information that is not present in the original text.
- Other than "id" and "xxx_id", if no value is found for a column, use "?" to represent the value. If all columns are empty, do not use extract for that row.

Python table schema:
{table_instruction_str}
"""

    return base_tool_use_instructions


def get_value_population_prompt_TS(entry):
    text = entry['text']
    schema = entry['schema']
    # data_template = str(generate_empty_data_template(schema))
    data_template = generate_empty_data_template_tooluse(schema)
    
    system_prompt = f"""You are an expert at populating values in a database from text based on a given database schema. I have provided a paragraph of text and a database schema. Using this information, your task is to extract relevant values and format them according to the schema. Do not provide code, do the task yourself."""
    user_prompt = f"""
### **Text:**
{text}
### **Schema:**
{schema}
### **Expected Output Format:**
{data_template}
Output as many rows as necessary to populate the data. Replace the '#' with the actual values from the text.

You will follow this chain-of-thought reasoning to generate the final output:
- Generate output entries relevant to the text.
- Follow the given output format strictly. Do not add any additional explanations or comments. Only output the data entries in given format. Do not provide code, do the task yourself.
### **Output:**
"""
    return system_prompt, user_prompt

# for each text
def get_value_population_prompt(data):
    text = data['text']
    schema = data['schema']
    superkey = data['superkey']
    # data_template = str(generate_empty_data_template(schema))
    data_template = generate_empty_data_template_tooluse(schema)
    identified_values = data['identified_values']

    system_prompt = f"""You are a database assistant that extracts data from each sentence of a given text, and populates data entries into a fixed relational database schema.

Text:
{text}

Schema:
{schema}

Superkey:
{superkey}

The user will give you the following inputs: a paragraph of the user's data, superkey of that paragraph, extracted <{superkey} | subject | relation | object > triplets from that paragraph.

Output format:
{data_template}
Output as many rows as necessary to populate the data. Replace the '#' with the actual values from the sentence.

You will follow this chain-of-thought reasoning to generate the final output:
- Generate output entries relevant to the current sentence. Fill in the superkey value as well.
- Validate each output entry by going through the triplets one by one, and ensure every unique data point from the triplets is captured into the correct table and fields.
- Follow the given output format strictly. Do not add any additional explanations or comments. Only output the data entries in given format. Do not provide code, do the task yourself.


Example of task:

Example Schema: [
    {{
        "table_name": "countries",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "iso_code", "type": "TEXT"}},
            {{"name": "country_name", "type": "TEXT"}},
            {{"name": "region", "type": "TEXT"}},
            {{"name": "human_development_index", "type": "REAL"}},
            {{"name": "year", "type": "INTEGER"}}
        ]
    }},
    {{
        "table_name": "education_inequality",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "country_id", "type": "INTEGER", "foreign_key": "countries(id)"}},
            {{"name": "year", "type": "INTEGER"}},
            {{"name": "inequality", "type": "REAL"}}
        ]
    }}
]

Example Sentence:
"In 2010, the inequality in education was a staggering 39.4, a number that remained relatively consistent in the following years - 38.55827 in 2011, 2012, and 2013."

Example Triplets:
<inequality | was | staggering>
<2010 | In 39.4 is | number>
<the following years | is in | 2011>
<inequality | is in | education>
<2011 | was | 38.55827>

Example superkey: countries.id: 1

***Example Output by you:
data = {{
    "education_inequality": [
        {{"id": 1, "country_id": 1, "year": 2010, "inequality": 39.4}},
        {{"id": 2, "country_id": 1, "year": 2011, "inequality": 38.55827}},
        {{"id": 3, "country_id": 1, "year": 2012, "inequality": 38.55827}},
        {{"id": 4, "country_id": 1, "year": 2013, "inequality": 38.55827}}
    ]
}}
"""
    user_prompts = []
    for item in identified_values:
        user_prompt = "Please perform the task as per the system instructions.\n"
        user_prompt += f"### Paragraph: {item['paragraph']}\n"
        user_prompt += f"### Extracted <{superkey} | subject | relation | object > triplets from paragraph:\n"
        for triplet in item['triplets']:
            if triplet['subject'] != "subject" and triplet['relation'] != "relation" and triplet['object'] != "object":
                user_prompt += f"<{item['superkey']} | {triplet['subject']} | {triplet['relation']} | {triplet['object']}>\n"
        user_prompt += "\n"
        user_prompt += "### Output:\n"
        user_prompts.append(user_prompt)
    return system_prompt, user_prompts

def get_value_population_prompt_with_llm_triplets(data,final_output):
    text = data['text']
    schema = data['schema']
    superkey = data['superkey']
    # data_template = str(generate_empty_data_template(schema))
    data_template = generate_empty_data_template_tooluse(schema, triplets=True, output=final_output)
    identified_values = data['identified_values']

    system_prompt = f"""You are a database assistant that extracts data from each sentence of a given text, and populates data entries into a fixed relational database schema.

The user will give you the following inputs: a paragraph of the user's data from the given text, superkey of that paragraph, extracted triplets from that paragraph in the following format:
<{superkey}, `table_name`, `column_name`, `value`>

You will follow this chain-of-thought reasoning to generate the final output:
- Generate output entries as per given instructions, relevant to the current paragraph.
- Validate each output entry by going through the triplets one by one, and ensure every unique data point from the triplets is captured into the correct table and fields.
- Follow the given output format strictly. Do not add any additional explanations or comments. Only output the data entries in given format. Do not provide code, do the task yourself.

Text:
{text}

Output format:
{data_template}

"""

    user_prompts = []
    for item in identified_values:
        user_prompt = "Please perform the task as per the system instructions.\n"
        user_prompt += f"### Paragraph: {item['paragraph']}\n"
        user_prompt += f"### Extracted <{superkey}, `table_name`, `column_name`, `value`> triplets from paragraph:\n"
        for triplet in item['triplets']:
            if triplet['value'] !="not mentioned" or triplet['value'] !="not provided":
                user_prompt += f"<{item['superkey']}, {triplet['table_name']}, {triplet['column_name']}, {triplet['value']}>\n"
        user_prompt += "\n"
        user_prompt += "### Output:\n"
        user_prompts.append(user_prompt)
    return system_prompt, user_prompts

def get_value_population_prompt_TST(data):
    text = data['text']
    schema = data['schema']
    superkey = data['superkey']
    # data_template = str(generate_empty_data_template(schema))
    data_template = generate_empty_data_template_tooluse(schema, triplets=True)
    identified_values = data['identified_values']

    system_prompt = f"""You are a database assistant that extracts data from each sentence of a given text, and populates data entries into a fixed relational database schema.

The user will give you the following inputs: a text with data of multiple users, the primary identifier (superkey) of that text, extracted triplets from that text in the following format:
<superkey, `subject`, `relation`, `object`>

You will follow this chain-of-thought reasoning to generate the final output:
- Generate output entries as per given instructions, relevant to the current paragraph.
- Validate each output entry by going through the triplets one by one, and ensure every unique data point from the triplets is captured into the correct table and fields.
- Follow the given output format strictly. Do not add any additional explanations or comments. Only output the data entries in given format. Do not provide code, do the task yourself.

Text:
{text}

Output format:
{data_template}

"""
    user_prompt = "Please perform the task as per the system instructions.\n"
    user_prompt += f"### Extracted <{superkey}, `subject`, `relation`, `object`> triplets from each paragraph:\n"
    for item in identified_values:
        for triplet in item['triplets']:
            if triplet['subject'] !="subject" or triplet['relation'] !="relation" or triplet['object'] !="object":
                user_prompt += f"<{item['superkey']}, {triplet['subject']}, {triplet['relation']}, {triplet['object']}>\n"
        user_prompt += "\n"
    user_prompt += "### Output:\n"
    return system_prompt, user_prompt


def get_value_population_prompt_TST_L(data):
    text = data['text']
    schema = data['schema']
    superkey = data['superkey']
    # data_template = str(generate_empty_data_template(schema))
    data_template = generate_empty_data_template_tooluse(schema, triplets=True)
    identified_values = data['identified_values']

    system_prompt = f"""You are a database assistant that extracts data from each sentence of a given text, and populates data entries into a fixed relational database schema.

The user will give you the following inputs: a text with data of multiple users, the primary identifier (superkey) of that text, extracted triplets from that text in the following format:
<superkey, `table_name`, `column_name`, `value`>

You will follow this chain-of-thought reasoning to generate the final output:
- Generate output entries as per given instructions, relevant to the current paragraph.
- Validate each output entry by going through the triplets one by one, and ensure every unique data point from the triplets is captured into the correct table and fields.
- Follow the given output format strictly. Do not add any additional explanations or comments. Only output the data entries in given format. Do not provide code, do the task yourself.

Text:
{text}

Output format:
{data_template}

"""
    user_prompt = "Please perform the task as per the system instructions.\n"
    user_prompt += f"### Extracted <{superkey}, `table_name`, `column_name`, `value`> triplets from each paragraph:\n"
    for item in identified_values:
        for triplet in item['triplets']:
            if triplet['value'] !="not mentioned" or triplet['value'] !="not provided":
                user_prompt += f"<{item['superkey']}, {triplet['table_name']}, {triplet['column_name']}, {triplet['value']}>\n"
        user_prompt += "\n"
    user_prompt += "### Output:\n"
    return system_prompt, user_prompt



def get_baseline_prompt(entry):
    text = entry['text']
    system_prompt = """You are a database expert. Your task is generating SQL 'CREATE TABLE' and 'INSERT INTO' statements from text."""
    user_prompt = f"""
### **Text:**
{text}
### **Output: (Write the create table and insert into statements together)
"""
    return system_prompt, user_prompt



def get_join_query(entry):
    sql_statements = entry['sql_statements']
    system_prompt = """You are a database expert. Your task is generating a sqlite join query from 'create table' and 'insert into' statements."""
    user_prompt = f"""
### **'CREATE TABLE' and 'INSERT INTO' statements:**
{sql_statements}
### **Output:**
"""
    return system_prompt, user_prompt

def old_get_value_population_prompt_with_llm_triplets(data):
    text = data['text']
    schema = data['schema']
    superkey = data['superkey']
    # data_template = str(generate_empty_data_template(schema))
    data_template = generate_empty_data_template_tooluse(schema)
    identified_values = data['identified_values']

    system_prompt = f"""You are a database assistant that extracts data from each sentence of a given text, and populates data entries into a fixed relational database schema.

Text:
{text}

Schema:
{schema}

Superkey:
{superkey}

The user will give you the following inputs: a paragraph of the user's data, superkey of that paragraph, extracted triplets from that paragraph in the following format:
<{superkey}, `table_name`, `column_name`, `value`>

Output format:
{data_template}
Output as many rows as necessary to populate the data. Replace the '#' with the actual values from the sentence.

You will follow this chain-of-thought reasoning to generate the final output:
- Generate output entries relevant to the current sentence. Fill in the superkey value as well.
- Validate each output entry by going through the triplets one by one, and ensure every unique data point from the triplets is captured into the correct table and fields.
- Follow the given output format strictly. Do not add any additional explanations or comments. Only output the data entries in given format. Do not provide code, do the task yourself.

Example of task:

Example Schema:
[
    {{
        "table_name": "countries",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": true}},
            {{"name": "iso_code", "type": "TEXT"}},
            {{"name": "country_name", "type": "TEXT"}},
            {{"name": "region", "type": "TEXT"}},
            {{"name": "human_development_index", "type": "REAL"}},
            {{"name": "year", "type": "INTEGER"}}
        ]
    }},
    {{
        "table_name": "education_inequality",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": true}},
            {{"name": "country_id", "type": "INTEGER", "foreign_key": "countries(id)"}},
            {{"name": "year", "type": "INTEGER"}},
            {{"name": "inequality", "type": "REAL"}}
        ]
    }}
]

Example Sentence:
"In 2010, the inequality in education was a staggering 39.4, a number that remained relatively consistent in the following years - 38.55827 in 2011, 2012, and 2013."

Example Triplets:
- education_inequality, inequality, 39.4
- education_inequality, year, 2010
- education_inequality, year, 2011
- education_inequality, year, 2012
- education_inequality, year, 2013
- education_inequality, inequality, 38.55827

Example Superkey: countries.id: 1

***Example Output by you:
extract 
"""

    user_prompts = []
    for item in identified_values:
        user_prompt = "Please perform the task as per the system instructions.\n"
        user_prompt += f"### Paragraph: {item['paragraph']}\n"
        user_prompt += f"### Extracted <{superkey}, `table_name`, `column_name`, `value`> triplets from paragraph:\n"
        for triplet in item['triplets']:
            if triplet['value'] !="not mentioned" or triplet['value'] !="not provided":
                user_prompt += f"<{item['superkey'], triplet['table_name']}, {triplet['column_name']}, {triplet['value']}>\n"
        user_prompt += "\n"
        user_prompt += "### Output:\n"
        user_prompts.append(user_prompt)
    return system_prompt, user_prompts


def get_t3_tuples_extraction_prompt(text):
    system_prompt = "You are going to summarize a table for this passage, but the first step is to extract useful information."
    user_prompt = f"Output them in (subject, attribute, value) or (subject,verb, object) format: {text}"
    return system_prompt, user_prompt

def get_triplets_extraction_with_llm_prompt(text, schema):
    system_prompt = (
        "You are an expert in Open Information Extraction and relational databases. "
        "Given a database schema and a natural language paragraph, your task is to extract all factual information from each sentence of the paragraph "
        "in the form of triplets, structured as a Python list of dictionaries. Each dictionary should have the following keys: "
        "'table_name', 'column_name', and 'value'. Ensure that the extracted triplets strictly follow the format: "
        "{'table_name': <table_name>, 'column_name': <column_name>, 'value': <value>} "
        "Only extract values that explicitly appear in the input sentence. The table_name and column_name must match the schema. "
        "Do not invent values or infer unstated facts. You don't need to generate triplets for values that are not mentioned. DO NOT generate code, do the task yourself. "
    )

    example_schema = '''
Schema:
[
    {
        "table_name": "flight",
        "columns": [
            {"name": "flight_id", "primary_key": True},
            {"name": "origin"},
            {"name": "destination"},
            {"name": "departure_time"},
            {"name": "arrival_time"}
        ]
    },
    {
        "table_name": "airline",
        "columns": [
            {"name": "airline_id", "primary_key": True},
            {"name": "name"},
            {"name": "country"}
        ]
    }
]
    '''

    example_text = "The Delta flight from New York to Los Angeles departs at 9 AM and arrives at noon."

    example_output = '''
Triplets:
[
    {"table_name": "flight", "column_name": "origin", "value": "New York"},
    {"table_name": "flight", "column_name": "destination", "value": "Los Angeles"},
    {"table_name": "flight", "column_name": "departure_time", "value": "9 AM"},
    {"table_name": "flight", "column_name": "arrival_time", "value": "noon"},
    {"table_name": "airline", "column_name": "name", "value": "Delta"}
]
    '''

    user_prompt = f"""
You will be given a database schema and a sentence. Extract all relevant triplets of the form:
{{"table_name": <table_name>, "column_name": <column_name>, "value": <value>}}

Your output must be a valid Python list of dictionaries. Do not include any explanations or notes—only return the list.

{example_schema}

Sentence: {example_text}

{example_output}

Now extract triplets for the following input:

Schema:
{schema}

Sentence: {text}
Triplets:
"""

    return system_prompt, user_prompt
