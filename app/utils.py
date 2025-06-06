import os
import sqlite3
import logging
import pandas as pd
from llama_cpp import Llama

BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "data", "example_data.db")
MODEL_PATH_LLM = os.path.join(BASE_DIR, "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MODEL_PATH_SLM = os.path.join(BASE_DIR, "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema_description.txt")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

_llm_instance = None
def load_llm_main():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = Llama(model_path=MODEL_PATH_LLM, n_ctx=1024, n_threads=16, n_batch=32, n_gpu_layers=28)
    return _llm_instance

_small_llm_instance = None
def load_llm_small():
    global _small_llm_instance
    if _small_llm_instance is None:
        _small_llm_instance = Llama(model_path=MODEL_PATH_SLM, n_ctx=512, n_threads=4, n_batch=16, n_gpu_layers=16)
    return _small_llm_instance

def generate_sql_from_prompt(prompt, schema_path=SCHEMA_PATH):
    with open(schema_path, "r") as f:
        schema_text = f.read()
    llm = load_llm_main()
    full_prompt = (
        f"You are a helpful assistant that writes SQL queries.\n\n"
        f"Given the following database schema:\n\n"
        f"{schema_text}\n\n"
        f"Write a valid SQL query to answer the following question:\n"
        f"{prompt}\n\n"
        f"Make sure the SQL is complete and executable.\nSQL:"
    )
    output = llm(prompt=full_prompt, max_tokens=80, stop=["#", ";"])
    sql = output["choices"][0]["text"].strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        lines = sql.splitlines()
        if lines and lines[0].lower().startswith("sql"):
            lines = lines[1:]
        sql = "\n".join(lines).strip()
    return sql

def run_sql_query(sql):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        return col_names, rows
    except sqlite3.Error as e:
        logging.error("SQLite error: %s", e)
        return None, f"SQLite error: {e}"

def explain_result(prompt, sql, df):
    llm = load_llm_main()
    result_text = df.head(5).to_markdown(index=False)
    summary_prompt = (
        f"You are a data analyst assistant.\n\n"
        f"A user asked: '{prompt}'\n\n"
        f"The system ran this SQL query:\n{sql}\n\n"
        f"The top results are:\n{result_text}\n\n"
        f"Summarize the result in 1–3 sentences."
    )
    output = llm(prompt=summary_prompt, max_tokens=100, stop=["#", ";"])
    return output["choices"][0]["text"].strip()
