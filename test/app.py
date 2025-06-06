import os
import sqlite3
import logging
import pandas as pd
import streamlit as st
from llama_cpp import Llama

# -------------------
# Path configuration
# -------------------
BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "../data/example_data.db")
MODEL_PATH_LLM = os.path.join(BASE_DIR, "../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MODEL_PATH_SLM = os.path.join(BASE_DIR, "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
SCHEMA_PATH = os.path.join(BASE_DIR, "../schema_description.txt")

# -------------------
# Logging
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# -------------------
# Load model (singleton)
# -------------------
_llm_instance = None
def load_llm_main():
    global _llm_instance
    if _llm_instance is None:
        logging.info("Loading LLM model...")
        _llm_instance = Llama(
            model_path=MODEL_PATH_LLM,
            n_ctx=1024,
            n_threads=16,
            n_batch=32,
            n_gpu_layers=28
        )
        logging.info("LLM model loaded.")
    return _llm_instance


_small_llm_instance = None
def load_llm_small():
    global _small_llm_instance
    if _small_llm_instance is None:
        logging.info("Loading SMALL LLM...")
        _small_llm_instance = Llama(
            model_path=MODEL_PATH_SLM,
            n_ctx=512,
            n_threads=4,
            n_batch=16,
            n_gpu_layers=16  # lighter use
        )
        logging.info("Small LLM loaded.")
    return _small_llm_instance

# -------------------
# Generate SQL from prompt + schema
# -------------------
def generate_sql_from_prompt(prompt: str, schema_path: str = SCHEMA_PATH) -> str:
    with open(schema_path, "r") as f:
        schema_text = f.read()

    llm = load_llm_main()

    full_prompt = (
        f"You are a helpful assistant that writes SQL queries.\n\n"
        f"Given the following database schema:\n\n"
        f"{schema_text}\n\n"
        f"Write a valid SQL query to answer the following question:\n"
        f"{prompt}\n\n"
        f"Make sure the SQL is complete and executable."
        f"SQL:"
    )


    output = llm(prompt=full_prompt, max_tokens=80, stop=["#", ";"])
    sql = output["choices"][0]["text"].strip()

    if sql.startswith("```"):
        sql = sql.strip("`")
        lines = sql.splitlines()
        if lines[0].lower().startswith("sql"):
            lines = lines[1:]
        sql = "\n".join(lines).strip()

    return sql

# -------------------
# Run SQL query
# -------------------
def run_sql_query(sql: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description]
        conn.close()
        return col_names, rows
    except sqlite3.Error as e:
        logging.error("SQLite error: %s", e)
        return None, f"SQLite error: {e}"
    

# -------------------
# Generate summary from SQL + result
# -------------------


def explain_result(prompt: str, sql: str, df: pd.DataFrame) -> str:
    llm = load_llm_main()
    result_text = df.head(5).to_markdown(index=False)

    summary_prompt = (
        f"You are a data analyst assistant.\n\n"
        f"A user asked: \"{prompt}\"\n\n"
        f"The system ran this SQL query:\n{sql}\n\n"
        f"Here are the first rows of the result table:\n\n"
        f"{result_text}\n\n"
        f"Summarize this result in 1–3 sentences, explaining what the data shows."
    )

    output = llm(prompt=summary_prompt, max_tokens=100, stop=["#", ";"])
    return output["choices"][0]["text"].strip()
 

# -------------------
# Streamlit UI
# -------------------
st.title("🧠 Natural Language to SQL")

# Initialize session state for history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

prompt = st.text_input("Enter your question about the sales database:")

if st.button("Generate and Execute SQL"):
    if not prompt.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Generating SQL..."):
            sql = generate_sql_from_prompt(prompt)
        
        # Save to history
        st.session_state.query_history.append({"prompt": prompt, "sql": sql})

        with st.expander("🧾 View Generated SQL"):
            st.code(sql, language="sql")

        with st.spinner("Executing SQL..."):
            columns, results = run_sql_query(sql)
            if isinstance(results, list):
                df = pd.DataFrame(results, columns=columns)
                st.subheader("📊 Query Results")
                st.dataframe(df)

                with st.spinner("Generating explanation..."):
                    explanation = explain_result(prompt, sql, df)
                    st.subheader("🗣️ Natural Language Summary")
                    st.markdown(explanation)

            else:
                st.error(results)

# -------------------
# Show history
# -------------------
if st.session_state.query_history:
    with st.expander("📚 Query History"):
        for i, item in enumerate(reversed(st.session_state.query_history), 1):
            st.markdown(f"**Prompt {len(st.session_state.query_history)-i+1}:** {item['prompt']}")
            st.code(item['sql'], language="sql")

