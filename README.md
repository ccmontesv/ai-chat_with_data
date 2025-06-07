# LLM-RAG Chatbot with Flask

This application enables users to ask **natural language questions** about sales data and receive answers in **SQL and natural language** using **local LLMs** (like Mistral and TinyLlama). It combines **retrieval-augmented generation (RAG)** with an interactive **Flask web interface** for a lightweight and fully local experience.

---

## How It Works

This is a **Natural Language to SQL + Explanation Engine**, and it runs entirely on your machine:

### 1. User Prompt (Input)
- The user types a question like:  
  *"What were the total sales for product Alpha in 2023?"*

### 2. LLM-based SQL Generation (Main LLM: Mistral)
- The app reads a structured schema description (`schema_description.txt`)
- The prompt and schema are sent to a **local Mistral model** using `llama-cpp-python`
- The model generates a valid **SQL query** to answer the question

### 3. SQL Query Execution (SQLite)
- The generated SQL query is executed against a local **SQLite database**
- The results are fetched and presented in tabular format

### 4. Natural Language Explanation (Tiny Model)
- A second (lighter) LLM like **TinyLlama** summarizes the result into a human-readable explanation  
  Example:  
  *"In 2023, product Alpha achieved a total sales revenue of $42,000."*

### 5. Flask UI
- Everything is wrapped in a friendly **Flask web interface**
- Users can enter prompts, view the generated SQL, see results, and read summaries

---

## Requirements

### System
- **Python 3.10+**
- **RAM**: Minimum 8–12 GB (for Mistral to run locally)
- **CPU**: Multithreaded CPU highly recommended
- **Optional**: GPU with VRAM (for faster inference using `n_gpu_layers`)
- **OS**: Windows, Linux or macOS

### Python Libraries

You must install all required libraries via:

```bash
pip install -r requirements.txt
