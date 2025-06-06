from flask import Blueprint, render_template, request
from app.utils import generate_sql_from_prompt, run_sql_query, explain_result
import pandas as pd

main = Blueprint("main", __name__)
query_history = []

@main.route("/", methods=["GET", "POST"])
def index():
    sql = ""
    explanation = ""
    results = []
    columns = []
    prompt = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if prompt:
            sql = generate_sql_from_prompt(prompt)
            query_history.append({"prompt": prompt, "sql": sql})
            columns, results = run_sql_query(sql)
            if isinstance(results, list) and results:
                df = pd.DataFrame(results, columns=columns)
                explanation = explain_result(prompt, sql, df)

    return render_template("index.html",
                           prompt=prompt,
                           sql=sql,
                           results=results,
                           columns=columns,
                           explanation=explanation,
                           history=reversed(query_history))


