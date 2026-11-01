import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Text-to-SQL Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Text-to-SQL Analyst Agent")
st.caption("Ask questions about your Chinook music-store database in plain English.")

with st.sidebar:
    st.header("About")
    st.write("Powered by FastAPI, ChromaDB, SQLite, and an LLM.")
    st.markdown("---")
    st.write("**Backend API**")
    st.code(API_URL)

    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=5,
            )
            response.raise_for_status()
            st.success("Backend is online.")
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

st.subheader("Ask your database")

examples = [
    "Which 10 artists have the most albums?",
    "What are the 5 most expensive tracks?",
    "Which customers have spent the most money?",
    "How many tracks belong to each genre?",
]

selected_example = st.selectbox(
    "Try an example question",
    ["Choose an example..."] + examples,
)

with st.form("query_form"):
    question = st.text_area(
        "Your question",
        value="" if selected_example == "Choose an example..." else selected_example,
        placeholder="For example: Which artists have the most albums?",
        height=100,
        max_chars=500,
    )
    submitted = st.form_submit_button(
        "🚀 Analyze",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Generating SQL and analyzing your data..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question.strip()},
                    timeout=120,
                )

            if not response.ok:
                try:
                    detail = response.json().get("detail", "Request failed.")
                except ValueError:
                    detail = response.text or "Request failed."

                st.error(f"API error ({response.status_code}): {detail}")
                st.stop()

            result = response.json()

        except requests.Timeout:
            st.error("The request timed out. Check the FastAPI terminal.")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")
            st.stop()

        st.session_state["last_result"] = result
        st.session_state["last_question"] = question.strip()

result = st.session_state.get("last_result")

if result:
    st.markdown("---")
    st.subheader("Analysis results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows returned", result.get("row_count", len(result.get("rows", []))))
    col2.metric("Attempts", result.get("attempts", "—"))

    latency = result.get("latency_ms")
    if latency is not None:
        col3.metric("Latency", f"{latency:.0f} ms")
    else:
        col3.metric("Latency", "Not provided")

    summary = result.get("summary")
    if summary:
        st.markdown("### 💡 AI Summary")
        st.info(summary)

    st.markdown("### Generated SQL")
    sql = result.get("sql", "")

    if sql:
        st.code(sql, language="sql")
        st.download_button(
            "Download SQL",
            data=sql,
            file_name="generated_query.sql",
            mime="text/plain",
        )

    rows = result.get("rows", [])
    columns = result.get("columns", [])

    st.markdown("### Query results")

    if rows:
        df = pd.DataFrame(rows, columns=columns or None)

        chart_type = str(result.get("chart_type", result.get("chart", "table"))).lower()

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = [column for column in df.columns if column not in numeric_columns]

        if chart_type in {"bar", "line"} and numeric_columns and categorical_columns:
            x_column = categorical_columns[0]
            y_column = numeric_columns[0]

            if chart_type == "bar":
                fig = px.bar(
                    df,
                    x=x_column,
                    y=y_column,
                    title=f"{y_column} by {x_column}",
                )
                if len(df) > 5:
                    fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "line":
                fig = px.line(
                    df,
                    x=x_column,
                    y=y_column,
                    markers=True,
                    title=f"{y_column} by {x_column}",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_data,
            file_name="query_results.csv",
            mime="text/csv",
        )

        with st.expander("Inspect response JSON"):
            st.json(result)
    else:
        st.info("The query returned no rows.")
        st.json(result)
else:
    st.info("Enter a question above to begin your analysis.")
