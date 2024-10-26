import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import sqlite3
import os
import re
from openai import OpenAI

# Helper functions to generate data
def generate_varchar(options, num_rows):
    return [random.choice(options) for _ in range(num_rows)]

def generate_date(start_date, end_date, num_rows):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return [start + timedelta(days=random.randint(0, (end - start).days)) for _ in range(num_rows)]

def generate_unique_ids(num_rows):
    return random.sample(range(1, num_rows * 10), num_rows)

def generate_numbers(start, end, num_rows):
    return [random.uniform(start, end) for _ in range(num_rows)]

# Streamlit app
def main():
    st.title("SQL Data Simulator")
    
    # Define table schemas
    num_tables = st.sidebar.number_input("Number of tables", min_value=1, max_value=10, value=1)
    tables = {}
    
    for i in range(num_tables):
        st.header(f"Table {i+1} Schema")
        table_name = st.text_input(f"Table {i+1} Name", value=f"Table_{i+1}")
        num_columns = st.number_input(f"Number of columns in {table_name}", min_value=1, max_value=20, value=3)
        
        columns = []
        for j in range(num_columns):
            col_name = st.text_input(f"Column {j+1} Name for {table_name}", value=f"Column_{j+1}")
            col_type = st.selectbox(f"Data Type for {col_name}", ["VARCHAR", "DATE", "ID", "NUMBER", "RECYCLED"], key=f"{table_name}_{col_name}")
            
            col_details = {}
            col_details["name"] = col_name
            col_details["type"] = col_type
            
            if col_type == "VARCHAR":
                options = st.text_area(f"Options for {col_name} (comma-separated)").split(",")
                col_details["options"] = [opt.strip() for opt in options]
            elif col_type == "DATE":
                start_date = st.date_input(f"Start date for {col_name}", key=f"start_{table_name}_{col_name}")
                end_date = st.date_input(f"End date for {col_name}", key=f"end_{table_name}_{col_name}")
                col_details["start_date"] = start_date
                col_details["end_date"] = end_date
            elif col_type == "ID":
                col_details["unique"] = True
            elif col_type == "NUMBER":
                start = st.number_input(f"Start range for {col_name}", value=0, key=f"start_{table_name}_{col_name}")
                end = st.number_input(f"End range for {col_name}", value=100, key=f"end_{table_name}_{col_name}")
                col_details["range"] = (start, end)
            elif col_type == "RECYCLED":
                ref_table = st.selectbox(f"Reference table for {col_name}", list(tables.keys()), key=f"ref_{table_name}_{col_name}")
                ref_column = st.selectbox(f"Reference column in {ref_table}", [col["name"] for col in tables[ref_table]], key=f"ref_col_{table_name}_{col_name}")
                col_details["ref_table"] = ref_table
                col_details["ref_column"] = ref_column
            
            columns.append(col_details)
        
        tables[table_name] = columns
    
    # Generate data
    st.header("Generated Data")
    if 'data_frames' not in st.session_state:
        st.session_state.data_frames = {}
    
    if len(st.session_state.data_frames) == 0:
        for table_name, columns in tables.items():
            num_rows = st.number_input(f"Number of rows for {table_name}", min_value=1, max_value=10000, value=10, key=f"rows_{table_name}")
            data = {}
            
            for col in columns:
                if col["type"] == "VARCHAR":
                    data[col["name"]] = generate_varchar(col["options"], num_rows)
                elif col["type"] == "DATE":
                    data[col["name"]] = generate_date(str(col["start_date"]), str(col["end_date"]), num_rows)
                elif col["type"] == "ID":
                    data[col["name"]] = generate_unique_ids(num_rows)
                elif col["type"] == "NUMBER":
                    start, end = col["range"]
                    data[col["name"]] = generate_numbers(start, end, num_rows)
                elif col["type"] == "RECYCLED":
                    ref_table = col["ref_table"]
                    ref_column = col["ref_column"]
                    data[col["name"]] = st.session_state.data_frames[ref_table][ref_column].values
            
            df = pd.DataFrame(data)
            st.session_state.data_frames[table_name] = df
            st.write(f"Data for {table_name}")
            st.write(df)
    
    # Write to SQLite database
    conn = sqlite3.connect(':memory:')
    for table_name, df in st.session_state.data_frames.items():
        df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    st.header("Database Schemas")
    for table_name in tables.keys():
        query = f"PRAGMA table_info({table_name})"
        schema = pd.read_sql(query, conn)
        st.write(f"Schema for {table_name}")
        st.write(schema)
    
    # Pivot table and chart visualization
    st.header("Pivot Table and Chart Visualization")
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None
    
    selected_table = st.selectbox("Select a table to visualise", list(st.session_state.data_frames.keys()))
    if selected_table:
        st.session_state.selected_table = selected_table
        df = st.session_state.data_frames[selected_table]
        x_axis = st.selectbox("Select X-axis (horizontal axis)", df.columns)
        y_axis = st.selectbox("Select Y-axis (vertical axis)", df.columns)
        category = st.selectbox("Select Category Column (Optional)", [None] + list(df.columns))
        filter_column = st.selectbox("Select Filter Column (Optional)", [None] + list(df.columns))
        filter_value = None
        if filter_column is not None:
            filter_value = st.text_input(f"Enter value to filter by {filter_column}")
        
        if filter_value:
            df = df[df[filter_column] == filter_value]
        
        if category is None:
            pivot = df.groupby(x_axis)[y_axis].count().reset_index()
        else:
            pivot = df.groupby([x_axis, category])[y_axis].count().reset_index()
        
        if pd.api.types.is_numeric_dtype(df[x_axis]) or pd.api.types.is_datetime64_any_dtype(df[x_axis]):
            fig = px.line(pivot, x=x_axis, y=y_axis, color=category)
        else:
            fig = px.bar(pivot, x=x_axis, y=y_axis, color=category)
        
        st.write("Pivot Table")
        st.write(pivot)
        st.plotly_chart(fig)

    # Chatbot for natural language to SQL
    st.header("Chatbot: Natural Language to SQL Query")
    if 'query_result' not in st.session_state:
        st.session_state.query_result = None

    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    user_query = st.text_area("Enter your enquiry about the data")

    if st.button("Generate SQL Query"):
        date_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2}|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b)')
        if not date_pattern.search(user_query):
            st.warning("You have not specified a date period in your request. Please add a date range to proceed.")
        else:
            if openai_api_key and user_query:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                table_schemas = []
                for table_name in tables.keys():
                    query = f"PRAGMA table_info({table_name})"
                    schema = pd.read_sql(query, conn)
                    columns_info = ", ".join([f"{row.name} ({row.type})" + (f" with options {col['options']}" if 'options' in col else "") for row, col in zip(schema.itertuples(index=False), tables[table_name])])
                    table_schemas.append(f"{table_name}: {columns_info}")
                
                schema_info = "\n".join(table_schemas)
                prompt = f"""
You are given the following table schemas:
{schema_info}

Translate the following natural language request into an SQL query, using standard SQLite syntax and avoiding non-standard functions. Ensure to only use the values present in the column options where applicable.
:
{user_query}
"""
            
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model="gpt-4o-mini"
                    )
                    sql_query = chat_completion.choices[0].message.content.split('```sql')[-1].split('```')[0].strip()
                    st.session_state.query_result = sql_query
                    st.text_area("Generated SQL Query", value=st.session_state.query_result, height=100)
                    summary_prompt = f"""
The following SQL query has been generated:
```sql
{sql_query}
```
Provide a very brief but specific natural language summary of the data that have been pulled:
"""
                    try:
                        summary_completion = client.chat.completions.create(
                            messages=[
                                {"role": "user", "content": summary_prompt}
                            ],
                            model="gpt-4o-mini"
                        )
                        summary = summary_completion.choices[0].message.content.strip()
                        st.text_area("Query Summary", value=summary, height=100)
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                    try:
                        result = pd.read_sql(st.session_state.query_result, conn)
                        st.write("Query Result")
                        st.write(result)
                        st.session_state.query_result = result
                    except Exception as e:
                        st.error(f"Error executing query: {e}")
                except Exception as e:
                    st.error(f"Error generating SQL query: {e}")
            else:
                st.error("Please provide both an OpenAI API key and a query.")

    if st.session_state.query_result is not None:
        clarification_query = st.text_area("Do you have any clarification questions about the results?")
        if st.button("Ask Clarification Question"):
            if clarification_query:
                clarification_prompt = f"""
        The user has been provided with the following SQL query and its result:
        ```sql
        {st.session_state.query_result}
        ```
        Query Result: {st.session_state.query_result.head().to_string(index=False)}
        The user now has the following follow-up question:
        {clarification_query}
        Provide an appropriate response.
        """
                try:
                    clarification_completion = client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": clarification_prompt}
                        ],
                        model="gpt-4o-mini"
                    )
                    clarification_response = clarification_completion.choices[0].message.content.strip()
                    st.text_area("Clarification Response", value=clarification_response, height=150)
                except Exception as e:
                    st.error(f"Error generating clarification response: {e}")

if __name__ == "__main__":
    main()

