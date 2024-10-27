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

def preload_nhs_tables(num_rows):
    patient_ids = generate_unique_ids(num_rows)
    
    patients_data = {
        "PatientID": patient_ids,
        "Name": generate_varchar(["Alice", "Bob", "Charlie", "David", "Ella"], num_rows),
        "Gender": generate_varchar(["Male", "Female"], num_rows),
        "DateOfBirth": generate_date("1940-01-01", "2020-12-31", num_rows),
    }
    patients_df = pd.DataFrame(patients_data)

    appointments_data = {
        "AppointmentID": generate_unique_ids(num_rows),
        "PatientID": random.choices(patient_ids, k=num_rows),
        "Date": generate_date("2023-01-01", "2023-12-31", num_rows),
        "Department": generate_varchar(["Cardiology", "Oncology", "Paediatrics", "General Medicine"], num_rows),
    }
    appointments_df = pd.DataFrame(appointments_data)

    admissions_data = {
        "AdmissionID": generate_unique_ids(num_rows),
        "PatientID": random.choices(patient_ids, k=num_rows),
        "AdmissionDate": generate_date("2023-01-01", "2023-12-31", num_rows),
        "DischargeDate": generate_date("2023-01-02", "2024-01-01", num_rows),
        "Ward": generate_varchar(["A1", "B2", "C3", "D4"], num_rows),
    }
    admissions_df = pd.DataFrame(admissions_data)

    prescriptions_data = {
        "PrescriptionID": generate_unique_ids(num_rows),
        "PatientID": random.choices(patient_ids, k=num_rows),
        "Medication": generate_varchar(["Paracetamol", "Ibuprofen", "Aspirin", "Metformin"], num_rows),
        "Dose": generate_varchar(["500mg", "200mg", "100mg", "1g"], num_rows),
        "DatePrescribed": generate_date("2023-01-01", "2023-12-31", num_rows),
    }
    prescriptions_df = pd.DataFrame(prescriptions_data)

    surgeries_data = {
        "SurgeryID": generate_unique_ids(num_rows),
        "PatientID": random.choices(patient_ids, k=num_rows),
        "SurgeryType": generate_varchar(["Appendectomy", "Hip Replacement", "Cataract Surgery", "Gallbladder Removal"], num_rows),
        "SurgeryDate": generate_date("2023-01-01", "2023-12-31", num_rows),
        "Surgeon": generate_varchar(["Dr. Smith", "Dr. Jones", "Dr. Brown", "Dr. Taylor"], num_rows),
    }
    surgeries_df = pd.DataFrame(surgeries_data)

    st.session_state.data_frames = {
        "Patients": patients_df,
        "Appointments": appointments_df,
        "Admissions": admissions_df,
        "Prescriptions": prescriptions_df,
        "Surgeries": surgeries_df
    }

def main():
    st.title("Text to Data Bot")

    password = st.text_input("Enter Password", type="password")
    
    if "OPENAI_API_KEY" not in st.secrets or "PASSWORD" not in st.secrets:
        st.error("API key or password not found.")

    if password == "":
        st.warning("Please enter password")
    elif password != st.secrets["PASSWORD"]:
        st.error("Incorrect password. Please try again.")
    else:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        num_rows = st.number_input("Select number of rows to generate", min_value=0, max_value=100000, value=1000)

        if 'data_frames' not in st.session_state or len(st.session_state.data_frames['Patients']) != num_rows:
            preload_nhs_tables(num_rows)
        
        st.header("Tables")
        table_definitions = {
            "Patients": {
                "PatientID": "Unique identifier for each patient",
                "Name": "Name of the patient",
                "Gender": "Gender of the patient",
                "DateOfBirth": "Date of birth of the patient"
            },
            "Appointments": {
                "AppointmentID": "Unique identifier for each appointment",
                "PatientID": "Identifier linking to the patient",
                "Date": "Date of the appointment",
                "Department": "Department for the appointment"
            },
            "Admissions": {
                "AdmissionID": "Unique identifier for each admission",
                "PatientID": "Identifier linking to the patient",
                "AdmissionDate": "Date of admission",
                "DischargeDate": "Date of discharge",
                "Ward": "Ward to which the patient was admitted"
            },
            "Prescriptions": {
                "PrescriptionID": "Unique identifier for each prescription",
                "PatientID": "Identifier linking to the patient",
                "Medication": "Name of the medication prescribed",
                "Dose": "Dosage of the medication",
                "DatePrescribed": "Date the medication was prescribed"
            },
            "Surgeries": {
                "SurgeryID": "Unique identifier for each surgery",
                "PatientID": "Identifier linking to the patient",
                "SurgeryType": "Type of surgery performed",
                "SurgeryDate": "Date of the surgery",
                "Surgeon": "Name of the surgeon who performed the surgery"
            }
        }
        
        conn = sqlite3.connect(':memory:')
        for table_name, df in st.session_state.data_frames.items():
            df.to_sql(table_name, conn, index=False, if_exists='replace')

        if num_rows > 0:
            st.write("Tables generated are:")
            for table in table_definitions:
                st.write(f"{chr(8226)} {table}")
            
            selected_table = st.selectbox("Select a table to see more detail", list(st.session_state.data_frames.keys()))
            if selected_table:    
                
                st.subheader(f"{selected_table} Table Schema and Data")
                for column, definition in table_definitions[selected_table].items():
                    st.write(f"**{column}**: {definition}")
                st.write(f"Data for {selected_table}")
                st.write(df.head(5))
                query = f"PRAGMA table_info({selected_table})"
                schema = pd.read_sql(query, conn)
                st.write(f"Schema for {selected_table}")
                st.write(schema)
                
                st.subheader("Display the Generated Data")
                df = st.session_state.data_frames[selected_table]
                count_column = st.selectbox("Select a column to count", df.columns, key="count_column")
                filter_column = st.selectbox("Add a column to filter if needed", [None] + list(df.columns), key="filter_column")
                filter_value = None
                if filter_column is not None:
                    filter_value = st.text_input(f"Enter value to filter by {filter_column}") 
                if filter_value:
                    df = df[df[filter_column] == filter_value]
                pivot = df[count_column].value_counts().reset_index()
                pivot.columns = [count_column, 'Count']                

                fig = px.bar(pivot, x=count_column, y='Count', title=f"Pivot Table for {selected_table}")
                st.write("Pivot Table Data")
                st.write(pivot)
                st.plotly_chart(fig)

                show_suggested_prompts = st.checkbox("Show Suggested Prompts for Querying the Data")
                if show_suggested_prompts:
                    difficulty = st.selectbox("Select difficulty level", ["Simple", "Tricky", "Challenging", "Impossible"])
                    suggested_prompts = {
                        "Simple": [
                            "Show all appointments for a specific patient.",
                            "List all patients who had a surgery in 2023.",
                            "Get the count of admissions per ward.",
                            "Find patients who have been prescribed a specific medication."
                        ],
                        "Tricky": [
                            "Show all patients with both appointments and admissions in 2023.",
                            "List patients who have had more than one surgery.",
                            "Get the number of prescriptions per patient who has been admitted to ward A1."
                        ],
                        "Challenging": [
                            "Show all patients who have had multiple surgeries, multiple admissions, and at least three different prescriptions.",
                            "List patients with two or more surgeries, who had appointments in multiple departments and were prescribed Ibuprofen in the second half of 2023."
                        ],
                        "Impossible": [
                            "Identify patients whose total length of stay across all admissions is more than twice the average length of stay for their ward",
                            "Calculate the 3 month moving average of the number of admissions per month and average length of stay for each ward."
                        ]
                    }
                    st.subheader(f"{difficulty} Queries")
                    for prompt in suggested_prompts[difficulty]:
                        st.write(f"- {prompt}")

                st.header("Chatbot: Natural Language to SQL Query")
                if 'query_result' not in st.session_state:
                    st.session_state.query_result = None

                user_query = st.text_area("Enter your data request:")

                if st.button("Generate SQL Query"):
                    date_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2}|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s\d{4}|\b\d{4}\b)', re.IGNORECASE)
                    if not date_pattern.search(user_query):
                        st.warning("It looks like you haven't specified a date period in your request!")
                    if openai_api_key and user_query:
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                        
                        table_schemas = []
                        for table_name in st.session_state.data_frames.keys():
                            query = f"PRAGMA table_info({table_name})"
                            schema = pd.read_sql(query, conn)
                            columns_info = ", ".join([f"{row.name} ({row.type})" for row in schema.itertuples(index=False)])
                            table_schemas.append(f"{table_name}: {columns_info}")
                        
                        schema_info = "\n".join(table_schemas)
                        prompt = f"""
            You are given the following table schemas:
            {schema_info}

            Translate the following natural language request into an SQL query, using standard SQLite syntax and avoiding non-standard functions.
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
                            
                            try:
                                result = pd.read_sql(st.session_state.query_result, conn)
                                st.write("Query Result")
                                st.write(result)
                                st.session_state.query_result = result
                                if 'clarification_query_checkbox' not in st.session_state:
                                    st.session_state.clarification_query_checkbox = False
                                st.session_state.clarification_query_checkbox = st.checkbox("Ask a Clarification Query About the Data", value=st.session_state.clarification_query_checkbox)
                                if st.session_state.clarification_query_checkbox:
                                    st.text_area("Enter your clarification:")

                            except Exception as e:
                                st.error(f"Error executing query: {e}")
                        except Exception as e:
                            st.error(f"Error generating SQL query: {e}")
                    else:
                        st.error("Either OpenAI API key or data query is missing.")

        else:
            st.write("Enter a number of rows to generate some data!")
            st.write("Generated tables and data will be shown below")
            st.write("Then you can query the data with natural language!")

if __name__ == "__main__":
    main()
