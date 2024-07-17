import streamlit as st
import pandas as pd
import json
from io import BytesIO

def parse_json_columns(df, json_columns):
    def try_loads(x):
        try:
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return None

    parsed_dfs = [df.drop(columns=json_columns)]
    for json_column in json_columns:
        parsed_data = df[json_column].apply(try_loads)
        json_df = pd.json_normalize(parsed_data)
        json_df.columns = [f"{json_column}.{col}" for col in json_df.columns]
        parsed_dfs.append(json_df)
    
    parsed_df = pd.concat(parsed_dfs, axis=1)
    return parsed_df

st.title("File to JSON Parser")

uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "xls", "csv", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type in ['xlsx', 'xls']:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        sheet_name = st.selectbox("Select sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type == 'txt':
        df = pd.read_csv(uploaded_file, delimiter='\t')

    st.write("Original Data:")
    st.dataframe(df)

    json_columns = st.multiselect("Select JSON columns", df.columns)

    if st.button("Parse JSON Columns"):
        try:
            parsed_df = parse_json_columns(df, json_columns)
            st.write("Parsed Data:")
            st.dataframe(parsed_df)

            # Option zum Herunterladen der aufbereiteten Daten
            csv = parsed_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download as CSV", data=csv, file_name='parsed_data.csv', mime='text/csv')

            # Option zum Herunterladen der aufbereiteten Daten als Excel-Datei
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                parsed_df.to_excel(writer, index=False, sheet_name='Sheet1')
                writer.save()
            output.seek(0)
            st.download_button(label="Download as Excel", data=output, file_name='parsed_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except Exception as e:
            st.error(f"An error occurred: {e}")

