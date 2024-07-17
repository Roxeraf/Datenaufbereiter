import streamlit as st
import pandas as pd
import json

def parse_json_columns(df, json_columns):
    # Funktion zum sicheren Parsen von JSON-Strings in mehreren Spalten
    def try_loads(x):
        try:
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return None

    parsed_dfs = []
    
    for json_column in json_columns:
        parsed_data = df[json_column].apply(try_loads)
        json_df = pd.json_normalize(parsed_data.dropna())
        json_df.columns = [f"{json_column}.{col}" for col in json_df.columns]  # Prefix the columns with the original column name
        parsed_dfs.append(json_df)

    df = df.drop(columns=json_columns)  # Remove the original JSON columns
    parsed_df = pd.concat([df] + parsed_dfs, axis=1)  # Concatenate the original dataframe with all parsed JSON dataframes
    
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
            excel_buffer = pd.ExcelWriter("parsed_data.xlsx", engine='xlsxwriter')
            parsed_df.to_excel(excel_buffer, index=False)
            excel_buffer.save()
            st.download_button(label="Download as Excel", data=excel_buffer, file_name='parsed_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except Exception as e:
            st.error(f"An error occurred: {e}")
