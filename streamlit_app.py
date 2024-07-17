import streamlit as st
import pandas as pd
import json

def parse_json_column(df, json_column):
    parsed_data = df[json_column].apply(json.loads)
    json_df = pd.json_normalize(parsed_data)
    return pd.concat([df.drop(columns=[json_column]), json_df], axis=1)

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

    json_column = st.selectbox("Select JSON column", df.columns)

    if st.button("Parse JSON Column"):
        parsed_df = parse_json_column(df, json_column)
        st.write("Parsed Data:")
        st.dataframe(parsed_df)

        # Option zum Herunterladen der aufbereiteten Daten
        csv = parsed_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download as CSV", data=csv, file_name='parsed_data.csv', mime='text/csv')
