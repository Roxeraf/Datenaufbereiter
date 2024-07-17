import streamlit as st
import pandas as pd
import json
from io import BytesIO

def try_loads(x):
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except (TypeError, json.JSONDecodeError):
        return x

def flatten_json(nested_json, prefix=''):
    flattened = {}
    for key, value in nested_json.items():
        if isinstance(value, dict):
            flattened.update(flatten_json(value, f"{prefix}{key}."))
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened

def parse_json_columns(df, json_columns):
    for json_column in json_columns:
        parsed_data = df[json_column].apply(try_loads)
        flattened_data = parsed_data.apply(lambda x: flatten_json(x) if isinstance(x, dict) else x)
        json_df = pd.DataFrame(flattened_data.tolist())
        json_df.columns = [f"{json_column}.{col}" for col in json_df.columns]
        df = df.drop(columns=[json_column]).join(json_df)
    return df

def preserve_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    return df

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
            parsed_df = preserve_dtypes(parsed_df)
            st.write("Parsed Data:")
            st.dataframe(parsed_df)

            # Option zum Herunterladen der aufbereiteten Daten als CSV-Datei
            csv = parsed_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download as CSV", data=csv, file_name='parsed_data.csv', mime='text/csv')

            # Option zum Herunterladen der aufbereiteten Daten als Excel-Datei
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                parsed_df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
                for col_num, value in enumerate(parsed_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                worksheet.autofit()
            output.seek(0)
            st.download_button(label="Download as Excel", data=output, file_name='parsed_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except ValueError as ve:
            st.error(f"Fehler beim Parsen der JSON-Daten: {ve}")
        except KeyError as ke:
            st.error(f"Spalte nicht gefunden: {ke}")
        except Exception as e:
            st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")