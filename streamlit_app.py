import streamlit as st
import pandas as pd
import io

def parse_data(data):
    # Teile die Daten in Zeilen
    lines = data.strip().split('\n')
    
    # Die erste Zeile enthält die Spaltenüberschriften
    headers = lines[0].split(',')
    
    # Parse die restlichen Zeilen
    rows = []
    for line in lines[1:]:
        # Teile jede Zeile an den Leerzeichen, aber behalte das Datum/Zeit zusammen
        parts = line.split(' ', 2)
        date_time = f"{parts[0]} {parts[1]}"
        values = parts[2].split(',')
        row = [date_time] + values
        rows.append(row)
    
    # Erstelle ein DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df

st.title("File to Excel Parser")

uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv"])

if uploaded_file:
    # Lese den Inhalt der Datei
    content = uploaded_file.getvalue().decode('utf-8')
    
    try:
        # Parse die Daten
        df = parse_data(content)
        
        st.write("Parsed Data:")
        st.dataframe(df)

        # Option zum Herunterladen der aufbereiteten Daten als Excel-Datei
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        
        st.download_button(
            label="Download as Excel",
            data=output,
            file_name='parsed_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")