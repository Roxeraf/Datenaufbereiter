import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

st.set_page_config(page_title="Datenanalyse Dashboard", layout="wide")

@st.cache_data
def load_data(file):
    return pd.read_excel(file, engine='openpyxl')

def display_nan_info(df):
    nan_counts = df.isna().sum()
    nan_percentages = (nan_counts / len(df)) * 100
    nan_info = pd.DataFrame({
        'NaN Count': nan_counts,
        'NaN Percentage': nan_percentages
    })
    nan_info = nan_info[nan_info['NaN Count'] > 0].sort_values('NaN Count', ascending=False)
    return nan_info

st.title('Datenanalyse Dashboard für Temperatur, Luftfeuchtigkeit und Qualität')

uploaded_file = st.file_uploader("Laden Sie Ihre Excel-Datei hoch", type="xlsx")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Daten erfolgreich geladen. Form:", data.shape)

    # NaN-Informationen anzeigen
    st.subheader("NaN-Werte in den Daten")
    nan_info = display_nan_info(data)
    if not nan_info.empty:
        st.write(nan_info)
        
        # Visualisierung der NaN-Werte
        fig = px.bar(nan_info, x=nan_info.index, y='NaN Percentage', 
                     title='Prozentsatz der NaN-Werte pro Spalte')
        fig.update_layout(xaxis_title='Spalten', yaxis_title='Prozentsatz der NaN-Werte')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Keine NaN-Werte in den Daten gefunden.")

    # Spalten auswählen
    time_col = st.selectbox("Wählen Sie die Zeitspalte", data.columns)
    temp_col = st.selectbox("Wählen Sie die Temperaturspalte", data.columns)
    humidity_col = st.selectbox("Wählen Sie die Luftfeuchtigkeitsspalte", data.columns)
    quality_cols = st.multiselect("Wählen Sie die Qualitätsspalten", data.columns)

    if time_col and temp_col and humidity_col and quality_cols:
        # Daten vorbereiten
        data[time_col] = pd.to_datetime(data[time_col])
        numeric_cols = [temp_col, humidity_col] + quality_cols
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Dashboard erstellen
        st.header("Datenanalyse Dashboard")

        # Zeitreihen-Plot
        st.subheader("Zeitreihen-Analyse")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=data[time_col], y=data[temp_col], name="Temperatur"), secondary_y=False)
        fig.add_trace(go.Scatter(x=data[time_col], y=data[humidity_col], name="Luftfeuchtigkeit"), secondary_y=True)
        fig.update_layout(title_text="Temperatur und Luftfeuchtigkeit über Zeit")
        fig.update_xaxes(title_text="Zeit")
        fig.update_yaxes(title_text="Temperatur", secondary_y=False)
        fig.update_yaxes(title_text="Luftfeuchtigkeit", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Korrelationsmatrix
        st.subheader("Korrelationsanalyse")
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        fig.update_layout(title_text="Korrelationsmatrix")
        st.plotly_chart(fig, use_container_width=True)

        # Streudiagramme
        st.subheader("Streudiagramme")
        for quality_col in quality_cols:
            fig = make_subplots(rows=1, cols=2)
            fig.add_trace(go.Scatter(x=data[temp_col], y=data[quality_col], mode='markers', name="vs Temperatur"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data[humidity_col], y=data[quality_col], mode='markers', name="vs Luftfeuchtigkeit"), row=1, col=2)
            fig.update_layout(title_text=f"Qualität ({quality_col}) vs Temperatur und Luftfeuchtigkeit")
            fig.update_xaxes(title_text="Temperatur", row=1, col=1)
            fig.update_xaxes(title_text="Luftfeuchtigkeit", row=1, col=2)
            fig.update_yaxes(title_text=quality_col)
            st.plotly_chart(fig, use_container_width=True)

        # Statistische Zusammenfassung
        st.subheader("Statistische Zusammenfassung")
        st.write(data[numeric_cols].describe())

        # Hypothesentest
        st.subheader("Hypothesentest: Korrelation zwischen Variablen")
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 != col2:
                    correlation, p_value = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                    st.write(f"Korrelation zwischen {col1} und {col2}: {correlation:.2f} (p-Wert: {p_value:.4f})")

else:
    st.write("Bitte laden Sie eine Excel-Datei hoch, um zu beginnen.")