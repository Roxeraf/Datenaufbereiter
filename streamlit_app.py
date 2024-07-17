import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import openai

# Konfiguration für OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

def load_data(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type == 'csv':
        return pd.read_csv(file)
    elif file_type in ['xlsx', 'xls']:
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

def get_llm_preprocessing_steps(df):
    # ... [Funktion bleibt unverändert] ...

def apply_preprocessing(df, preprocessing_steps):
    # ... [Funktion bleibt unverändert] ...

def train_model(X, y):
    # ... [Funktion bleibt unverändert] ...

st.title('ML-Modell mit LLM-unterstützter Datenvorverarbeitung')

# Erstellen Sie drei optionale File Uploader
uploaded_files = []
for i in range(3):
    file = st.file_uploader(f"Laden Sie Datei {i+1} hoch (optional)", type=["csv", "xlsx", "xls"], key=f"file_{i}")
    if file is not None:
        uploaded_files.append(file)

if uploaded_files:
    try:
        # Laden und Zusammenführen der Daten
        dataframes = [load_data(file) for file in uploaded_files]
        data = pd.concat(dataframes, ignore_index=True)
        st.write(f"Daten aus {len(uploaded_files)} Datei(en) geladen und zusammengeführt. Form:", data.shape)

        if st.button('Datenvorverarbeitung starten'):
            preprocessing_steps = get_llm_preprocessing_steps(data)
            st.subheader("Vorgeschlagene Vorverarbeitungsschritte:")
            st.code(preprocessing_steps)

            if st.button('Vorverarbeitung anwenden'):
                data = apply_preprocessing(data, preprocessing_steps)
                st.write("Daten nach Vorverarbeitung. Form:", data.shape)

        target_variable = st.selectbox("Wählen Sie die Zielvariable", data.columns)
        feature_cols = st.multiselect("Wählen Sie die Eingabevariablen", 
                                      [col for col in data.columns if col != target_variable])

        if st.button('Modell trainieren'):
            if not feature_cols:
                st.error("Bitte wählen Sie mindestens eine Eingabevariable aus.")
            else:
                X = data[feature_cols]
                y = data[target_variable]

                model, scaler, score = train_model(X, y)

                st.subheader("Modellergebnisse")
                st.write(f"R²-Score: {score:.2f}")

                importance = model.feature_importances_
                fig = px.bar(x=importance, y=feature_cols, orientation='h', title='Feature Importance')
                st.plotly_chart(fig)

                # Optional: Vorhersagen
                if st.checkbox("Vorhersagen machen"):
                    input_data = {}
                    for feature in feature_cols:
                        input_data[feature] = st.number_input(f"Geben Sie einen Wert für {feature} ein")
                    
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(scaler.transform(input_df))
                    st.write(f"Vorhersage für {target_variable}: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Fehler beim Laden oder Verarbeiten der Dateien: {str(e)}")

else:
    st.write("Bitte laden Sie mindestens eine CSV- oder Excel-Datei hoch, um zu beginnen.")