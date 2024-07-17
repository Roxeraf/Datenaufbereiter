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

# ... [Rest der Funktionen bleiben unverändert] ...

st.title('ML-Modell mit LLM-unterstützter Datenvorverarbeitung')

uploaded_file = st.file_uploader("Laden Sie Ihre Datei hoch", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        st.write("Daten geladen. Form:", data.shape)

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
        st.error(f"Fehler beim Laden oder Verarbeiten der Datei: {str(e)}")

else:
    st.write("Bitte laden Sie eine CSV- oder Excel-Datei hoch, um zu beginnen.")