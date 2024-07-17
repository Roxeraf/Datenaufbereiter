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
    return pd.read_csv(file)  # Vereinfachte Datenladung

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def get_llm_analysis(data_description, model_results):
    prompt = f"""
    Analysiere die folgenden Daten und Modellergebnisse:
    
    Datenbeschreibung:
    {data_description}
    
    Modellergebnisse:
    {model_results}
    
    Bitte gib eine detaillierte Analyse und Empfehlungen basierend auf diesen Informationen.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Fehler bei der LLM-Analyse: {str(e)}")
        return "LLM-Analyse konnte nicht durchgeführt werden."

st.title('ML-Modell mit LLM-Analyse')

uploaded_file = st.file_uploader("Laden Sie Ihre CSV-Datei hoch", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Daten geladen. Form:", data.shape)

    target_variable = st.selectbox("Wählen Sie die Zielvariable", data.columns)
    feature_cols = st.multiselect("Wählen Sie die Eingabevariablen", 
                                  [col for col in data.columns if col != target_variable])

    if st.button('Analyse starten'):
        X = data[feature_cols]
        y = data[target_variable]

        model, scaler = train_model(X, y)
        importance = model.feature_importances_

        # Daten für LLM vorbereiten
        data_description = f"Zielvariable: {target_variable}\nEingabevariablen: {', '.join(feature_cols)}"
        model_results = f"R²-Score: {model.score(scaler.transform(X), y):.2f}\n"
        model_results += "Top 5 wichtigste Features:\n"
        for idx in importance.argsort()[-5:][::-1]:
            model_results += f"{feature_cols[idx]}: {importance[idx]:.4f}\n"

        # LLM-Analyse
        llm_analysis = get_llm_analysis(data_description, model_results)

        # Dashboard
        st.subheader("Modellanalyse")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Feature Importance")
            fig = px.bar(x=importance, y=feature_cols, orientation='h')
            st.plotly_chart(fig)
        with col2:
            st.write("LLM-Analyse")
            st.write(llm_analysis)

        # Optional: Vorhersagen
        if st.checkbox("Vorhersagen machen"):
            input_data = {}
            for feature in feature_cols:
                input_data[feature] = st.number_input(f"Geben Sie einen Wert für {feature} ein")
            
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(scaler.transform(input_df))
            st.write(f"Vorhersage für {target_variable}: {prediction[0]:.2f}")

else:
    st.write("Bitte laden Sie eine CSV-Datei hoch, um zu beginnen.")