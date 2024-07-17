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
    return pd.read_csv(file)

def get_llm_preprocessing_steps(df):
    # Erstellen Sie eine Beschreibung des Datensatzes
    data_description = df.dtypes.to_string() + "\n\n"
    data_description += df.describe().to_string() + "\n\n"
    data_description += df.isnull().sum().to_string()

    prompt = f"""
    Analysiere den folgenden Datensatz und schlage Schritte zur Datenbereinigung und -vorverarbeitung vor:

    {data_description}

    Bitte gib eine Liste von Python-Codezeilen, die folgende Aufgaben erfüllen:
    1. Behandlung von fehlenden Werten
    2. Umgang mit Ausreißern
    3. Kodierung kategorialer Variablen
    4. Normalisierung oder Standardisierung numerischer Variablen
    5. Erstellung neuer Features, falls sinnvoll

    Gib nur den Python-Code zurück, ohne zusätzliche Erklärungen.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du bist ein Experte für Datenvorverarbeitung in Python."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Fehler bei der LLM-Analyse: {str(e)}")
        return "LLM-Analyse konnte nicht durchgeführt werden."

def apply_preprocessing(df, preprocessing_steps):
    # Führen Sie die vom LLM vorgeschlagenen Vorverarbeitungsschritte aus
    try:
        exec(preprocessing_steps, globals(), {"df": df})
        return df
    except Exception as e:
        st.error(f"Fehler bei der Anwendung der Vorverarbeitungsschritte: {str(e)}")
        return df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    return model, scaler, model.score(X_test_scaled, y_test)

st.title('ML-Modell mit LLM-unterstützter Datenvorverarbeitung')

uploaded_file = st.file_uploader("Laden Sie Ihre CSV-Datei hoch", type="csv")

if uploaded_file is not None:
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

else:
    st.write("Bitte laden Sie eine CSV-Datei hoch, um zu beginnen.")