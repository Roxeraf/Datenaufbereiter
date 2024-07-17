import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io
from datetime import datetime, timedelta
import concurrent.futures
from joblib import Parallel, delayed

@st.cache_data
def load_data(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type in ['xlsx', 'xls']:
        return pd.read_excel(file)
    elif file_type == 'csv':
        return pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format")

@st.cache_data
def align_timestamps(df1, df2, time_col1, time_col2, tolerance_minutes=5):
    df1[time_col1] = pd.to_datetime(df1[time_col1])
    df2[time_col2] = pd.to_datetime(df2[time_col2])
    
    merged = pd.merge_asof(df1.sort_values(time_col1), 
                           df2.sort_values(time_col2), 
                           left_on=time_col1, 
                           right_on=time_col2, 
                           tolerance=pd.Timedelta(minutes=tolerance_minutes))
    
    return merged

def train_single_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    score = model.score(X_test_scaled, y_test)
    importance = model.feature_importances_
    
    return model, scaler, score, importance

def train_multioutput_model(X, Y):
    models = []
    scalers = []
    scores = []
    importances = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_single_model, X, Y[col]) for col in Y.columns]
        for future in concurrent.futures.as_completed(futures):
            model, scaler, score, importance = future.result()
            models.append(model)
            scalers.append(scaler)
            scores.append(score)
            importances.append(importance)
    
    return models, scalers, scores, importances

def analyze_feature_importance(importances, feature_names, target_names):
    feature_importance = pd.DataFrame({'feature': feature_names})
    for target, importance in zip(target_names, importances):
        feature_importance[target] = importance
    feature_importance['average_importance'] = feature_importance.iloc[:, 1:].mean(axis=1)
    feature_importance = feature_importance.sort_values('average_importance', ascending=False)
    return feature_importance

st.title('Optimierte Prozess- und Qualitätsanalyse mit ML')

num_files = st.number_input("Anzahl der zu ladenden Dateien", min_value=2, value=2)
files = []
data_frames = []

for i in range(num_files):
    file = st.file_uploader(f"Laden Sie Datei {i+1} hoch", type=["xlsx", "xls", "csv"])
    if file:
        files.append(file)
        data_frames.append(load_data(file))

if len(files) == num_files:
    try:
        for i, df in enumerate(data_frames):
            st.write(f"Datei {i+1} geladen. Form:", df.shape)

        # Auswahl der Zeitstempelspalten für jede Datei
        time_columns = [st.selectbox(f"Wählen Sie die Zeitstempelspalte für Datei {i+1}", df.columns) for i, df in enumerate(data_frames)]

        # Toleranz für Zeitstempelabgleich
        tolerance = st.number_input("Toleranz für Zeitstempelabgleich (in Minuten)", min_value=1, value=5)

        # Zusammenführen der Daten
        merged_data = data_frames[0]
        for i in range(1, len(data_frames)):
            merged_data = align_timestamps(merged_data, data_frames[i], time_columns[0], time_columns[i], tolerance)
        st.write("Zusammengeführte Daten. Form:", merged_data.shape)

        # Auswahl der Zielvariablen
        target_variables = st.multiselect("Wählen Sie die Zielvariablen", merged_data.columns)

        # Auswahl der Eingabevariablen
        feature_cols = st.multiselect("Wählen Sie die Eingabevariablen", 
                                      [col for col in merged_data.columns if col not in target_variables])

        if st.button('Modell trainieren und analysieren'):
            X = merged_data[feature_cols]
            Y = merged_data[target_variables]

            models, scalers, scores, importances = train_multioutput_model(X, Y)

            # Modellbewertung
            for target, score in zip(target_variables, scores):
                st.write(f"Modell R²-Score für {target}: {score:.2f}")

            # Merkmalswichtigkeit
            feature_importance = analyze_feature_importance(importances, feature_cols, target_variables)
            fig = px.bar(feature_importance, x='average_importance', y='feature', orientation='h',
                         title='Durchschnittliche Merkmalswichtigkeit')
            st.plotly_chart(fig)

            # Verbesserungsvorschläge
            st.subheader("Verbesserungsvorschläge:")
            top_features = feature_importance.head(5)['feature'].tolist()
            for feature in top_features:
                st.write(f"- Fokussieren Sie sich auf die Optimierung von '{feature}', da es einen starken Einfluss auf die Zielvariablen hat.")

            # Partial Dependence Plots
            for target, model, scaler in zip(target_variables, models, scalers):
                st.subheader(f"Partial Dependence Plot für {target}")
                top_feature = feature_importance.iloc[0]['feature']
                pdp_feature = st.selectbox(f"Wählen Sie ein Merkmal für den Partial Dependence Plot ({target})", 
                                           [top_feature] + feature_cols)
                from sklearn.inspection import partial_dependence
                X_scaled = scaler.transform(X)
                pdp = partial_dependence(model, X_scaled, [list(X.columns).index(pdp_feature)])
                fig_pdp = px.line(x=pdp['values'][0], y=pdp['average'][0], 
                                  labels={'x': pdp_feature, 'y': f'Partial dependence on {target}'})
                st.plotly_chart(fig_pdp)

                st.write(f"Der Partial Dependence Plot zeigt, wie sich Änderungen in '{pdp_feature}' auf '{target}' auswirken.")

            # Option zum Herunterladen der Ergebnisse als Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                feature_importance.to_excel(writer, index=False, sheet_name='Feature Importance')
                merged_data.to_excel(writer, index=False, sheet_name='Merged Data')
            output.seek(0)
            
            st.download_button(
                label="Download Ergebnisse als Excel",
                data=output,
                file_name="analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")

else:
    st.write(f"Bitte laden Sie alle {num_files} Dateien hoch, um zu beginnen.")