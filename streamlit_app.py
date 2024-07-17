import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
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
def align_timestamps(df1, df2, time_col1, date_col2, time_col2, tolerance_minutes=5):
    def parse_timestamp(ts):
        if isinstance(ts, str):
            try:
                return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S')
        return ts

    df1[time_col1] = df1[time_col1].apply(parse_timestamp)
    df2['combined_datetime'] = pd.to_datetime(df2[date_col2].astype(str) + ' ' + df2[time_col2].astype(str))
    
    merged = pd.merge_asof(df1.sort_values(time_col1), 
                           df2.sort_values('combined_datetime'), 
                           left_on=time_col1, 
                           right_on='combined_datetime', 
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

def explain_model_results(feature_importance, scores, target_variables):
    explanation = "Modellauswertung:\n\n"
    
    for target, score in zip(target_variables, scores):
        explanation += f"Für die Zielvariable '{target}' hat das Modell einen R²-Score von {score:.2f}. "
        if score > 0.7:
            explanation += "Dies deutet auf eine gute Vorhersagekraft des Modells hin.\n"
        elif score > 0.5:
            explanation += "Dies deutet auf eine moderate Vorhersagekraft des Modells hin.\n"
        else:
            explanation += "Dies deutet auf eine schwache Vorhersagekraft des Modells hin. Möglicherweise fehlen wichtige Einflussfaktoren oder die Beziehung ist nicht linear.\n"
    
    explanation += "\nWichtigste Einflussfaktoren:\n"
    top_features = feature_importance.head(5)['feature'].tolist()
    for i, feature in enumerate(top_features, 1):
        explanation += f"{i}. {feature}: Dieses Merkmal hat einen starken Einfluss auf die Zielvariablen. "
        explanation += "Eine Optimierung dieses Faktors könnte zu signifikanten Verbesserungen führen.\n"
    
    explanation += "\nEmpfehlungen:\n"
    explanation += "1. Konzentrieren Sie sich auf die Optimierung der oben genannten Top-Einflussfaktoren.\n"
    explanation += "2. Untersuchen Sie die Partial Dependence Plots, um zu verstehen, wie Änderungen in diesen Faktoren die Zielvariablen beeinflussen.\n"
    explanation += "3. Wenn die Modellleistung niedrig ist, erwägen Sie die Sammlung zusätzlicher relevanter Daten oder die Anwendung fortgeschrittener Modellierungstechniken.\n"
    
    return explanation

st.title('Optimierte Prozess- und Qualitätsanalyse mit ML')

file1 = st.file_uploader("Laden Sie die erste Datei hoch (Datum und Zeit in einer Spalte)", type=["xlsx", "xls", "csv"])
file2 = st.file_uploader("Laden Sie die zweite Datei hoch (Datum und Zeit in getrennten Spalten)", type=["xlsx", "xls", "csv"])

if file1 and file2:
    df1 = load_data(file1)
    df2 = load_data(file2)

    st.write("Erste Datei geladen. Form:", df1.shape)
    st.write("Zweite Datei geladen. Form:", df2.shape)

    time_col1 = st.selectbox("Wählen Sie die Zeitstempelspalte für die erste Datei", df1.columns)
    date_col2 = st.selectbox("Wählen Sie die Datumsspalte für die zweite Datei", df2.columns)
    time_col2 = st.selectbox("Wählen Sie die Zeitspalte für die zweite Datei", df2.columns)

    tolerance = st.number_input("Toleranz für Zeitstempelabgleich (in Minuten)", min_value=1, value=5)

    if st.button('Daten zusammenführen'):
        try:
            merged_data = align_timestamps(df1, df2, time_col1, date_col2, time_col2, tolerance)
            st.session_state['merged_data'] = merged_data
            st.write("Daten erfolgreich zusammengeführt. Form:", merged_data.shape)
            st.success("Sie können nun die Zielvariablen und Eingabevariablen auswählen.")
        except Exception as e:
            st.error(f"Ein Fehler ist aufgetreten beim Zusammenführen der Daten: {e}")

    if 'merged_data' in st.session_state:
        merged_data = st.session_state['merged_data']
        
        if 'target_variables' not in st.session_state:
            st.session_state['target_variables'] = []
        
        all_columns = list(merged_data.columns)
        target_variables = st.multiselect(
            "Wählen Sie die Zielvariablen",
            options=all_columns,
            default=st.session_state['target_variables']
        )
        if st.button('Zielvariablen bestätigen'):
            st.session_state['target_variables'] = target_variables
            st.success(f"Zielvariablen ausgewählt: {', '.join(target_variables)}")

        if 'feature_cols' not in st.session_state:
            st.session_state['feature_cols'] = []
        
        feature_cols = st.multiselect(
            "Wählen Sie die Eingabevariablen",
            options=[col for col in all_columns if col not in target_variables],
            default=st.session_state['feature_cols']
        )
        if st.button('Eingabevariablen bestätigen'):
            st.session_state['feature_cols'] = feature_cols
            st.success(f"Eingabevariablen ausgewählt: {', '.join(feature_cols)}")

        if st.button('Analyse starten') and st.session_state['target_variables'] and st.session_state['feature_cols']:
            X = merged_data[st.session_state['feature_cols']]
            Y = merged_data[st.session_state['target_variables']]

            # Überprüfen und bereinigen der Daten
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    st.warning(f"Die Eingabevariable '{col}' enthält nicht-numerische Werte. Versuche Konvertierung...")
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())

            for col in Y.columns:
                if not pd.api.types.is_numeric_dtype(Y[col]):
                    st.warning(f"Die Zielvariable '{col}' enthält nicht-numerische Werte. Versuche Konvertierung...")
                    Y[col] = pd.to_numeric(Y[col], errors='coerce')
                if Y[col].isnull().any():
                    st.warning(f"Die Zielvariable '{col}' enthält NaN-Werte. Diese Zeilen werden entfernt.")
                    valid_indices = Y[col].notnull()
                    X = X[valid_indices]
                    Y = Y[valid_indices]

            st.write("Datenübersicht nach Bereinigung:")
            st.write(f"Eingabevariablen (X) Form: {X.shape}")
            st.write(f"Zielvariablen (Y) Form: {Y.shape}")

            # Zusätzliche Überprüfungen
            st.write("Datenübersicht vor dem Modelltraining:")
            for col in X.columns:
                st.write(f"{col}: Typ {X[col].dtype}, Nicht-Null: {X[col].count()}, Unique: {X[col].nunique()}")
            for col in Y.columns:
                st.write(f"{col}: Typ {Y[col].dtype}, Nicht-Null: {Y[col].count()}, Unique: {Y[col].nunique()}")

            # Überprüfen Sie auf konstante Spalten
            constant_cols_X = [col for col in X.columns if X[col].nunique() <= 1]
            constant_cols_Y = [col for col in Y.columns if Y[col].nunique() <= 1]
            if constant_cols_X:
                st.warning(f"Konstante Spalten in X gefunden: {constant_cols_X}")
            if constant_cols_Y:
                st.warning(f"Konstante Spalten in Y gefunden: {constant_cols_Y}")

            try:
                models, scalers, scores, importances = train_multioutput_model(X, Y)

                feature_importance = analyze_feature_importance(importances, st.session_state['feature_cols'], st.session_state['target_variables'])
                
                # Erklärung der Modellergebnisse
                explanation = explain_model_results(feature_importance, scores, st.session_state['target_variables'])
                st.subheader("Modellauswertung und Erklärung")
                st.write(explanation)

                fig = px.bar(feature_importance, x='average_importance', y='feature', orientation='h',
                             title='Durchschnittliche Merkmalswichtigkeit')
                st.plotly_chart(fig)

                for target, model, scaler in zip(st.session_state['target_variables'], models, scalers):
                    st.subheader(f"Partial Dependence Plot für {target}")
                    top_feature = feature_importance.iloc[0]['feature']
                    pdp_feature = st.selectbox(f"Wählen Sie ein Merkmal für den Partial Dependence Plot ({target})", 
                                               [top_feature] + st.session_state['feature_cols'])
                    try:
                        X_scaled = scaler.transform(X)
                        pdp_result = partial_dependence(model, X_scaled, [list(X.columns).index(pdp_feature)])
                        
                        st.write("PDP Ergebnis Struktur:", pdp_result.keys())
                        
                        if 'values' in pdp_result and 'average' in pdp_result:
                            fig_pdp = px.line(x=pdp_result['values'][0], y=pdp_result['average'][0], 
                                              labels={'x': pdp_feature, 'y': f'Partial dependence on {target}'})
                            st.plotly_chart(fig_pdp)
                            st.write(f"Der Partial Dependence Plot zeigt, wie sich Änderungen in '{pdp_feature}' auf '{target}' auswirken.")
                        else:
                            st.warning(f"Unerwartete Struktur der PDP-Ergebnisse für {target}. Überspringe Plot.")
                        
                        st.write("Datenübersicht für dieses Modell:")
                        st.write(f"X Shape: {X.shape}")
                        st.write(f"Y Shape für {target}: {Y[target].shape}")
                        st.write(f"Nicht-null Werte in Y für {target}: {Y[target].count()}")
                        st.write(f"Unique Werte in Y für {target}: {Y[target].nunique()}")
                        
                    except Exception as e:
                        st.error(f"Fehler beim Erstellen des Partial Dependence Plots für {target}: {str(e)}")
                        st.write("Debug-Informationen:")
                        st.write(f"Ausgewähltes Feature: {pdp_feature}")
                        st.write(f"Feature-Index: {list(X.columns).index(pdp_feature)}")
                        st.write(f"X Datentypen: {X.dtypes}")
                        st.write(f"Y Datentyp für {target}: {Y[target].dtype}")

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
                st.error(f"Ein Fehler ist während der Analyse aufgetreten: {str(e)}")
                st.write("Bitte überprüfen Sie Ihre Daten und stellen Sie sicher, dass alle ausgewählten Variablen numerisch sind und keine fehlenden Werte enthalten.")

else:
    st.write("Bitte laden Sie beide Dateien hoch, um zu beginnen.")