import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io
from datetime import datetime, timedelta

def load_data(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type in ['xlsx', 'xls']:
        return pd.read_excel(file)
    elif file_type == 'csv':
        return pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format")

def align_timestamps(df1, df2, time_col1, time_col2, tolerance_minutes=5):
    df1[time_col1] = pd.to_datetime(df1[time_col1])
    df2[time_col2] = pd.to_datetime(df2[time_col2])
    
    merged = pd.merge_asof(df1.sort_values(time_col1), 
                           df2.sort_values(time_col2), 
                           left_on=time_col1, 
                           right_on=time_col2, 
                           tolerance=pd.Timedelta(minutes=tolerance_minutes))
    
    return merged

def train_multioutput_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, Y_train)
    
    return model, scaler, X_test_scaled, Y_test

def analyze_feature_importance(model, feature_names, target_names):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    return feature_importance

st.title('Erweiterte Prozess- und Qualitätsanalyse mit ML')

process_file = st.file_uploader("Laden Sie Ihre Prozessdaten-Datei hoch", type=["xlsx", "xls", "csv"])
quality_file = st.file_uploader("Laden Sie Ihre Qualitätsdaten-Datei hoch", type=["xlsx", "xls", "csv"])

if process_file is not None and quality_file is not None:
    try:
        process_data = load_data(process_file)
        quality_data = load_data(quality_file)

        st.write("Prozessdaten geladen. Form:", process_data.shape)
        st.write("Qualitätsdaten geladen. Form:", quality_data.shape)

        # Auswahl der Zeitstempelspalten
        process_time = st.selectbox("Wählen Sie die Zeitstempelspalte für Prozessdaten", process_data.columns)
        quality_time = st.selectbox("Wählen Sie die Zeitstempelspalte für Qualitätsdaten", quality_data.columns)

        # Toleranz für Zeitstempelabgleich
        tolerance = st.number_input("Toleranz für Zeitstempelabgleich (in Minuten)", min_value=1, value=5)

        # Zusammenführen der Daten
        merged_data = align_timestamps(process_data, quality_data, process_time, quality_time, tolerance)
        st.write("Zusammengeführte Daten. Form:", merged_data.shape)

        # Auswahl der Zielvariablen
        target_variables = st.multiselect("Wählen Sie die Zielvariablen", merged_data.columns)

        # Auswahl der Eingabevariablen
        feature_cols = st.multiselect("Wählen Sie die Eingabevariablen", 
                                      [col for col in merged_data.columns if col not in target_variables])

        if st.button('Modell trainieren und analysieren'):
            X = merged_data[feature_cols]
            Y = merged_data[target_variables]

            model, scaler, X_test_scaled, Y_test = train_multioutput_model(X, Y)

            # Modellbewertung
            scores = model.score(X_test_scaled, Y_test)
            for target, score in zip(target_variables, scores):
                st.write(f"Modell R²-Score für {target}: {score:.2f}")

            # Merkmalswichtigkeit
            feature_importance = analyze_feature_importance(model, feature_cols, target_variables)
            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                         title='Gesamte Merkmalswichtigkeit')
            st.plotly_chart(fig)

            # Verbesserungsvorschläge
            st.subheader("Verbesserungsvorschläge:")
            top_features = feature_importance.head(5)['feature'].tolist()
            for feature in top_features:
                st.write(f"- Fokussieren Sie sich auf die Optimierung von '{feature}', da es einen starken Einfluss auf die Zielvariablen hat.")

            # Partial Dependence Plots
            for target in target_variables:
                st.subheader(f"Partial Dependence Plot für {target}")
                top_feature = feature_importance.iloc[0]['feature']
                pdp_feature = st.selectbox(f"Wählen Sie ein Merkmal für den Partial Dependence Plot ({target})", 
                                           [top_feature] + feature_cols)
                from sklearn.inspection import partial_dependence
                pdp = partial_dependence(model, X, [list(X.columns).index(pdp_feature)], target=list(Y.columns).index(target))
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
    st.write("Bitte laden Sie sowohl die Prozess- als auch die Qualitätsdaten-Datei hoch, um zu beginnen.")