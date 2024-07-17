import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

def load_data(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type in ['xlsx', 'xls']:
        return pd.read_excel(file)
    elif file_type == 'csv':
        return pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format")

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

def analyze_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    return feature_importance

st.title('Prozess- und Qualitätsanalyse mit ML')

process_file = st.file_uploader("Laden Sie Ihre Prozessdaten-Datei hoch", type=["xlsx", "xls", "csv"])
quality_file = st.file_uploader("Laden Sie Ihre Qualitätsdaten-Datei hoch", type=["xlsx", "xls", "csv"])

if process_file is not None and quality_file is not None:
    try:
        process_data = load_data(process_file)
        quality_data = load_data(quality_file)

        st.write("Prozessdaten geladen. Form:", process_data.shape)
        st.write("Qualitätsdaten geladen. Form:", quality_data.shape)

        # Auswahl der Schlüsselspalte für das Zusammenführen
        process_key = st.selectbox("Wählen Sie die Schlüsselspalte für Prozessdaten", process_data.columns)
        quality_key = st.selectbox("Wählen Sie die Schlüsselspalte für Qualitätsdaten", quality_data.columns)

        # Zusammenführen der Daten
        merged_data = pd.merge(process_data, quality_data, left_on=process_key, right_on=quality_key, how='inner')
        st.write("Zusammengeführte Daten. Form:", merged_data.shape)

        # Auswahl der Zielvariable
        target_variable = st.selectbox("Wählen Sie die Zielvariable", merged_data.columns)

        # Auswahl der Eingabevariablen
        feature_cols = st.multiselect("Wählen Sie die Eingabevariablen", 
                                      [col for col in merged_data.columns if col != target_variable])

        if st.button('Modell trainieren und analysieren'):
            X = merged_data[feature_cols]
            y = merged_data[target_variable]

            model, scaler, X_test_scaled, y_test = train_model(X, y)

            # Modellbewertung
            score = model.score(X_test_scaled, y_test)
            st.write(f"Modell R²-Score: {score:.2f}")

            # Merkmalswichtigkeit
            feature_importance = analyze_feature_importance(model, feature_cols)
            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                         title='Merkmalswichtigkeit')
            st.plotly_chart(fig)

            # Verbesserungsvorschläge
            st.subheader("Verbesserungsvorschläge:")
            top_features = feature_importance.head(3)['feature'].tolist()
            for feature in top_features:
                st.write(f"- Fokussieren Sie sich auf die Optimierung von '{feature}', da es einen starken Einfluss auf {target_variable} hat.")

            # Partial Dependence Plot
            top_feature = feature_importance.iloc[0]['feature']
            pdp_feature = st.selectbox("Wählen Sie ein Merkmal für den Partial Dependence Plot", 
                                       [top_feature] + feature_cols)
            from sklearn.inspection import partial_dependence
            pdp = partial_dependence(model, X, [list(X.columns).index(pdp_feature)])
            fig_pdp = px.line(x=pdp[1][0], y=pdp[0][0], 
                              labels={'x': pdp_feature, 'y': f'Partial dependence on {target_variable}'})
            st.plotly_chart(fig_pdp)

            st.write(f"Der Partial Dependence Plot zeigt, wie sich Änderungen in '{pdp_feature}' auf '{target_variable}' auswirken.")

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