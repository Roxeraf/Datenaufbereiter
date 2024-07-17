import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from datetime import datetime, timedelta

# Set up OpenAI API (make sure to use your API key)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load data
@st.cache_data
def load_data():
    quality_data = pd.read_excel("PS1 SERKEM 036 PP53 komplett – Kopie – Kopie.xlsx")
    weather_data = pd.read_csv("DecTod_Hum.csv")
    return quality_data, weather_data

# Preprocess data
def preprocess_data(quality_data, weather_data):
    # Combine date and time for quality data
    quality_data['DateTime'] = pd.to_datetime(quality_data['Datum'].astype(str) + ' ' + quality_data['Zeit'].astype(str))
    
    # Convert weather data timestamp
    weather_data['DateTime'] = pd.to_datetime(weather_data['DateTime'])
    
    # Merge data on nearest timestamp
    merged_data = pd.merge_asof(weather_data.sort_values('DateTime'), 
                                quality_data.sort_values('DateTime'), 
                                on='DateTime', 
                                direction='nearest')
    
    return merged_data

# Train ML model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, model.score(X_test_scaled, y_test)

# Get LLM explanation
def get_llm_explanation(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains data analysis results."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
st.title('Manufacturing Process Analysis')

# Load data
quality_data, weather_data = load_data()
merged_data = preprocess_data(quality_data, weather_data)

# Time frame selection
st.sidebar.header('Time Frame Selection')
start_date = st.sidebar.date_input('Start Date', merged_data['DateTime'].min())
end_date = st.sidebar.date_input('End Date', merged_data['DateTime'].max())

# Filter data based on selected time frame
filtered_data = merged_data[(merged_data['DateTime'].dt.date >= start_date) & 
                            (merged_data['DateTime'].dt.date <= end_date)]

# Select features and target
st.sidebar.header('Feature Selection')
feature_cols = st.sidebar.multiselect('Select features', merged_data.columns)
target_col = st.sidebar.selectbox('Select target variable', merged_data.columns)

if feature_cols and target_col:
    X = filtered_data[feature_cols]
    y = filtered_data[target_col]
    
    # Train model
    model, scaler, score = train_model(X, y)
    
    st.write(f"Model R² Score: {score:.2f}")
    
    # Feature importance
    importance = model.feature_importances_
    feat_importance = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    feat_importance = feat_importance.sort_values('importance', ascending=False)
    
    fig = px.bar(feat_importance, x='feature', y='importance', title='Feature Importance')
    st.plotly_chart(fig)
    
    # Time series plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=filtered_data['DateTime'], y=filtered_data[target_col], name=target_col))
    for feature in feature_cols:
        fig.add_trace(go.Scatter(x=filtered_data['DateTime'], y=filtered_data[feature], name=feature, visible='legendonly'))
    fig.update_layout(title=f'{target_col} and Selected Features Over Time')
    st.plotly_chart(fig)
    
    # LLM explanation
    st.header("Ask for Explanation")
    user_question = st.text_input("What would you like to know about the analysis?")
    if user_question:
        prompt = f"""
        The machine learning model analyzed manufacturing process data with the following results:
        - Target variable: {target_col}
        - Features used: {', '.join(feature_cols)}
        - Model R² Score: {score:.2f}
        - Top important features: {', '.join(feat_importance['feature'].head().tolist())}
        
        User question: {user_question}
        
        Please provide a clear and concise explanation.
        """
        explanation = get_llm_explanation(prompt)
        st.write(explanation)

else:
    st.write("Please select features and a target variable to begin the analysis.")