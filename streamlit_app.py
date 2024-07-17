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

# Get LLM guidance
def get_llm_guidance(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant guiding users through data analysis."},
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

# Display data info
st.subheader("Quality Data Columns")
st.write(quality_data.columns.tolist())
st.subheader("Weather Data Columns")
st.write(weather_data.columns.tolist())

# LLM guidance for column selection
guidance_prompt = f"""
Given the following column names for quality and weather data, please ask the user to identify the correct columns for date and time (or datetime if combined).

Quality Data Columns: {quality_data.columns.tolist()}
Weather Data Columns: {weather_data.columns.tolist()}

Please formulate questions to ask the user about:
1. Which column in the quality data represents the date?
2. Which column in the quality data represents the time?
3. Which column in the weather data represents the datetime?

Format your response as questions that can be directly presented to the user.
"""

guidance = get_llm_guidance(guidance_prompt)
st.write(guidance)

# User input based on LLM guidance
quality_date_col = st.text_input("Enter the column name for date in quality data:")
quality_time_col = st.text_input("Enter the column name for time in quality data:")
weather_datetime_col = st.text_input("Enter the column name for datetime in weather data:")

if st.button('Process Data'):
    # Preprocess data
    quality_data['DateTime'] = pd.to_datetime(quality_data[quality_date_col].astype(str) + ' ' + quality_data[quality_time_col].astype(str), errors='coerce')
    weather_data['DateTime'] = pd.to_datetime(weather_data[weather_datetime_col], errors='coerce')
    
    # Merge data on nearest timestamp
    merged_data = pd.merge_asof(weather_data.sort_values('DateTime'), 
                                quality_data.sort_values('DateTime'), 
                                on='DateTime', 
                                direction='nearest')
    
    st.write("First few rows of merged data:", merged_data.head())
    st.write("Merged data info:", merged_data.info())
    
    # Time frame selection
    st.sidebar.header('Time Frame Selection')
    start_date = st.sidebar.date_input('Start Date', merged_data['DateTime'].min().date())
    end_date = st.sidebar.date_input('End Date', merged_data['DateTime'].max().date())

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
            explanation = get_llm_guidance(prompt)
            st.write(explanation)

    else:
        st.write("Please select features and a target variable to begin the analysis.")
# Funktion zum Trainieren des Modells
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, scaler, score
