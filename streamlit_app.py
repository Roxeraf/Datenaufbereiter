import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta, date
import io

# Set up OpenAI API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load data
@st.cache_data
def load_data():
    quality_data = pd.read_excel("PS1 SERKEM 036 PP53 komplett – Kopie – Kopie.xlsx")
    weather_data = pd.read_csv("DecTod_Hum.csv")
    return quality_data, weather_data

# Get LLM guidance
def get_llm_guidance(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant guiding users through data analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to train the model
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, scaler, score

# Streamlit app
st.title('Manufacturing Process Analysis')

# Load data
quality_data, weather_data = load_data()

# Set column names for quality data and weather data
quality_date_col = 'Angelegt am'
quality_time_col = 'Uhrzeit'
weather_datetime_col = 'dtVar01_pddb_rxxs'

# Display first few rows of the weather data to check the datetime format
st.write("First few rows of weather data:")
st.write(weather_data.head())

# Preprocess weather data
try:
    weather_data['DateTime'] = pd.to_datetime(weather_data[weather_datetime_col])
except Exception as e:
    st.error(f"Error processing weather datetime column: {e}")

# Preprocess quality data
try:
    quality_data['DateTime'] = pd.to_datetime(quality_data[quality_date_col].astype(str) + ' ' + quality_data[quality_time_col].astype(str), errors='coerce')
except Exception as e:
    st.error(f"Error processing quality datetime columns: {e}")

# Remove rows with NaT values in DateTime columns
quality_data = quality_data.dropna(subset=['DateTime'])
weather_data = weather_data.dropna(subset=['DateTime'])

# Ensure both DataFrames have DateTime as datetime type
weather_data['DateTime'] = pd.to_datetime(weather_data['DateTime'])
quality_data['DateTime'] = pd.to_datetime(quality_data['DateTime'])

# Sort both DataFrames by DateTime
weather_data = weather_data.sort_values('DateTime')
quality_data = quality_data.sort_values('DateTime')

# Initialize session state for time frame selection
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = weather_data['DateTime'].min().date()
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = weather_data['DateTime'].max().date()

# Ensure session state values are date objects
if isinstance(st.session_state['start_date'], str):
    st.session_state['start_date'] = pd.to_datetime(st.session_state['start_date']).date()
if isinstance(st.session_state['end_date'], str):
    st.session_state['end_date'] = pd.to_datetime(st.session_state['end_date']).date()

# Time frame selection
st.sidebar.header('Time Frame Selection')
start_date = st.sidebar.date_input('Start Date', st.session_state['start_date'])
end_date = st.sidebar.date_input('End Date', st.session_state['end_date'])

# Update session state with selected dates
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date

# Filter data based on selected time frame
weather_data = weather_data[(weather_data['DateTime'].dt.date >= st.session_state['start_date']) & 
                            (weather_data['DateTime'].dt.date <= st.session_state['end_date'])]
quality_data = quality_data[(quality_data['DateTime'].dt.date >= st.session_state['start_date']) & 
                            (quality_data['DateTime'].dt.date <= st.session_state['end_date'])]

# Merge data on nearest timestamp
merged_data = pd.merge_asof(weather_data, quality_data, on='DateTime', direction='nearest')

st.write("Data preprocessing completed.")
st.write(f"Weather data shape: {weather_data.shape}")
st.write(f"Quality data shape: {quality_data.shape}")
st.write(f"Merged data shape: {merged_data.shape}")

st.write("First few rows of merged data:")
st.write(merged_data.head())

# Displaying DataFrame info in a text format to avoid BrokenPipeError
buffer = io.StringIO()
merged_data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Select features and target
st.sidebar.header('Feature Selection')
feature_cols = st.sidebar.multiselect('Select features', merged_data.columns)
target_col = st.sidebar.selectbox('Select target variable', merged_data.columns)

# Process Data button
if st.sidebar.button('Process Data'):
    if feature_cols and target_col:
        X = merged_data[feature_cols]
        y = merged_data[target_col]

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
        fig.add_trace(go.Scatter(x=merged_data['DateTime'], y=merged_data[target_col], name=target_col))
        for feature in feature_cols:
            fig.add_trace(go.Scatter(x=merged_data['DateTime'], y=merged_data[feature], name=feature, visible='legendonly'))
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