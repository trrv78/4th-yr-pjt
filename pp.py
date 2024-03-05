import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta
import time
import statsmodels.api as sm
import plotly.graph_objects as go  # Import the 'go' module

from supabase import create_client


API_URL = 'https://xcveldffznwantuastlu.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhjdmVsZGZmem53YW50dWFzdGx1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDI2MzQ1MDYsImV4cCI6MjAxODIxMDUwNn0.jfjqBFAMrdumZ8_S5BPmzAadKcvN9BZjm02xUcyIkPQ'
supabase = create_client(API_URL, API_KEY)

@st.cache_data(ttl=60)  # Cache the data for 60 seconds
def fetch_data():
    supabase_list = supabase.table('maintable2').select('*').execute().data
    df = pd.DataFrame(supabase_list)
    df["DateTime"] = pd.to_datetime(df["created_at"])  # Convert "DateTime" column to datetime data type
    return df

st.set_page_config(page_title="BRIDGE Dashboard", layout='wide', initial_sidebar_state='collapsed')

# Sidebar Navigation
st.sidebar.header('Sensor Insights')
selected_insight = st.sidebar.selectbox("Choose an insight:", ["Summary Statistics", "Trend Analysis", "Alerts"])

# Fetch data
df = fetch_data()
df_recent = df.tail(15)

# Main Dashboard Area
st.header('BRIDGE Sensor Readings')

# Show original data charts
st.subheader('Temperature Readings')

fig_temp = px.line(df_recent, x="DateTime", y="temperature", title=None, markers=True)
st.plotly_chart(fig_temp, use_container_width=True)

st.subheader('Water Level Readings')
fig_water_level = px.bar(df_recent, x="DateTime", y="distance", title=None)
fig_water_level.update_yaxes(autorange="reversed")
st.plotly_chart(fig_water_level, use_container_width=True)

st.subheader('Acceleration (Y-axis)')
fig_acc_y = px.line(df_recent, x="DateTime", y="acceleration_y", title=None, markers=True)
st.plotly_chart(fig_acc_y, use_container_width=True)

# Trend analysis section
if selected_insight == "Trend Analysis":
    st.subheader("Trend Analysis")

    # 1. Linear Regression - Water Level vs. Temperature
    X = df_recent['temperature']
    y = df_recent['distance']
    model = sm.OLS(y, X).fit()
    st.write(f"**Linear Regression Equation:** {model.summary().tables[0].data[1][0]}")  # Use two separate indices

    # Create a scatter plot with the regression line
    fig_reg = px.scatter(df_recent, x="temperature", y="distance", title="Water Level vs. Temperature")
    fig_reg.add_traces(go.Scatter(x=df_recent['temperature'], y=model.predict(df_recent['temperature']), mode='lines', line=dict(color='red')))  # Use go.Scatter and mode='lines'

    st.plotly_chart(fig_reg, use_container_width=True)

    # b. Linear Regression - Water Level vs. Vibrations
    X = df_recent['acceleration_y']
    y = df_recent['distance']
    model = sm.OLS(y, X).fit()
    st.write(f"**Linear Regression Equation:** {model.summary().tables[0].data[1][0]}")  # Use two separate indices

    # Create a scatter plot with the regression line
    fig_reg = px.scatter(df_recent, x="acceleration_y", y="distance", title="Water Level vs. vibrations")
    fig_reg.add_traces(go.Scatter(x=df_recent['acceleration_y'], y=model.predict(df_recent['acceleration_y']), mode='lines', line=dict(color='red')))  # Use go.Scatter and mode='lines'

    st.plotly_chart(fig_reg, use_container_width=True)

    # 2. Anomaly Detection - Acceleration
    # (You can choose an appropriate anomaly detection method)
    # Example using Interquartile Range (IQR)
    Q1 = df_recent['acceleration_y'].quantile(0.25)
    Q3 = df_recent['acceleration_y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df_recent[(df_recent['acceleration_y'] < lower_bound) | (df_recent['acceleration_y'] > upper_bound)]

    # Display anomaly points in red and normal points in blue
    fig_anom = px.scatter(df_recent, x="DateTime", y="acceleration_y", title="Acceleration (Y-axis)")

    # Use a list comprehension to create the marker colors directly, avoiding potential index access issues:
    marker_colors = ['#1f77b4' if ind not in anomalies.index else '#ff7f0e' for ind in df_recent.index]

    fig_anom.update_traces(marker=dict(color=marker_colors))
    st.plotly_chart(fig_anom, use_container_width=True)

elif selected_insight == "Summary Statistics":
    st.subheader("Summary Statistics")
    st.write(df.describe())  # Display descriptive statistics

elif selected_insight == "Alerts":
    st.subheader("")

    st.subheader("Alerts")

    # Check if distance is 0
    if (df_recent['distance'] == 0).any():
        st.warning("Alert: Distance is 0!")
