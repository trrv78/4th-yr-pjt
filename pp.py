import pandas as pd

import streamlit as st
import plotly.express as px
from datetime import timedelta
import time
import statsmodels.api as sm
import base64
import plotly.graph_objects as go
from supabase import create_client
from sklearn.svm import OneClassSVM


API_URL = 'https://xcveldffznwantuastlu.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhjdmVsZGZmem53YW50dWFzdGx1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDI2MzQ1MDYsImV4cCI6MjAxODIxMDUwNn0.jfjqBFAMrdumZ8_S5BPmzAadKcvN9BZjm02xUcyIkPQ'
supabase = create_client(API_URL, API_KEY)



@st.cache_data(ttl=60)  # Cache the data for 60 seconds
def fetch_data():
    supabase_list = supabase.table('maintable2').select('*').execute().data
    df = pd.DataFrame(supabase_list)
    df["DateTime"] = pd.to_datetime(df["created_at"])  # Convert "DateTime" column to datetime data type
    
    # Extract time-based features
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"] = df["DateTime"].dt.month
    df["Hour"] = df["DateTime"].dt.hour
    
    return df

st.set_page_config(page_title="BRIDGE Dashboard", layout='wide', initial_sidebar_state='collapsed')

# Sidebar Navigation
st.sidebar.header('Sensor Insights')
selected_insight = st.sidebar.selectbox("Choose an insight:", ["Summary Statistics", "Trend Analysis", "Alerts"])

# Fetch data
df = fetch_data()
df_recent = df.tail(10)

# Main Dashboard Area
st.header('Sensor Readings Dashboard')

# Show original data charts only on the Summary Statistics page
if selected_insight == "Summary Statistics":
    st.subheader('Temperature Readings')
    fig_temp = px.line(df.tail(5), x="DateTime", y="temperature", title=None, markers=True)
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader('Water Level Readings')
    fig_water_level = px.bar(df.tail(6), x="DateTime", y="distance", title=None)
    st.plotly_chart(fig_water_level, use_container_width=True)

    st.subheader('Vibration Scatter Plot')
    fig_acc_y = px.scatter(df.tail(20), x="DateTime", y="acceleration_y", title=None)
    st.plotly_chart(fig_acc_y, use_container_width=True)
    
    st.subheader('Weight Readings')
    weight = px.bar(df.tail(6), x="DateTime", y="weight", title=None)
    st.plotly_chart(weight, use_container_width=True)

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

# Alerts section
if selected_insight == "Alerts":
    # Check for zero distance values
    zero_distance_indices = df_recent[df_recent['distance'] == 0].index

    # Display alerts if there are zero distance readings
    if len(zero_distance_indices) > 0:
        st.error(f"**Alert:** Zero distance readings detected at {', '.join(map(str, zero_distance_indices))}")
       # st.error(f"**Alert:** Zero distance readings detected at {', '.join(zero_distance_indices.to_list())}")
    else:
        st.success("**No alerts:** No zero distance readings detected.")

    # Anomaly Detection using One-Class SVM
    svm_data = df_recent[['temperature', 'distance', 'acceleration_y']].dropna()  # Select relevant features
    svm_model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)  # Initialize One-Class SVM model
    svm_model.fit(svm_data)  # Fit the model on the data

    # Predict anomalies
    anomalies = svm_model.predict(svm_data)
    anomaly_indices = svm_data.index[anomalies == -1]  # Get indices of anomalies

    # Display anomalies
    if len(anomaly_indices) > 0:
        st.warning(f"**Anomalies Detected:** Anomalies detected at {', '.join(anomaly_indices.astype(str))}.")

        # Plot data with anomalies highlighted
        fig = px.scatter(svm_data, x='temperature', y='distance', color=anomalies)
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        st.plotly_chart(fig, use_container_width=True)

        # Display statistics related to anomalies
        st.subheader("Anomaly Statistics")
        st.write(f"Total anomalies detected: {len(anomaly_indices)}")
        st.write(f"Percentage of anomalies: {round(len(anomaly_indices)/len(svm_data) * 100, 2)}%")

    else:
        st.info("**No Anomalies Detected:** No anomalies detected in recent data.")

# Display summary statistics irrespective of selected insight
st.subheader("Summary Statistics")
st.write(df.describe())  # Display descriptive statistics


# Display time-based features on the Alerts page
if selected_insight == "Alerts":
    st.subheader("Time-Based Features")
    st.write(df_recent[["DateTime", "DayOfWeek", "Month", "Hour"]])

# Data export section
if st.button("Download Data as CSV"):
    csv = df_recent.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="sensor_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)




if st.button("Download Visualization as PDF"):
   
    # Add functionality to download the visualization as PDF
    # This would involve converting the visualization to PDF format and providing a download link to the user
    pass  # Placeholder for PDF export functionality

   
