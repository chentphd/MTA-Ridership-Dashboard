import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime

from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA

import numpy as np
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller 
from prophet import Prophet
import statsmodels.api as sm
 
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
#import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# Streamlit interface
st.title('MTA Subway Station Ridership')

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Ridership Map Visualization", #tab1
    "Station Ridership", #tab2
    "Station Comparison", #tab3
    "Hourly Trends", #tab4
    "Ridership Prediction 1", #tab5,
    "Ridership Prediction 2" #tab6
       
    
])
with tab1:
    st.subheader("Ridership Map Visualization in the First Week of 2023")

    # Load the CSV and process the datetime index
    #df_new = pd.read_csv(r"C:\Users\tonychen\Downloads\Ridership__Beginning_Jan_1_2023 to Jan_8_2023.csv") #Local Usage 
    df_new = pd.read_csv("Ridership__Beginning_Jan_1_2023 to Jan_8_2023.csv")
    df_new.index = pd.to_datetime(df_new['transit_timestamp'])
    df_new = df_new.drop(['transit_timestamp'], axis=1)

    # Add a date range and time selection slider
    min_date_tab4 = df_new.index.min().date()
    max_date_tab4 = df_new.index.max().date()

    start_date_tab4, end_date_tab4 = st.select_slider(
        'Select a Start and End date',
        options=pd.date_range(min_date_tab4, max_date_tab4).date,
        value=(min_date_tab4, max_date_tab4),
        key='select_slider_tab4'
    )

    # Filter the dataframe for the selected date range
    filtered_map_df = df_new[(df_new.index.date >= start_date_tab4) & (df_new.index.date <= end_date_tab4)]

    # Use latitude and longitude from the dataset for mapping and aggregate by station
    filtered_map_df = filtered_map_df.groupby(['station_complex', 'latitude', 'longitude'], as_index=False)["ridership"].sum()

    # Map using Plotly Express
    fig_map = px.scatter_mapbox(
        filtered_map_df,
        lat='latitude',
        lon='longitude',
        size='ridership',
        color='ridership',
        hover_name='station_complex',
        hover_data=['ridership'],
        color_continuous_scale='Viridis',
        title='Ridership by Station Location',
        zoom=10,
        height=600
    )

    fig_map.update_layout(mapbox_style='open-street-map')
    fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig_map.update_xaxes(rangeslider_visible=True)

    # Display map plot within the dashboard
    st.plotly_chart(fig_map)

    # Show total ridership for the filtered period
    total_ridership_map = filtered_map_df['ridership'].sum()
    st.metric("Total Ridership in Selected Period", f"{total_ridership_map:,}")

# Load your dataset
#df = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\MTA Peak Ridership\MTA_Subway_Ridership_2023.csv") #Only 2023 Data 
#df = pd.read_csv(r"C:\Users\tonychen\Downloads\MTA_Subway_Hourly_Ridership__Beginning_July_2020_20241024.csv") #All Data from 2022 to 2024 #Local Usage
df = pd.read_csv("MTA_Subway_Hourly_Ridership__Beginning_July_2020_20241024.csv") #All Data from 2022 to 2024 


# Set 'transit_timestamp' as the index and convert to datetime
df.index = df['transit_timestamp']
df = df.drop(['transit_timestamp'], axis=1)
df.index = pd.to_datetime(df.index)

with tab2:
    # Date range selection
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    # Add a progress bar for date selection
    progress_text = "Processing date selection. Please wait..."
    date_bar = st.progress(0, text=progress_text)

    # Add a select slider for Date range
    start_date, end_date = st.select_slider(
        'Select a Start and End date',
        options=pd.date_range(min_date, max_date).date,
        value=(min_date, max_date)
    )

    # Simulate progress for date selection
    for percent_complete in range(100):
        time.sleep(0.01)
        date_bar.progress(percent_complete + 1, text=progress_text)
    date_bar.empty()  # Clear the progress bar

    # Station selection
    stations = df['station_complex'].unique().tolist()

    # Add a progress bar for station selection
    progress_text = "Processing station selection. Please wait..."
    station_bar = st.progress(0, text=progress_text)

    # Station selection
    selected_stations = st.multiselect('Select stations (leave empty for all)', stations)

    # Simulate progress for station selection
    for percent_complete in range(100):
        time.sleep(0.01)
        station_bar.progress(percent_complete + 1, text=progress_text)
    station_bar.empty()  # Clear the progress bar

    # Filter data based on selection
    filtered_df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    if selected_stations:
        filtered_df = filtered_df[filtered_df['station_complex'].isin(selected_stations)]

    # Group by timestamp and sum ridership
    ridership_data = filtered_df.groupby(filtered_df.index).sum()

    # If all stations are selected, format the data with 'All Stations'
    if not selected_stations:
        ridership_data['all_stations'] = 'All Stations'
        ridership_data = ridership_data.loc[:, ['all_stations', 'ridership', 'station_complex']]

    # Create line chart
    fig = px.line(ridership_data, x=ridership_data.index, y='ridership', title='Ridership Over Time')
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)

    # Display the data used
    st.subheader('Ridership Data Used')
    st.dataframe(ridership_data)

    total_ridership = ridership_data['ridership'].sum()
    st.metric("Total Ridership", f"{total_ridership:,}")

with tab3:
    # Split layout for station comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Station 1")

        # Date range selection for Station 1
        start_date_1, end_date_1 = st.select_slider(
            'Select a Start and End date (Station 1)',
            options=pd.date_range(min_date, max_date).date,
            value=(min_date, max_date)
        )

        # Time range selection for Station 1
        start_time_1, end_time_1 = st.slider(
            'Select a Time range (Station 1)',
            min_value=0,
            max_value=23,
            value=(0, 23),
            step=1
        )

        # Station selection for Station 1
        selected_stations_1 = st.multiselect('Select stations for Station 1 (leave empty for all)', stations)

        # Filter data for Station 1
        filtered_df_1 = df[(df.index.date >= start_date_1) & (df.index.date <= end_date_1)]
        filtered_df_1 = filtered_df_1[(filtered_df_1.index.hour >= start_time_1) & (filtered_df_1.index.hour <= end_time_1)]
        if selected_stations_1:
            filtered_df_1 = filtered_df_1[filtered_df_1['station_complex'].isin(selected_stations_1)]

        ridership_data_1 = filtered_df_1.groupby(filtered_df_1.index).sum()

        if not selected_stations_1:
            ridership_data_1['all_stations'] = 'All Stations'
            ridership_data_1 = ridership_data_1.loc[:, ['all_stations', 'ridership', 'station_complex']]

        fig_1 = px.line(ridership_data_1, x=ridership_data_1.index, y='ridership', title='Station 1 Ridership')
        fig_1.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_1)

        # Display total ridership for Station 1
        total_ridership_1 = ridership_data_1['ridership'].sum()
        st.metric("Total Ridership (Station 1)", f"{total_ridership_1:,}")

    with col2:
        st.subheader("Station 2")

        # Date range selection for Station 2
        start_date_2, end_date_2 = st.select_slider(
            'Select a Start and End date (Station 2)',
            options=pd.date_range(min_date, max_date).date,
            value=(min_date, max_date)
        )

        # Time range selection for Station 2
        start_time_2, end_time_2 = st.slider(
            'Select a Time range (Station 2)',
            min_value=0,
            max_value=23,
            value=(0, 23),
            step=1
        )

        # Station selection for Station 2
        selected_stations_2 = st.multiselect('Select stations for Station 2 (leave empty for all)', stations)

        # Filter data for Station 2
        filtered_df_2 = df[(df.index.date >= start_date_2) & (df.index.date <= end_date_2)]
        filtered_df_2 = filtered_df_2[(filtered_df_2.index.hour >= start_time_2) & (filtered_df_2.index.hour <= end_time_2)]
        if selected_stations_2:
            filtered_df_2 = filtered_df_2[filtered_df_2['station_complex'].isin(selected_stations_2)]

        ridership_data_2 = filtered_df_2.groupby(filtered_df_2.index).sum()

        if not selected_stations_2:
            ridership_data_2['all_stations'] = 'All Stations'
            ridership_data_2 = ridership_data_2.loc[:, ['all_stations', 'ridership', 'station_complex']]

        fig_2 = px.line(ridership_data_2, x=ridership_data_2.index, y='ridership', title='Station 2 Ridership')
        fig_2.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_2)

        # Display total ridership for Station 2
        total_ridership_2 = ridership_data_2['ridership'].sum()
        st.metric("Total Ridership (Station 2)", f"{total_ridership_2:,}")

with tab4: 
    st.subheader("Hourly Trends by Station")

    # Load and preprocess the data
    #df_hourly_trends = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\MTA Peak Ridership\MTA_Subway_Ridership_2023.csv") #Only 2023
    #df_hourly_trends = pd.read_csv(r"C:\Users\tonychen\Downloads\MTA_Subway_Hourly_Ridership__Beginning_July_2020_20241024.csv") #All Data from 2022 to 2024 #Local Usage
    ##############################################
    #df_hourly_trends = pd.read_csv("MTA_Subway_Hourly_Ridership__Beginning_July_2020_20241024.csv") #All Data from 2022 to 2024
    # Convert 'transit_timestamp' to datetime format
    #df_hourly_trends['transit_timestamp'] = pd.to_datetime(df_hourly_trends['transit_timestamp'])
    # Set 'transit_timestamp' as the index
    #df_hourly_trends = df_hourly_trends.set_index('transit_timestamp')
    ##############################################

    df_hourly_trends = df

    # Station selection
    stations_4 = df_hourly_trends['station_complex'].unique().tolist()
    station_hourly = st.selectbox('Select a station for hourly trend analysis', stations_4, key = 'selectbox_tab4')
    
    # Filter data for the selected station
    station_df_hourly = df_hourly_trends[df_hourly_trends['station_complex'] == station_hourly]
    
    # Group by hour of day and calculate mean ridership
    hourly_trends = station_df_hourly.groupby(station_df_hourly.index.hour)['ridership'].mean()
    
    # Plot hourly trends
    fig_hourly = px.line(hourly_trends, x=hourly_trends.index, y=hourly_trends.values, 
                         title=f'Average Hourly Ridership for {station_hourly}', 
                         labels={'x': 'Hour of Day', 'y': 'Average Ridership'})

    fig_hourly.update_xaxes(rangeslider_visible=True)                      
    st.plotly_chart(fig_hourly)




with tab5:
    st.subheader("Predict Hourly Ridership Using Prophet")
    #df_5 = pd.read_csv(r"C:\Users\tonychen\Documents\Python Files\MTA Peak Ridership\MTA_Subway_Ridership_2023.csv") #Only 2023 Data 
    #df_5 = pd.read_csv(r"C:\Users\tonychen\Downloads\MTA_Subway_Hourly_Ridership__Beginning_July_2020_20241024.csv") #All Data from 2022 to 2024 #Local Usage 
    #df_5 = pd.read_csv("MTA_Subway_Hourly_Ridership__Beginning_July_2020_20241024.csv") #All Data from 2022 to 2024 
    ######################################
    #df_5.index = df_5['transit_timestamp']
    #df_5 = df_5.drop(['transit_timestamp'], axis=1)
    #df_5.index = pd.to_datetime(df_5.index)
    ###############################################
    df_5 = df
    stations_prophet = df_5['station_complex'].unique().tolist()

    # Select station for prediction
    station_for_prediction = st.selectbox('Select a station for prediction', stations_prophet)

    # Filter data for selected station
    station_df = df_5[df_5['station_complex'] == station_for_prediction]

    # Prepare data for Prophet
    prophet_df = station_df.resample('H').sum().reset_index()
    prophet_df = prophet_df.rename(columns={'transit_timestamp': 'ds', 'ridership': 'y'})
