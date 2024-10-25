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
