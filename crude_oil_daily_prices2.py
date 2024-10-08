#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_excel('C:/Users/91970/Downloads/Crude Oil Prices Daily.xlsx')

# Assuming the correct column names are 'Date' and 'Closing Value'
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
df.set_index('Date', inplace=True)

# Function to make predictions with the ARIMA model
def predict_crude_oil_price(data, days_to_forecast):
    # Fit the ARIMA model
    model = ARIMA(data['Closing Value'], order=(5, 1, 0))
    model_fit = model.fit()  # Fit the model
    forecast = model_fit.forecast(steps=days_to_forecast)  # Forecast using the fitted model
    
    # Create forecast dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)

    # Combine dates and forecast into a DataFrame
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})
    return forecast_df

# Streamlit app
st.title("C:/Users/91970/Downloads/Crude Oil Price Forecasting App for ARIMA Model")


# Add an image of crude oil
st.image("/crude-oil-image.jpg", caption="Crude Oil", use_column_width=True)


# Date range input for the historical data
start_date = st.date_input("Select the start date for historical data:", df.index.min())
end_date = st.date_input("Select the end date for historical data:", df.index.max())

# Filter the DataFrame based on user-selected dates
filtered_df = df[start_date:end_date]

# Input for the number of days to forecast
days_to_forecast = st.number_input("Enter the number of days to forecast:", min_value=1, max_value=365, value=30)

# Button to trigger prediction
if st.button("Forecast"):
    if not filtered_df.empty:
        forecast_df = predict_crude_oil_price(filtered_df, days_to_forecast)

        # Display the forecast
        st.subheader("Forecasted Crude Oil Prices:")
        st.write(forecast_df)

        # Prepare for plotting
        plt.figure(figsize=(10, 6))
        
        # Plot the original closing values
        plt.plot(filtered_df['Closing Value'], label='Original Prices', color='blue', linewidth=2)

        # Create forecast index for plotting
        forecast_index = forecast_df['Date']  # Use the forecast dates directly

        # Ensure forecast is a pandas Series
        forecast_series = pd.Series(forecast_df['Forecasted Price'].values, index=forecast_index)

        # Plot the ARIMA forecast
        plt.plot(forecast_series.index, forecast_series, label='ARIMA Forecast', color='orange', linestyle='--', linewidth=2)

        plt.xlabel("Date")
        plt.ylabel("Crude Oil Price")
        plt.title("Crude Oil Price Forecast")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("No data available for the selected date range.")


# In[ ]:




