#!/usr/bin/env python3
"""
Simplified Time Series Forecasting App
Fully compatible with the new sequential LSTM and CNN architectures
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from data_processor import TimeSeriesDataProcessor
from deep_learning_forecasting import LSTMForecasting, CNNForecasting
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

# Title and description
st.title("Time Series Forecasting with Deep Learning")
st.write("This app allows you to visualize, process, and forecast time series data using LSTM and CNN models.")

# Sidebar for dataset and model selection
st.sidebar.header("Configuration")
dataset_options = ["airline_passengers", "female_births", "restaurant_visitors", "superstore_sales"]
selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)
model_type = st.sidebar.selectbox("Select Model", ["LSTM", "CNN"])
seq_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=50, value=20)
forecast_steps = st.sidebar.slider("Forecast Steps", min_value=10, max_value=100, value=30)
train_model = st.sidebar.button("Train Model")

# Data loading and processing
@st.cache_data
def load_and_process_data(dataset_name):
    try:
        processor = TimeSeriesDataProcessor()
        data = processor.load_dataset(dataset_name)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

data = load_and_process_data(selected_dataset)

if data is not None:
    # Visualize raw data
    st.subheader("Raw Data Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data, label="Raw Data")
    ax.set_title(f"{selected_dataset.replace('_', ' ').title()} Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
    
    # Initialize and train model
    if train_model or "model" not in st.session_state:
        try:
            if model_type == "LSTM":
                model = LSTMForecasting(seq_length=seq_length, hidden_size=128, num_layers=2, dropout=0.2)
                training_result = model.train_model(data, epochs=100, learning_rate=0.001, validation_split=0.2)
            else:  # CNN
                model = CNNForecasting(seq_length=seq_length, filters=128, kernel_size=3)
                training_result = model.train_model(data, epochs=100, batch_size=32, validation_split=0.2)
            
            if training_result.get('success'):
                st.session_state.model = model
                st.session_state.training_result = training_result
                st.success("Model trained successfully!")
            else:
                st.error(f"Training failed: {training_result.get('error')}")
        except Exception as e:
            st.error(f"Error during training: {e}")
    
    if "model" in st.session_state:
        model = st.session_state.model
        training_result = st.session_state.training_result
        
        try:
            # Generate forecast - FIXED: removed seq_length parameter
            forecast = model.forecast(data, forecast_steps)
            
            # Validate forecast
            validation_result = model.validate_forecast(forecast, data)
            
            # Display training metrics
            st.subheader("Training Metrics")
            if model_type == "LSTM":
                st.write(f"Final Training RMSE: {training_result['final_train_rmse']:.2f}")
                st.write(f"Final Validation RMSE: {training_result['final_val_rmse']:.2f}")
            else:
                st.write(f"Final Training MAE: {training_result['mae'][-1]:.2f}")
                st.write(f"Final Validation MAE: {training_result['val_mae'][-1]:.2f}")
            
            # Display forecast metrics
            st.subheader("Forecast Validation")
            st.write(f"Forecast Length: {validation_result['forecast_length']}")
            st.write(f"Forecast Mean: {validation_result['forecast_mean']:.2f}")
            st.write(f"Forecast Std: {validation_result['forecast_std']:.2f}")
            st.write(f"Data Mean: {validation_result['data_mean']:.2f}")
            st.write(f"Data Std: {validation_result['data_std']:.2f}")
            
            if validation_result['is_reasonable']:
                st.success("Forecast is reasonable!")
            else:
                st.warning("Forecast may be unreasonable. Warnings: " + ", ".join(validation_result['warnings']))
            
            # Visualize forecast
            st.subheader("Forecast Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(data)), data, label="Historical Data")
            ax.plot(range(len(data), len(data) + forecast_steps), forecast, label="Forecast", color="red")
            ax.set_title(f"{selected_dataset.replace('_', ' ').title()} Forecast")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)
            
            # Option to download forecast
            csv = pd.DataFrame({'Forecast': forecast})
            st.download_button(
                label="Download Forecast as CSV",
                data=csv.to_csv(index=False),
                file_name=f"{selected_dataset}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.error("This might be due to model compatibility issues. Please retrain the model.")

# Footer
st.sidebar.text(f"App last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PKT")
