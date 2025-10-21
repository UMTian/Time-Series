#!/usr/bin/env python3
"""
Test script to verify that the MinMaxScaler fix works for all CSV datasets
"""

from deep_learning_forecasting import LSTMForecasting, CNNForecasting
from data_processor import TimeSeriesDataProcessor
import numpy as np

def test_dataset_scaling(dataset_name, model_type='lstm'):
    """Test scaling for a specific dataset"""
    print(f"\n=== Testing {dataset_name} with {model_type.upper()} ===")
    
    try:
        # Load dataset
        dp = TimeSeriesDataProcessor()
        data = dp.load_dataset(dataset_name)
        numeric_col = dp.get_numeric_column(data)
        train_data = data[numeric_col].values
        
        print(f"Data shape: {train_data.shape}")
        print(f"Data mean: {np.mean(train_data):.2f}")
        print(f"Data range: [{np.min(train_data):.2f}, {np.max(train_data):.2f}]")
        
        # Load model
        if model_type == 'lstm':
            forecaster = LSTMForecasting()
            model_path = f'models/lstm_{dataset_name}.pth'
        else:
            forecaster = CNNForecasting()
            model_path = f'models/cnn_{dataset_name}.keras'
        
        result = forecaster.load_model(model_path)
        if 'error' in result:
            print(f"❌ Failed to load model: {result['error']}")
            return False
        
        # Generate forecast
        forecast = forecaster.forecast(train_data, 5)
        print(f"Forecast values: {forecast[:3]}...")
        print(f"Forecast mean: {np.mean(forecast):.2f}")
        print(f"Forecast range: [{np.min(forecast):.2f}, {np.max(forecast):.2f}]")
        
        # Check if forecast is in reasonable range
        data_mean = np.mean(train_data)
        data_std = np.std(train_data)
        forecast_mean = np.mean(forecast)
        
        # Calculate scale ratio
        scale_ratio = forecast_mean / data_mean if data_mean != 0 else float('inf')
        
        if 0.1 <= scale_ratio <= 10.0:
            print(f"✅ Scale ratio: {scale_ratio:.2f} (REASONABLE)")
            return True
        else:
            print(f"❌ Scale ratio: {scale_ratio:.2f} (UNREASONABLE)")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {dataset_name}: {str(e)}")
        return False

def main():
    """Test all datasets with both LSTM and CNN"""
    datasets = ['airline_passengers', 'female_births', 'restaurant_visitors', 'superstore_sales']
    
    print("🔍 TESTING MINMAXSCALER FIX FOR ALL DATASETS")
    print("=" * 60)
    
    # Test LSTM models
    print("\n🧠 TESTING LSTM MODELS:")
    lstm_success = 0
    for dataset in datasets:
        if test_dataset_scaling(dataset, 'lstm'):
            lstm_success += 1
    
    # Test CNN models
    print("\n🖼️ TESTING CNN MODELS:")
    cnn_success = 0
    for dataset in datasets:
        if test_dataset_scaling(dataset, 'cnn'):
            cnn_success += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"LSTM Models: {lstm_success}/{len(datasets)} working correctly")
    print(f"CNN Models: {cnn_success}/{len(datasets)} working correctly")
    
    if lstm_success == len(datasets) and cnn_success == len(datasets):
        print("🎉 ALL MODELS WORKING CORRECTLY WITH PROPER SCALING!")
    else:
        print("⚠️ Some models still have scaling issues")

if __name__ == "__main__":
    main()
