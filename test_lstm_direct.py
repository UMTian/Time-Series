#!/usr/bin/env python3
"""
Direct test of LSTM forecasting on superstore_sales dataset
to identify the exact issue causing flat forecasts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from deep_learning_forecasting import LSTMForecasting
from data_processor import TimeSeriesDataProcessor

def test_lstm_direct():
    """Test LSTM forecasting directly on superstore_sales dataset"""
    print("🔍 DIRECT TESTING OF LSTM FORECASTING ON SUPERSTORE_SALES")
    print("=" * 60)
    
    try:
        # Load the superstore_sales dataset
        processor = TimeSeriesDataProcessor()
        df = processor.load_dataset('superstore_sales')
        
        if df is None:
            print("❌ Failed to load superstore_sales dataset")
            return False
        
        print(f"📊 Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Get the numeric column for forecasting
        numeric_col = processor.get_numeric_column(df)
        if numeric_col is None:
            print("❌ No numeric column found")
            return False
        
        print(f"   Numeric column: {numeric_col}")
        print(f"   Data range: [{df[numeric_col].min():.2f}, {df[numeric_col].max():.2f}]")
        print(f"   Data mean: {df[numeric_col].mean():.2f}")
        print(f"   Data std: {df[numeric_col].std():.2f}")
        
        # Get the data values
        y = df[numeric_col].values
        print(f"   Data values shape: {y.shape}")
        print(f"   First 10 values: {y[:10]}")
        print(f"   Last 10 values: {y[-10:]}")
        
        # Initialize LSTM forecaster
        lstm = LSTMForecasting()
        
        # Load the pre-trained model
        print(f"\n🚀 Loading LSTM model...")
        model_path = "models/lstm_superstore_sales.pth"
        load_result = lstm.load_model(model_path)
        
        if 'error' in load_result:
            print(f"❌ Model loading failed: {load_result['error']}")
            return False
        
        print(f"✅ Model loaded successfully")
        print(f"   Model is trained: {lstm.is_trained}")
        print(f"   Model exists: {lstm.model is not None}")
        
        # Test forecasting
        print(f"\n🔮 Testing LSTM forecasting...")
        print(f"   Input data shape: {y.shape}")
        print(f"   Input data type: {type(y)}")
        print(f"   Input data sample: {y[:5]}")
        
        # Generate forecast
        forecast = lstm.forecast(y, 30)
        
        print(f"✅ Forecast generated successfully")
        print(f"   Forecast shape: {forecast.shape}")
        print(f"   Forecast type: {type(forecast)}")
        print(f"   Forecast values: {forecast}")
        print(f"   Forecast mean: {np.mean(forecast):.2f}")
        print(f"   Forecast std: {np.std(forecast):.2f}")
        print(f"   Forecast range: [{np.min(forecast):.2f}, {np.max(forecast):.2f}]")
        
        # Check if forecast is flat
        if np.std(forecast) < 1e-6:
            print(f"❌ FORECAST IS FLAT! Standard deviation: {np.std(forecast)}")
        else:
            print(f"✅ Forecast has variability: std = {np.std(forecast):.2f}")
        
        # Check if forecast values are reasonable
        data_mean = np.mean(y)
        data_std = np.std(y)
        forecast_mean = np.mean(forecast)
        
        print(f"\n📊 FORECAST ANALYSIS:")
        print(f"   Data Mean: {data_mean:.2f}")
        print(f"   Forecast Mean: {forecast_mean:.2f}")
        print(f"   Mean Ratio: {forecast_mean/data_mean:.2f}x")
        
        if 0.1 <= forecast_mean/data_mean <= 10.0:
            print(f"   ✅ Forecast scale: REASONABLE")
        else:
            print(f"   ❌ Forecast scale: UNREASONABLE")
        
        # Test seasonal decomposition
        print(f"\n📈 Testing Seasonal Decomposition...")
        decomposition = processor.seasonal_decompose(y, period=12)
        
        if 'error' in decomposition:
            print(f"❌ Seasonal decomposition failed: {decomposition['error']}")
        else:
            print(f"✅ Seasonal decomposition successful")
            print(f"   Trend shape: {decomposition['trend'].shape}")
            print(f"   Seasonal shape: {decomposition['seasonal'].shape}")
            print(f"   Residual shape: {decomposition['residual'].shape}")
            print(f"   Period: {decomposition['period']}")
            
            # Check if trend has variability
            trend_std = np.std(decomposition['trend'][~np.isnan(decomposition['trend'])])
            print(f"   Trend std: {trend_std:.2f}")
            
            if trend_std > 100:
                print(f"   ✅ Trend has significant variability")
            else:
                print(f"   ⚠️ Trend has low variability")
        
        print(f"\n🎉 Direct LSTM test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lstm_direct()
    if success:
        print("\n✅ Direct LSTM test completed. Check the results above.")
    else:
        print("\n❌ Direct LSTM test failed. Check the error details above.")
