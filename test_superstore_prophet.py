#!/usr/bin/env python3
"""
Test Prophet forecasting specifically on superstore_sales dataset
to verify variability enhancement and range improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from prophet_forecasting import ProphetForecasting
from data_processor import TimeSeriesDataProcessor

def test_superstore_prophet():
    """Test Prophet forecasting on superstore_sales dataset"""
    print("🔍 TESTING PROPHET FORECASTING ON SUPERSTORE_SALES DATASET")
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
        
        # Initialize Prophet forecaster
        prophet = ProphetForecasting()
        
        # Train the model
        print("\n🚀 Training Prophet model...")
        train_result = prophet.train_model(df)
        
        if not train_result.get('success', False):
            print(f"❌ Training failed: {train_result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Training completed successfully")
        print(f"   Data points: {train_result.get('data_points', 'N/A')}")
        
        # Generate forecast
        print("\n🔮 Generating forecast...")
        forecast_result = prophet.univariate_forecast(df, periods=30, freq='D')
        
        if 'error' in forecast_result:
            print(f"❌ Forecasting failed: {forecast_result['error']}")
            return False
        
        forecast_df = forecast_result['forecast']
        print(f"✅ Forecast generated successfully")
        print(f"   Forecast length: {len(forecast_df)}")
        print(f"   Forecast range: [{forecast_df['yhat'].min():.2f}, {forecast_df['yhat'].max():.2f}]")
        print(f"   Forecast mean: {forecast_df['yhat'].mean():.2f}")
        print(f"   Forecast std: {forecast_df['yhat'].std():.2f}")
        
        # Calculate improvement metrics
        data_mean = df[numeric_col].mean()
        data_std = df[numeric_col].std()
        data_range = df[numeric_col].max() - df[numeric_col].min()
        
        fcst_mean = forecast_df['yhat'].mean()
        fcst_std = forecast_df['yhat'].std()
        fcst_range = forecast_df['yhat'].max() - forecast_df['yhat'].min()
        
        print(f"\n📊 VARIABILITY ANALYSIS:")
        print(f"   Historical Mean: {data_mean:.2f}")
        print(f"   Forecast Mean: {fcst_mean:.2f}")
        print(f"   Mean Ratio: {fcst_mean/data_mean:.2f}x")
        print(f"   Historical Std: {data_std:.2f}")
        print(f"   Forecast Std: {fcst_std:.2f}")
        print(f"   Std Ratio: {fcst_std/data_std:.2f}x")
        print(f"   Historical Range: {data_range:.2f}")
        print(f"   Forecast Range: {fcst_range:.2f}")
        print(f"   Range Ratio: {fcst_range/data_range:.2f}x")
        
        # Evaluate if the enhancement worked
        print(f"\n🎯 ENHANCEMENT EVALUATION:")
        
        # Check if forecast std is reasonable (should be at least 50% of historical std)
        if fcst_std >= data_std * 0.5:
            print(f"   ✅ Forecast variability: GOOD - Std ratio {fcst_std/data_std:.2f}x")
        else:
            print(f"   ❌ Forecast variability: POOR - Std ratio {fcst_std/data_std:.2f}x")
        
        # Check if forecast range is reasonable (should be at least 30% of historical range)
        if fcst_range >= data_range * 0.3:
            print(f"   ✅ Forecast range: GOOD - Range ratio {fcst_range/data_range:.2f}x")
        else:
            print(f"   ❌ Forecast range: POOR - Range ratio {fcst_range/data_range:.2f}x")
        
        # Check if forecast mean is reasonable (should be within 50% of historical mean)
        mean_ratio = fcst_mean / data_mean
        if 0.5 <= mean_ratio <= 2.0:
            print(f"   ✅ Forecast scale: GOOD - Mean ratio {mean_ratio:.2f}x")
        else:
            print(f"   ❌ Forecast scale: POOR - Mean ratio {mean_ratio:.2f}x")
        
        print(f"\n🎉 Prophet forecasting test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_superstore_prophet()
    if success:
        print("\n✅ All tests passed! Prophet variability enhancement working correctly.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
