#!/usr/bin/env python3
"""
Prophet Testing Framework
Tests the advanced Prophet implementation step by step
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prophet_basic_functionality():
    """Test 1: Basic Prophet Functionality"""
    print("🧪 Testing Prophet Basic Functionality...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        # Test initialization
        prophet = ProphetForecasting()
        print("✅ Prophet forecaster initialized successfully")
        
        # Test model creation
        model = prophet.create_prophet_model()
        print("✅ Prophet model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet basic functionality test failed: {e}")
        return False

def test_prophet_data_preparation():
    """Test 2: Prophet Data Preparation"""
    print("\n🧪 Testing Prophet Data Preparation...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        prophet = ProphetForecasting()
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values = np.random.randn(100) + np.linspace(0, 5, 100)  # Trend + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Test data preparation
        prepared_df = prophet.prepare_data(df, date_col='date', value_col='value')
        
        if 'ds' in prepared_df.columns and 'y' in prepared_df.columns:
            print("✅ Data preparation working correctly")
            print(f"   Prepared data shape: {prepared_df.shape}")
        else:
            print("❌ Data preparation failed - missing required columns")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet data preparation test failed: {e}")
        return False

def test_prophet_training():
    """Test 3: Prophet Training Process"""
    print("\n🧪 Testing Prophet Training Process...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        prophet = ProphetForecasting()
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        values = np.random.randn(200) + np.linspace(0, 3, 200)  # Trend + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Test training
        training_result = prophet.train_model(df, seasonality_mode='additive')
        
        if training_result['success']:
            print("✅ Prophet training completed successfully")
            print(f"   Data points: {training_result['data_points']}")
        else:
            print(f"❌ Prophet training failed: {training_result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet training test failed: {e}")
        return False

def test_prophet_forecasting():
    """Test 4: Prophet Forecasting"""
    print("\n🧪 Testing Prophet Forecasting...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        prophet = ProphetForecasting()
        
        # Create and train model
        dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
        values = np.random.randn(150) + np.linspace(0, 2, 150)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Train model
        prophet.train_model(df, seasonality_mode='additive')
        
        # Test forecasting
        forecast_result = prophet.univariate_forecast(df, periods=30, freq='D')
        
        if 'forecast' in forecast_result:
            forecast_df = forecast_result['forecast']
            if len(forecast_df) == 30:
                print("✅ Prophet forecasting working correctly")
                print(f"   Forecast length: {len(forecast_df)}")
                print(f"   Forecast range: {forecast_df['yhat'].min():.2f} to {forecast_df['yhat'].max():.2f}")
            else:
                print(f"❌ Prophet forecasting failed - expected 30 periods, got {len(forecast_df)}")
                return False
        else:
            print(f"❌ Prophet forecasting failed: {forecast_result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet forecasting test failed: {e}")
        return False

def test_prophet_metrics():
    """Test 5: Prophet Metrics Calculation"""
    print("\n🧪 Testing Prophet Metrics Calculation...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        prophet = ProphetForecasting()
        
        # Create sample forecast and actual data
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=20, freq='D'),
            'yhat': np.random.randn(20) + 100,
            'yhat_lower': np.random.randn(20) + 95,
            'yhat_upper': np.random.randn(20) + 105
        })
        
        actual_values = np.random.randn(20) + 100
        
        # Test metrics calculation
        metrics = prophet.calculate_metrics(forecast_df, actual_values)
        
        if 'error' not in metrics:
            print("✅ Prophet metrics calculation working correctly")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   MAPE: {metrics['mape']:.2f}%")
        else:
            print(f"❌ Prophet metrics calculation failed: {metrics['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet metrics test failed: {e}")
        return False

def test_prophet_model_summary():
    """Test 6: Prophet Model Summary"""
    print("\n🧪 Testing Prophet Model Summary...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        prophet = ProphetForecasting()
        
        # Create and train model
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        values = np.random.randn(100) + np.linspace(0, 1, 100)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Train model
        prophet.train_model(df, seasonality_mode='additive')
        
        # Test model summary
        summary = prophet.get_model_summary()
        
        if 'error' not in summary:
            print("✅ Prophet model summary working correctly")
            print(f"   Is trained: {summary['is_trained']}")
            print(f"   Seasonalities: {summary['seasonalities']}")
        else:
            print(f"❌ Prophet model summary failed: {summary['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet model summary test failed: {e}")
        return False

def main():
    """Run all Prophet tests"""
    print("🚀 Starting Prophet Testing Framework")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Basic Functionality", test_prophet_basic_functionality()))
    test_results.append(("Data Preparation", test_prophet_data_preparation()))
    test_results.append(("Training Process", test_prophet_training()))
    test_results.append(("Forecasting", test_prophet_forecasting()))
    test_results.append(("Metrics Calculation", test_prophet_metrics()))
    test_results.append(("Model Summary", test_prophet_model_summary()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PROPHET TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} | {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Prophet tests passed! Ready for integration testing.")
        return True
    else:
        print("⚠️ Some Prophet tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
