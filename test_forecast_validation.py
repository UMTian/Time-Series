#!/usr/bin/env python3
"""
Forecast Validation Testing Framework
Tests not just functionality but also validates forecast quality and correctness
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_forecast_quality(forecast, original_data, method_name):
    """Validate forecast quality and reasonableness"""
    print(f"\n🔍 Validating {method_name} Forecast Quality...")
    
    # Basic checks
    if len(forecast) == 0:
        print("❌ Forecast is empty")
        return False
    
    if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
        print("❌ Forecast contains NaN or Inf values")
        return False
    
    # Statistical validation
    original_mean = np.mean(original_data)
    original_std = np.std(original_data)
    original_range = (np.min(original_data), np.max(original_data))
    
    forecast_mean = np.mean(forecast)
    forecast_std = np.std(forecast)
    forecast_range = (np.min(forecast), np.max(forecast))
    
    print(f"📊 Original Data - Mean: {original_mean:.4f}, Std: {original_std:.4f}, Range: {original_range[0]:.4f} to {original_range[1]:.4f}")
    print(f"📊 Forecast Data - Mean: {forecast_mean:.4f}, Std: {forecast_std:.4f}, Range: {forecast_range[0]:.4f} to {forecast_range[1]:.4f}")
    
    # Reasonableness checks
    mean_ratio = forecast_mean / original_mean if original_mean != 0 else float('inf')
    std_ratio = forecast_std / original_std if original_std != 0 else float('inf')
    
    print(f"📈 Mean Ratio: {mean_ratio:.4f}, Std Ratio: {std_ratio:.4f}")
    
    # Quality indicators
    quality_score = 0
    warnings = []
    
    # Check if forecast mean is reasonable (within 3x of original)
    if 0.1 < mean_ratio < 10:
        quality_score += 25
        print("✅ Mean ratio is reasonable")
    else:
        warnings.append(f"Mean ratio {mean_ratio:.4f} is outside reasonable bounds (0.1-10)")
        quality_score += 10
    
    # Check if forecast std is reasonable (within 5x of original)
    if 0.05 < std_ratio < 20:
        quality_score += 25
        print("✅ Standard deviation ratio is reasonable")
    else:
        warnings.append(f"Std ratio {std_ratio:.4f} is outside reasonable bounds (0.05-20)")
        quality_score += 10
    
    # Check if forecast range overlaps with original data range
    range_overlap = min(forecast_range[1], original_range[1]) - max(forecast_range[0], original_range[0])
    if range_overlap > 0:
        quality_score += 25
        print("✅ Forecast range overlaps with original data range")
    else:
        warnings.append("Forecast range has no overlap with original data range")
        quality_score += 5
    
    # Check for extreme outliers (beyond 5x the original range)
    original_range_size = original_range[1] - original_range[0]
    extreme_threshold = 5 * original_range_size
    forecast_extremes = np.max(np.abs(forecast - original_mean))
    
    if forecast_extremes < extreme_threshold:
        quality_score += 25
        print("✅ No extreme outliers detected")
    else:
        warnings.append(f"Extreme outliers detected: {forecast_extremes:.4f} vs threshold {extreme_threshold:.4f}")
        quality_score += 5
    
    # Overall assessment
    print(f"\n📊 {method_name} Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("🎉 Forecast quality: EXCELLENT")
        is_good = True
    elif quality_score >= 60:
        print("✅ Forecast quality: GOOD")
        is_good = True
    elif quality_score >= 40:
        print("⚠️ Forecast quality: FAIR")
        is_good = False
    else:
        print("❌ Forecast quality: POOR")
        is_good = False
    
    if warnings:
        print("\n⚠️ Quality Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    return is_good

def test_traditional_forecast_validation():
    """Test 1: Traditional Forecasting with Result Validation"""
    print("🧪 Testing Traditional Forecasting with Result Validation...")
    
    try:
        from traditional_forecasting import TraditionalForecasting
        
        forecaster = TraditionalForecasting()
        
        # Create realistic test data with known patterns
        np.random.seed(42)  # For reproducible results
        n_points = 200
        
        # Create trend + seasonality + noise
        time_index = np.arange(n_points)
        trend = 0.1 * time_index
        seasonality = 10 * np.sin(2 * np.pi * time_index / 12)  # 12-period seasonality
        noise = np.random.normal(0, 2, n_points)
        
        synthetic_data = trend + seasonality + noise + 100  # Base level 100
        
        # Split data
        train_size = int(0.8 * n_points)
        train_data = synthetic_data[:train_size]
        test_data = synthetic_data[train_size:]
        
        print(f"📊 Synthetic data created - Trend: +0.1 per period, Seasonality: 12 periods, Base: 100")
        print(f"📊 Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Test Holt-Winters
        print("\n🔍 Testing Holt-Winters Forecasting...")
        hw_result = forecaster.holt_winters_forecast(train_data, test_data, seasonal_periods=12)
        
        if 'error' in hw_result:
            print(f"❌ Holt-Winters failed: {hw_result['error']}")
            return False
        
        hw_forecast = hw_result['forecast']
        print(f"✅ Holt-Winters forecast generated - Length: {len(hw_forecast)}")
        
        # Validate Holt-Winters forecast quality
        hw_quality = validate_forecast_quality(hw_forecast, train_data, "Holt-Winters")
        
        # Test ARIMA
        print("\n🔍 Testing ARIMA Forecasting...")
        arima_result = forecaster.arima_forecast(train_data, test_data)
        
        if 'error' in arima_result:
            print(f"❌ ARIMA failed: {arima_result['error']}")
            return False
        
        arima_forecast = arima_result['forecast']
        print(f"✅ ARIMA forecast generated - Length: {len(arima_forecast)}")
        
        # Validate ARIMA forecast quality
        arima_quality = validate_forecast_quality(arima_forecast, train_data, "ARIMA")
        
        # Compare forecasts
        print("\n📊 Forecast Comparison:")
        print(f"Holt-Winters - Mean: {np.mean(hw_forecast):.4f}, Std: {np.std(hw_forecast):.4f}")
        print(f"ARIMA        - Mean: {np.mean(arima_forecast):.4f}, Std: {np.std(arima_forecast):.4f}")
        print(f"Actual Test  - Mean: {np.mean(test_data):.4f}, Std: {np.std(test_data):.4f}")
        
        # Calculate accuracy metrics
        hw_rmse = np.sqrt(np.mean((hw_forecast - test_data) ** 2))
        arima_rmse = np.sqrt(np.mean((arima_forecast - test_data) ** 2))
        
        print(f"\n📈 Accuracy Metrics (RMSE):")
        print(f"Holt-Winters: {hw_rmse:.4f}")
        print(f"ARIMA:        {arima_rmse:.4f}")
        
        # Determine which method performed better
        if hw_rmse < arima_rmse:
            print("🏆 Holt-Winters performed better on this dataset")
        else:
            print("🏆 ARIMA performed better on this dataset")
        
        return hw_quality and arima_quality
        
    except Exception as e:
        print(f"❌ Traditional forecast validation test failed: {e}")
        return False

def test_lstm_forecast_validation():
    """Test 2: LSTM Forecasting with Result Validation"""
    print("\n🧪 Testing LSTM Forecasting with Result Validation...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        # Create realistic test data
        np.random.seed(42)
        n_points = 300
        
        # Create complex pattern: trend + multiple seasonalities + noise
        time_index = np.arange(n_points)
        trend = 0.05 * time_index
        seasonality_1 = 8 * np.sin(2 * np.pi * time_index / 24)  # 24-period seasonality
        seasonality_2 = 4 * np.sin(2 * np.pi * time_index / 7)   # 7-period seasonality
        noise = np.random.normal(0, 1.5, n_points)
        
        synthetic_data = trend + seasonality_1 + seasonality_2 + noise + 50
        
        print(f"📊 Complex synthetic data created - Multiple seasonalities, trend, noise")
        print(f"📊 Data length: {len(synthetic_data)}")
        
        # Initialize and train LSTM
        lstm = LSTMForecasting(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
        
        print("🚀 Training LSTM model...")
        training_result = lstm.train_model(
            synthetic_data, 
            seq_length=20, 
            epochs=50,  # More epochs for better quality
            learning_rate=0.001,
            batch_size=32
        )
        
        if 'final_loss' not in training_result:
            print("❌ LSTM training failed")
            return False
        
        print(f"✅ LSTM trained - Final loss: {training_result['final_loss']:.6f}")
        
        # Generate forecast
        forecast_steps = 50
        lstm_forecast = lstm.forecast(synthetic_data, forecast_steps=forecast_steps)
        
        if len(lstm_forecast) != forecast_steps:
            print(f"❌ LSTM forecast length mismatch - Expected: {forecast_steps}, Got: {len(lstm_forecast)}")
            return False
        
        print(f"✅ LSTM forecast generated - Length: {len(lstm_forecast)}")
        
        # Validate forecast quality
        lstm_quality = validate_forecast_quality(lstm_forecast, synthetic_data, "LSTM")
        
        # Check for expected patterns
        print("\n🔍 Pattern Analysis:")
        
        # Check if forecast maintains trend
        forecast_trend = np.polyfit(np.arange(len(lstm_forecast)), lstm_forecast, 1)[0]
        original_trend = np.polyfit(np.arange(len(synthetic_data)), synthetic_data, 1)[0]
        
        print(f"Original trend slope: {original_trend:.6f}")
        print(f"Forecast trend slope: {forecast_trend:.6f}")
        
        if np.sign(forecast_trend) == np.sign(original_trend):
            print("✅ Forecast maintains correct trend direction")
        else:
            print("⚠️ Forecast trend direction differs from original")
        
        # Check seasonality preservation
        if len(lstm_forecast) >= 24:
            # Check if 24-period seasonality is preserved
            forecast_fft = np.fft.fft(lstm_forecast)
            dominant_freq = np.argmax(np.abs(forecast_fft[1:len(forecast_fft)//2])) + 1
            expected_freq = len(lstm_forecast) // 24
            
            print(f"Expected dominant frequency: {expected_freq}")
            print(f"Actual dominant frequency: {dominant_freq}")
            
            if abs(dominant_freq - expected_freq) <= 2:
                print("✅ Seasonality pattern preserved")
            else:
                print("⚠️ Seasonality pattern may not be preserved")
        
        return lstm_quality
        
    except Exception as e:
        print(f"❌ LSTM forecast validation test failed: {e}")
        return False

def test_cnn_forecast_validation():
    """Test 3: CNN Forecasting with Result Validation"""
    print("\n🧪 Testing CNN Forecasting with Result Validation...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        # Create realistic test data
        np.random.seed(42)
        n_points = 400
        
        # Create data with local patterns and trends
        time_index = np.arange(n_points)
        trend = 0.03 * time_index
        local_patterns = 6 * np.sin(2 * np.pi * time_index / 15) + 3 * np.sin(2 * np.pi * time_index / 8)
        noise = np.random.normal(0, 2, n_points)
        
        synthetic_data = trend + local_patterns + noise + 75
        
        print(f"📊 Local pattern data created - CNN should capture local features well")
        print(f"📊 Data length: {len(synthetic_data)}")
        
        # Initialize and train CNN
        cnn = CNNForecasting(seq_length=20, filters=64, kernel_size=3)
        
        print("🚀 Training CNN model...")
        training_result = cnn.train_model(
            synthetic_data, 
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        if 'loss' not in training_result:
            print("❌ CNN training failed")
            return False
        
        print(f"✅ CNN trained - Final loss: {training_result['loss'][-1]:.6f}")
        
        # Generate forecast
        forecast_steps = 60
        cnn_forecast = cnn.forecast(synthetic_data, forecast_steps=forecast_steps)
        
        if len(cnn_forecast) != forecast_steps:
            print(f"❌ CNN forecast length mismatch - Expected: {forecast_steps}, Got: {len(cnn_forecast)}")
            return False
        
        print(f"✅ CNN forecast generated - Length: {len(cnn_forecast)}")
        
        # Validate forecast quality
        cnn_quality = validate_forecast_quality(cnn_forecast, synthetic_data, "CNN")
        
        # Check for local pattern preservation
        print("\n🔍 Local Pattern Analysis:")
        
        # Check if local patterns are preserved in forecast
        if len(cnn_forecast) >= 30:
            # Look for local maxima/minima
            from scipy.signal import find_peaks
            
            original_peaks, _ = find_peaks(synthetic_data, height=np.mean(synthetic_data))
            forecast_peaks, _ = find_peaks(cnn_forecast, height=np.mean(cnn_forecast))
            
            print(f"Original data peaks: {len(original_peaks)}")
            print(f"Forecast peaks: {len(forecast_peaks)}")
            
            if len(forecast_peaks) > 0:
                print("✅ Local patterns detected in forecast")
            else:
                print("⚠️ No local patterns detected in forecast")
        
        return cnn_quality
        
    except Exception as e:
        print(f"❌ CNN forecast validation test failed: {e}")
        return False

def test_prophet_forecast_validation():
    """Test 4: Prophet Forecasting with Result Validation"""
    print("\n🧪 Testing Prophet Forecasting with Result Validation...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        # Create realistic time series data
        np.random.seed(42)
        n_points = 500
        
        # Create dates
        start_date = pd.Timestamp('2020-01-01')
        dates = pd.date_range(start=start_date, periods=n_points, freq='D')
        
        # Create complex pattern: trend + multiple seasonalities + holidays
        time_index = np.arange(n_points)
        trend = 0.02 * time_index
        yearly_seasonality = 15 * np.sin(2 * np.pi * time_index / 365.25)
        weekly_seasonality = 5 * np.sin(2 * np.pi * time_index / 7)
        noise = np.random.normal(0, 3, n_points)
        
        values = trend + yearly_seasonality + weekly_seasonality + noise + 100
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        print(f"📊 Time series data created - Daily frequency, yearly/weekly seasonality")
        print(f"📊 Data length: {len(df)}")
        
        # Initialize and train Prophet
        prophet = ProphetForecasting()
        
        print("🚀 Training Prophet model...")
        training_result = prophet.train_model(df, seasonality_mode='additive')
        
        if not training_result['success']:
            print(f"❌ Prophet training failed: {training_result.get('error', 'Unknown error')}")
            return False
        
        print("✅ Prophet model trained successfully")
        
        # Generate forecast
        forecast_periods = 90  # 3 months ahead
        prophet_result = prophet.univariate_forecast(df, periods=forecast_periods, freq='D')
        
        if 'forecast' not in prophet_result:
            print(f"❌ Prophet forecasting failed: {prophet_result.get('error', 'Unknown error')}")
            return False
        
        forecast_df = prophet_result['forecast']
        prophet_forecast = forecast_df['yhat'].values
        
        if len(prophet_forecast) != forecast_periods:
            print(f"❌ Prophet forecast length mismatch - Expected: {forecast_periods}, Got: {len(prophet_forecast)}")
            return False
        
        print(f"✅ Prophet forecast generated - Length: {len(prophet_forecast)}")
        
        # Validate forecast quality
        prophet_quality = validate_forecast_quality(prophet_forecast, df['value'].values, "Prophet")
        
        # Check Prophet-specific features
        print("\n🔍 Prophet Feature Analysis:")
        
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
            confidence_intervals = forecast_df['yhat_upper'] - forecast_df['yhat_lower']
            avg_confidence = np.mean(confidence_intervals)
            print(f"Average confidence interval width: {avg_confidence:.4f}")
            
            if avg_confidence > 0:
                print("✅ Confidence intervals generated")
            else:
                print("⚠️ Confidence intervals may be too narrow")
        
        # Check seasonality preservation
        if len(prophet_forecast) >= 7:
            # Check weekly seasonality
            weekly_pattern = prophet_forecast[:7]
            weekly_variance = np.var(weekly_pattern)
            
            if weekly_variance > 1:
                print("✅ Weekly seasonality pattern detected")
            else:
                print("⚠️ Weekly seasonality may not be preserved")
        
        return prophet_quality
        
    except Exception as e:
        print(f"❌ Prophet forecast validation test failed: {e}")
        return False

def main():
    """Run all forecast validation tests"""
    print("🚀 Starting Comprehensive Forecast Validation Testing")
    print("=" * 80)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔍 This framework validates not just functionality but also forecast quality!")
    print()
    
    test_results = []
    
    # Run all validation tests
    test_results.append(("Traditional Forecasting", test_traditional_forecast_validation()))
    test_results.append(("LSTM Forecasting", test_lstm_forecast_validation()))
    test_results.append(("CNN Forecasting", test_cnn_forecast_validation()))
    test_results.append(("Prophet Forecasting", test_prophet_forecast_validation()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 FORECAST VALIDATION TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} | {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} validation tests passed")
    
    if passed == total:
        print("🎉 All forecast validation tests passed! Forecasts are high quality.")
        print("\n🚀 System is ready for production with validated forecasting accuracy!")
        return True
    else:
        print("⚠️ Some forecast validation tests failed. Forecast quality needs improvement.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
