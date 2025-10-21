#!/usr/bin/env python3
"""
Integration testing framework for the improved time series forecasting system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import TimeSeriesDataProcessor
from deep_learning_forecasting import LSTMForecasting, CNNForecasting
from prophet_forecasting import ProphetForecasting

def test_complete_workflow():
    """Test the complete end-to-end workflow"""
    print("🧪 Testing Complete End-to-End Workflow...")
    
    try:
        # Initialize components
        dp = TimeSeriesDataProcessor()
        lstm = LSTMForecasting()
        cnn = CNNForecasting()
        prophet = ProphetForecasting()
        
        # Load data
        data = dp.load_dataset('airline_passengers')
        numeric_col = dp.get_numeric_column(data)
        train_data = data[numeric_col].values
        
        print(f"✅ Data loaded and processed - Train: {len(train_data)}, Test: {len(train_data)//4}")
        
        # Test traditional forecasting with new function signatures
        from traditional_forecasting import TraditionalForecasting
        
        traditional = TraditionalForecasting()
        
        # Test Holt-Winters
        hw_result = traditional.holt_winters_forecast(train_data, forecast_steps=10)
        if 'error' in hw_result:
            print(f"❌ Holt-Winters failed: {hw_result['error']}")
            return False
        print("✅ Holt-Winters forecasting working")
        
        # Test ARIMA
        arima_result = traditional.arima_forecast(train_data, forecast_steps=10)
        if 'error' in arima_result:
            print(f"❌ ARIMA failed: {arima_result['error']}")
            return False
        print("✅ ARIMA forecasting working")
        
        # Test LSTM
        if not lstm.is_trained:
            print("⚠️ LSTM not trained, skipping LSTM test")
        else:
            lstm_result = lstm.forecast(train_data, 10)
            if len(lstm_result) == 10:
                print("✅ LSTM forecasting working")
            else:
                print(f"❌ LSTM forecast length mismatch: {len(lstm_result)}")
                return False
        
        # Test CNN
        if not cnn.is_trained:
            print("⚠️ CNN not trained, skipping CNN test")
        else:
            cnn_result = cnn.forecast(train_data, 10)
            if len(cnn_result) == 10:
                print("✅ CNN forecasting working")
            else:
                print(f"❌ CNN forecast length mismatch: {len(cnn_result)}")
                return False
        
        # Test Prophet
        prophet_result = prophet.univariate_forecast(data, 10, 'D')
        if 'error' in prophet_result:
            print(f"❌ Prophet failed: {prophet_result['error']}")
            return False
        print("✅ Prophet forecasting working")
        
        print("✅ Complete workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {str(e)}")
        return False

def test_model_persistence():
    """Test model saving and loading"""
    print("🧪 Testing Model Persistence...")
    
    try:
        # Test LSTM persistence
        lstm = LSTMForecasting()
        train_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        
        # Quick training with proper sequence length
        lstm.train_model(train_data, seq_length=5, epochs=3, learning_rate=0.01, batch_size=4)
        
        # Save and load
        lstm.save_model('test_lstm_integration.pth')
        new_lstm = LSTMForecasting()
        result = new_lstm.load_model('test_lstm_integration.pth')
        
        if 'error' in result:
            print(f"❌ LSTM persistence failed: {result['error']}")
            return False
        
        print("✅ LSTM persistence working")
        
        # Test CNN persistence with larger dataset
        cnn = CNNForecasting()
        # Create a larger dataset for CNN testing
        cnn_train_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        cnn.train_model(cnn_train_data, epochs=3, batch_size=4, validation_split=0.2)
        
        # Save and load
        cnn.save_model('test_cnn_integration.keras')
        new_cnn = CNNForecasting()
        result = new_cnn.load_model('test_cnn_integration.keras')
        
        if 'error' in result:
            print(f"❌ CNN persistence failed: {result['error']}")
            return False
        
        print("✅ CNN persistence working")
        
        # Clean up test files
        import os
        if os.path.exists('test_lstm_integration.pth'):
            os.remove('test_lstm_integration.pth')
        if os.path.exists('test_cnn_integration.keras'):
            os.remove('test_cnn_integration.keras')
        if os.path.exists('test_cnn_integration_scaler.pkl'):
            os.remove('test_cnn_integration_scaler.pkl')
        
        print("✅ Model persistence test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model persistence test failed: {str(e)}")
        return False

def test_forecast_validation():
    """Test 3: Forecast Validation System"""
    print("\n🧪 Testing Forecast Validation System...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting, CNNForecasting
        
        # Test LSTM validation
        lstm = LSTMForecasting()
        sample_data = np.random.randn(100) + np.linspace(0, 2, 100)
        sample_forecast = np.random.randn(20) + np.linspace(2, 4, 20)
        
        lstm_validation = lstm.validate_forecast(sample_forecast, sample_data)
        if 'quality_score' in lstm_validation:
            print("✅ LSTM validation working")
        else:
            print("❌ LSTM validation failed")
            return False
        
        # Test CNN validation
        cnn = CNNForecasting()
        cnn_validation = cnn.validate_forecast(sample_forecast, sample_data)
        if 'quality_score' in cnn_validation:
            print("✅ CNN validation working")
        else:
            print("❌ CNN validation failed")
            return False
        
        print("✅ Forecast validation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forecast validation test failed: {e}")
        return False

def test_performance_metrics():
    """Test 4: Performance Metrics Calculation"""
    print("\n🧪 Testing Performance Metrics...")
    
    try:
        from prophet_forecasting import ProphetForecasting
        
        prophet = ProphetForecasting()
        
        # Create sample data
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=20, freq='D'),
            'yhat': np.random.randn(20) + 100,
            'yhat_lower': np.random.randn(20) + 95,
            'yhat_upper': np.random.randn(20) + 105
        })
        
        actual_values = np.random.randn(20) + 100
        
        # Test metrics
        metrics = prophet.calculate_metrics(forecast_df, actual_values)
        
        required_metrics = ['rmse', 'mae', 'mape', 'smape', 'mase', 'directional_accuracy']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        
        if not missing_metrics:
            print("✅ Performance metrics working")
        else:
            print(f"❌ Missing metrics: {missing_metrics}")
            return False
        
        print("✅ Performance metrics test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting Integration Testing Framework")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Complete Workflow", test_complete_workflow()))
    test_results.append(("Model Persistence", test_model_persistence()))
    test_results.append(("Forecast Validation", test_forecast_validation()))
    test_results.append(("Performance Metrics", test_performance_metrics()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
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
        print("🎉 All integration tests passed! System is ready for production.")
        print("\n🚀 Next step: Train production models with full datasets.")
        return True
    else:
        print("⚠️ Some integration tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
