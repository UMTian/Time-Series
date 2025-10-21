#!/usr/bin/env python3
"""
LSTM Testing Framework
Tests the advanced LSTM implementation step by step
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

def test_lstm_basic_functionality():
    """Test 1: Basic LSTM Functionality"""
    print("🧪 Testing LSTM Basic Functionality...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        # Test initialization
        lstm = LSTMForecasting(input_size=1, hidden_size=64, num_layers=1, dropout=0.1)
        print("✅ LSTM initialized successfully")
        
        # Test model building
        model = lstm.build_model()
        print("✅ LSTM model built successfully")
        
        # Test sequence creation
        sample_data = np.random.randn(100)
        X, y = lstm.create_sequences(sample_data, seq_length=10)
        print(f"✅ Sequences created - X: {X.shape}, y: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM basic functionality test failed: {e}")
        return False

def test_lstm_data_processing():
    """Test 2: LSTM Data Processing (Normalization)"""
    print("\n🧪 Testing LSTM Data Processing...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        lstm = LSTMForecasting()
        
        # Test data normalization
        sample_data = np.random.randn(100) * 1000 + 5000  # Large scale data
        normalized = lstm.normalize_data(sample_data)
        
        # Check normalization range
        if np.all(normalized >= -1) and np.all(normalized <= 1):
            print("✅ Data normalization working correctly")
        else:
            print("❌ Data normalization failed - values outside [-1, 1] range")
            return False
        
        # Test denormalization
        denormalized = lstm.denormalize_data(normalized)
        
        # Check if denormalization is approximately correct
        if np.allclose(sample_data, denormalized, rtol=1e-5):
            print("✅ Data denormalization working correctly")
        else:
            print("❌ Data denormalization failed - values don't match original")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM data processing test failed: {e}")
        return False

def test_lstm_training():
    """Test 3: LSTM Training Process"""
    print("\n🧪 Testing LSTM Training Process...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        lstm = LSTMForecasting(input_size=1, hidden_size=32, num_layers=1, dropout=0.1)
        
        # Create synthetic training data
        sample_data = np.random.randn(200) + np.linspace(0, 5, 200)  # Trend + noise
        
        # Test training
        training_result = lstm.train_model(
            sample_data, 
            seq_length=10, 
            epochs=5,  # Short training for testing
            learning_rate=0.01,
            batch_size=16
        )
        
        if 'final_loss' in training_result and 'best_loss' in training_result:
            print("✅ LSTM training completed successfully")
            print(f"   Final loss: {training_result['final_loss']:.6f}")
            print(f"   Best loss: {training_result['best_loss']:.6f}")
        else:
            print("❌ LSTM training failed - missing loss information")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM training test failed: {e}")
        return False

def test_lstm_forecasting():
    """Test 4: LSTM Forecasting"""
    print("\n🧪 Testing LSTM Forecasting...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        lstm = LSTMForecasting(input_size=1, hidden_size=32, num_layers=1, dropout=0.1)
        
        # Create and train model
        sample_data = np.random.randn(150) + np.linspace(0, 3, 150)
        lstm.train_model(sample_data, seq_length=10, epochs=5, learning_rate=0.01, batch_size=16)
        
        # Test forecasting
        forecast = lstm.forecast(sample_data, forecast_steps=20)
        
        if len(forecast) == 20 and not np.any(np.isnan(forecast)):
            print("✅ LSTM forecasting working correctly")
            print(f"   Forecast length: {len(forecast)}")
            print(f"   Forecast range: {forecast.min():.2f} to {forecast.max():.2f}")
        else:
            print("❌ LSTM forecasting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM forecasting test failed: {e}")
        return False

def test_lstm_validation():
    """Test 5: LSTM Forecast Validation"""
    print("\n🧪 Testing LSTM Forecast Validation...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        lstm = LSTMForecasting()
        
        # Create sample data and forecast
        sample_data = np.random.randn(100) + np.linspace(0, 2, 100)
        sample_forecast = np.random.randn(20) + np.linspace(2, 4, 20)
        
        # Test validation
        validation = lstm.validate_forecast(sample_forecast, sample_data)
        
        if 'quality_score' in validation and 'is_reasonable' in validation:
            print("✅ LSTM validation working correctly")
            print(f"   Quality score: {validation['quality_score']:.1f}")
            print(f"   Is reasonable: {validation['is_reasonable']}")
        else:
            print("❌ LSTM validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM validation test failed: {e}")
        return False

def test_lstm_save_load():
    """Test 6: LSTM Model Save/Load"""
    print("\n🧪 Testing LSTM Model Save/Load...")
    
    try:
        from deep_learning_forecasting import LSTMForecasting
        
        lstm = LSTMForecasting(input_size=1, hidden_size=32, num_layers=1, dropout=0.1)
        
        # Train model
        sample_data = np.random.randn(100) + np.linspace(0, 1, 100)
        lstm.train_model(sample_data, seq_length=10, epochs=3, learning_rate=0.01, batch_size=16)
        
        # Save model
        save_path = "test_lstm_model.pth"
        lstm.save_model(save_path)
        print("✅ LSTM model saved successfully")
        
        # Create new instance and load model
        new_lstm = LSTMForecasting()
        new_lstm.load_model(save_path)
        print("✅ LSTM model loaded successfully")
        
        # Test if loaded model works
        if new_lstm.is_trained:
            print("✅ Loaded model is properly trained")
        else:
            print("❌ Loaded model is not properly trained")
            return False
        
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)
            print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM save/load test failed: {e}")
        return False

def main():
    """Run all LSTM tests"""
    print("🚀 Starting LSTM Testing Framework")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Basic Functionality", test_lstm_basic_functionality()))
    test_results.append(("Data Processing", test_lstm_data_processing()))
    test_results.append(("Training Process", test_lstm_training()))
    test_results.append(("Forecasting", test_lstm_forecasting()))
    test_results.append(("Validation", test_lstm_validation()))
    test_results.append(("Save/Load", test_lstm_save_load()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 LSTM TEST SUMMARY")
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
        print("🎉 All LSTM tests passed! Ready for CNN testing.")
        return True
    else:
        print("⚠️ Some LSTM tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
