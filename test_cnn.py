#!/usr/bin/env python3
"""
CNN Testing Framework
Tests the advanced CNN implementation step by step
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

def test_cnn_basic_functionality():
    """Test 1: Basic CNN Functionality"""
    print("🧪 Testing CNN Basic Functionality...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        # Test initialization
        cnn = CNNForecasting(seq_length=20, filters=64, kernel_size=3)
        print("✅ CNN initialized successfully")
        
        # Test model building
        model = cnn.build_model()
        print("✅ CNN model built successfully")
        
        # Test sequence creation
        sample_data = np.random.randn(100)
        X, y = cnn.create_sequences(sample_data)
        print(f"✅ Sequences created - X: {X.shape}, y: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CNN basic functionality test failed: {e}")
        return False

def test_cnn_data_processing():
    """Test 2: CNN Data Processing (Normalization)"""
    print("\n🧪 Testing CNN Data Processing...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        cnn = CNNForecasting()
        
        # Test data normalization
        sample_data = np.random.randn(100) * 1000 + 5000  # Large scale data
        normalized = cnn.normalize_data(sample_data)
        
        # Check normalization range
        if np.all(normalized >= -1) and np.all(normalized <= 1):
            print("✅ Data normalization working correctly")
        else:
            print("❌ Data normalization failed - values outside [-1, 1] range")
            return False
        
        # Test denormalization
        denormalized = cnn.denormalize_data(normalized)
        
        # Check if denormalization is approximately correct
        if np.allclose(sample_data, denormalized, rtol=1e-5):
            print("✅ Data denormalization working correctly")
        else:
            print("❌ Data denormalization failed - values don't match original")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CNN data processing test failed: {e}")
        return False

def test_cnn_training():
    """Test 3: CNN Training Process"""
    print("\n🧪 Testing CNN Training Process...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        cnn = CNNForecasting(seq_length=20, filters=32, kernel_size=3)
        
        # Create synthetic training data
        sample_data = np.random.randn(200) + np.linspace(0, 5, 200)  # Trend + noise
        
        # Test training
        training_result = cnn.train_model(
            sample_data, 
            epochs=5,  # Short training for testing
            batch_size=16,
            validation_split=0.2
        )
        
        if 'loss' in training_result and 'val_loss' in training_result:
            print("✅ CNN training completed successfully")
            print(f"   Final loss: {training_result['loss'][-1]:.6f}")
            print(f"   Final val_loss: {training_result['val_loss'][-1]:.6f}")
        else:
            print("❌ CNN training failed - missing loss information")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CNN training test failed: {e}")
        return False

def test_cnn_forecasting():
    """Test 4: CNN Forecasting"""
    print("\n🧪 Testing CNN Forecasting...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        cnn = CNNForecasting(seq_length=20, filters=32, kernel_size=3)
        
        # Create and train model
        sample_data = np.random.randn(150) + np.linspace(0, 3, 150)
        cnn.train_model(sample_data, epochs=5, batch_size=16, validation_split=0.2)
        
        # Test forecasting
        forecast = cnn.forecast(sample_data, forecast_steps=20)
        
        if len(forecast) == 20 and not np.any(np.isnan(forecast)):
            print("✅ CNN forecasting working correctly")
            print(f"   Forecast length: {len(forecast)}")
            print(f"   Forecast range: {forecast.min():.2f} to {forecast.max():.2f}")
        else:
            print("❌ CNN forecasting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CNN forecasting test failed: {e}")
        return False

def test_cnn_validation():
    """Test 5: CNN Forecast Validation"""
    print("\n🧪 Testing CNN Forecast Validation...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        cnn = CNNForecasting()
        
        # Create sample data and forecast
        sample_data = np.random.randn(100) + np.linspace(0, 2, 100)
        sample_forecast = np.random.randn(20) + np.linspace(2, 4, 20)
        
        # Test validation
        validation = cnn.validate_forecast(sample_forecast, sample_data)
        
        if 'quality_score' in validation and 'is_reasonable' in validation:
            print("✅ CNN validation working correctly")
            print(f"   Quality score: {validation['quality_score']:.1f}")
            print(f"   Is reasonable: {validation['is_reasonable']}")
        else:
            print("❌ CNN validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CNN validation test failed: {e}")
        return False

def test_cnn_save_load():
    """Test 6: CNN Model Save/Load"""
    print("\n🧪 Testing CNN Model Save/Load...")
    
    try:
        from deep_learning_forecasting import CNNForecasting
        
        cnn = CNNForecasting(seq_length=20, filters=32, kernel_size=3)
        
        # Train model
        sample_data = np.random.randn(100) + np.linspace(0, 1, 100)
        cnn.train_model(sample_data, epochs=3, batch_size=16, validation_split=0.2)
        
        # Save model
        save_path = "test_cnn_model.keras"
        cnn.save_model(save_path)
        print("✅ CNN model saved successfully")
        
        # Create new instance and load model
        new_cnn = CNNForecasting()
        new_cnn.load_model(save_path)
        print("✅ CNN model loaded successfully")
        
        # Test if loaded model works
        if new_cnn.is_trained:
            print("✅ Loaded model is properly trained")
        else:
            print("❌ Loaded model is not properly trained")
            return False
        
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)
            print("✅ Test file cleaned up")
        
        # Clean up scaler file
        scaler_path = save_path.replace('.keras', '_scaler.pkl')
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            print("✅ Scaler file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ CNN save/load test failed: {e}")
        return False

def main():
    """Run all CNN tests"""
    print("🚀 Starting CNN Testing Framework")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Basic Functionality", test_cnn_basic_functionality()))
    test_results.append(("Data Processing", test_cnn_data_processing()))
    test_results.append(("Training Process", test_cnn_training()))
    test_results.append(("Forecasting", test_cnn_forecasting()))
    test_results.append(("Validation", test_cnn_validation()))
    test_results.append(("Save/Load", test_cnn_save_load()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CNN TEST SUMMARY")
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
        print("🎉 All CNN tests passed! Ready for Prophet testing.")
        return True
    else:
        print("⚠️ Some CNN tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
