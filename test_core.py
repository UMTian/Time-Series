#!/usr/bin/env python3
"""
Core Testing Framework for Time Series Forecasting
Tests each component systematically to ensure bug-free operation
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

def test_data_processor():
    """Test 1: Data Processor Functionality"""
    print("🧪 Testing Data Processor...")
    
    try:
        from data_processor import TimeSeriesDataProcessor
        
        # Initialize processor
        processor = TimeSeriesDataProcessor()
        print("✅ Data processor initialized successfully")
        
        # Test dataset loading
        test_datasets = ['airline_passengers', 'female_births', 'restaurant_visitors', 'superstore_sales']
        
        for dataset in test_datasets:
            try:
                df = processor.load_dataset(dataset)
                print(f"✅ Dataset '{dataset}' loaded successfully - Shape: {df.shape}")
                
                # Test numeric column detection
                numeric_col = processor.get_numeric_column(df)
                print(f"✅ Numeric column detected: {numeric_col}")
                
                # Test data preparation
                y, train_data, test_data = processor.prepare_data_for_forecasting(df)
                print(f"✅ Data prepared - Train: {len(train_data)}, Test: {len(test_data)}")
                
            except Exception as e:
                print(f"❌ Dataset '{dataset}' failed: {e}")
                return False
        
        print("✅ Data processor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def test_traditional_forecasting():
    """Test 2: Traditional Forecasting Methods"""
    print("\n🧪 Testing Traditional Forecasting...")
    
    try:
        from traditional_forecasting import TraditionalForecasting
        
        # Initialize forecaster
        forecaster = TraditionalForecasting()
        print("✅ Traditional forecaster initialized successfully")
        
        # Test with sample data
        sample_data = np.random.randn(100) + np.linspace(0, 10, 100)  # Trend + noise
        train_data = sample_data[:80]
        test_data = sample_data[80:]
        
        # Test Holt-Winters
        try:
            hw_result = forecaster.holt_winters_forecast(train_data, test_data, seasonal_periods=12)
            if 'error' not in hw_result:
                print("✅ Holt-Winters forecasting working")
            else:
                print(f"❌ Holt-Winters error: {hw_result['error']}")
                return False
        except Exception as e:
            print(f"❌ Holt-Winters test failed: {e}")
            return False
        
        # Test ARIMA
        try:
            arima_result = forecaster.arima_forecast(train_data, test_data)
            if 'error' not in arima_result:
                print("✅ ARIMA forecasting working")
            else:
                print(f"❌ ARIMA error: {arima_result['error']}")
                return False
        except Exception as e:
            print(f"❌ ARIMA test failed: {e}")
            return False
        
        print("✅ Traditional forecasting tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Traditional forecasting test failed: {e}")
        return False

def test_basic_imports():
    """Test 0: Basic Import Functionality"""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test core libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ Core libraries imported successfully")
        
        # Test ML libraries
        try:
            import torch
            print("✅ PyTorch imported successfully")
        except ImportError:
            print("⚠️ PyTorch not available")
        
        try:
            import tensorflow as tf
            print("✅ TensorFlow imported successfully")
        except ImportError:
            print("⚠️ TensorFlow not available")
        
        try:
            from prophet import Prophet
            print("✅ Prophet imported successfully")
        except ImportError:
            print("⚠️ Prophet not available")
        
        print("✅ Basic import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic import test failed: {e}")
        return False

def main():
    """Run all tests systematically"""
    print("🚀 Starting Systematic Testing Framework")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Test 0: Basic imports
    test_results.append(("Basic Imports", test_basic_imports()))
    
    # Test 1: Data processor
    test_results.append(("Data Processor", test_data_processor()))
    
    # Test 2: Traditional forecasting
    test_results.append(("Traditional Forecasting", test_traditional_forecasting()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
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
        print("🎉 All core tests passed! Ready for next phase.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
