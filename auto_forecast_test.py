#!/usr/bin/env python3
"""
Automatic Forecast Testing Script
Tests the forecasting system comprehensively to ensure quality results
"""

import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from streamlit_app import TimeSeriesForecastingApp
from data_processor import TimeSeriesDataProcessor

def test_data_loading():
    """Test data loading and preprocessing"""
    print("🔍 Testing Data Loading...")
    
    try:
        # Test data processor
        data_processor = TimeSeriesDataProcessor()
        
        # Test superstore_sales dataset
        df = data_processor.load_dataset('superstore_sales')
        print(f"✅ Data loaded successfully: {df.shape}")
        
        # Check for required columns
        required_cols = ['Sales', 'Total Amount', 'Quantity']
        available_cols = [col for col in required_cols if col in df.columns]
        print(f"✅ Available columns: {available_cols}")
        
        # Check data quality
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"✅ Numeric columns: {numeric_cols}")
        
        # Check for missing values
        missing_data = df.isnull().sum().sum()
        print(f"✅ Missing values: {missing_data}")
        
        return df, data_processor
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None

def test_model_loading():
    """Test model loading and validation"""
    print("\n🧠 Testing Model Loading...")
    
    try:
        app = TimeSeriesForecastingApp()
        
        if hasattr(app, 'optimized_models') and app.optimized_models:
            print(f"✅ {len(app.optimized_models)} models loaded successfully")
            
            # Check model quality scores
            for name, info in app.optimized_models.items():
                quality = info['metrics']['quality_score']
                print(f"   • {name}: Quality Score {quality:.2f}")
            
            return app
        else:
            print("❌ No models loaded")
            return None
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_forecast_generation(app, df, target_col='Sales'):
    """Test forecast generation with detailed validation"""
    print(f"\n📈 Testing Forecast Generation for {target_col}...")
    
    try:
        # Prepare data
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found")
            return None
            
        y = df[target_col].values
        
        # Basic data statistics
        print(f"✅ Data Statistics:")
        print(f"   • Records: {len(y)}")
        print(f"   • Range: {y.min():.2f} to {y.max():.2f}")
        print(f"   • Mean: {y.mean():.2f}")
        print(f"   • Std: {y.std():.2f}")
        
        # Test each model
        results = {}
        
        for model_name, model_info in app.optimized_models.items():
            print(f"\n🔍 Testing {model_name}...")
            
            try:
                # Get model and scaler
                model = model_info['model']
                scaler = model_info['scaler']
                
                # Prepare sequence data
                sequence_length = 60
                if len(y) < sequence_length:
                    print(f"   ⚠️ Data too short for sequence length {sequence_length}")
                    continue
                
                # Create test sequence
                test_sequence = y[-sequence_length:]
                test_sequence_scaled = scaler.transform(test_sequence.reshape(-1, 1))
                test_sequence_tensor = torch.FloatTensor(test_sequence_scaled).unsqueeze(0)
                
                # Test model forward pass
                with torch.no_grad():
                    test_output = model(test_sequence_tensor)
                    
                    # Ensure scalar output
                    if test_output.numel() > 1:
                        test_output = torch.mean(test_output)
                    
                    test_output_scalar = test_output.item()
                
                print(f"   ✅ Model forward pass successful: {test_output_scalar:.4f}")
                
                # Generate forecast
                forecast_length = 30
                forecast = app._generate_optimized_forecast(
                    model, scaler, y, forecast_length, sequence_length
                )
                
                if forecast is not None and len(forecast) > 0:
                    # Convert to numpy array if needed
                    if isinstance(forecast, list):
                        forecast = np.array(forecast)
                    
                    # Calculate forecast metrics
                    forecast_std = np.std(forecast)
                    forecast_range = forecast.max() - forecast.min()
                    forecast_mean = np.mean(forecast)
                    
                    # Calculate variation ratio
                    data_std = np.std(y)
                    variation_ratio = forecast_std / data_std if data_std > 0 else 0
                    
                    print(f"   ✅ Forecast generated successfully:")
                    print(f"      • Length: {len(forecast)}")
                    print(f"      • Range: {forecast.min():.2f} to {forecast.max():.2f}")
                    print(f"      • Mean: {forecast_mean:.2f}")
                    print(f"      • Std: {forecast_std:.2f}")
                    print(f"      • Variation Ratio: {variation_ratio:.4f}")
                    
                    # Validate forecast bounds
                    data_min, data_max = y.min(), y.max()
                    data_range = data_max - data_min
                    
                    # Check if forecast is within reasonable bounds
                    max_allowed_range = data_range * 2.0  # Allow 2x data range
                    forecast_min_allowed = max(0, data_min * 0.1)
                    forecast_max_allowed = data_max + max_allowed_range
                    
                    # Count out-of-bounds values
                    out_of_bounds = np.sum((forecast < forecast_min_allowed) | (forecast > forecast_max_allowed))
                    
                    if out_of_bounds == 0:
                        print(f"      ✅ All forecast values within reasonable bounds")
                    else:
                        print(f"      ⚠️ {out_of_bounds} values outside reasonable bounds")
                    
                    # Check variation quality
                    if variation_ratio < 0.1:
                        print(f"      ⚠️ Low variation detected (ratio: {variation_ratio:.4f})")
                    elif variation_ratio > 3.0:
                        print(f"      ⚠️ High variation detected (ratio: {variation_ratio:.4f})")
                    else:
                        print(f"      ✅ Good variation level (ratio: {variation_ratio:.4f})")
                    
                    # Store results
                    results[model_name] = {
                        'forecast': forecast,
                        'std': forecast_std,
                        'range': forecast_range,
                        'mean': forecast_mean,
                        'variation_ratio': variation_ratio,
                        'out_of_bounds': out_of_bounds
                    }
                    
                else:
                    print(f"   ❌ Forecast generation failed")
                    
            except Exception as e:
                print(f"   ❌ Model test failed: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"❌ Forecast testing failed: {e}")
        return None

def test_forecast_validation(app, df, results):
    """Test forecast validation and bounds checking"""
    print(f"\n🔍 Testing Forecast Validation...")
    
    try:
        if not results:
            print("❌ No forecast results to validate")
            return
            
        for model_name, result in results.items():
            print(f"\n🔍 Validating {model_name}...")
            
            forecast = result['forecast']
            y = df['Sales'].values if 'Sales' in df.columns else df.iloc[:, -1].values
            
            # Test bounds validation
            validated_forecast = app._validate_forecast_bounds(forecast, y)
            
            if validated_forecast is not None:
                print(f"   ✅ Bounds validation successful")
                
                # Check if validation made changes
                if np.array_equal(forecast, validated_forecast):
                    print(f"   ✅ No clipping needed - forecast within bounds")
                else:
                    print(f"   ⚠️ Forecast clipped to reasonable bounds")
                    
                    # Show before/after statistics
                    print(f"      • Original range: {forecast.min():.2f} to {forecast.max():.2f}")
                    print(f"      • Validated range: {validated_forecast.min():.2f} to {validated_forecast.max():.2f}")
            else:
                print(f"   ❌ Bounds validation failed")
                
    except Exception as e:
        print(f"❌ Forecast validation testing failed: {e}")

def run_comprehensive_test():
    """Run comprehensive forecast testing"""
    print("🚀 Starting Comprehensive Forecast Testing")
    print("=" * 60)
    
    # Test 1: Data Loading
    df, data_processor = test_data_loading()
    if df is None:
        print("❌ Cannot proceed without data")
        return
    
    # Test 2: Model Loading
    app = test_model_loading()
    if app is None:
        print("❌ Cannot proceed without models")
        return
    
    # Test 3: Forecast Generation
    results = test_forecast_generation(app, df, 'Sales')
    if results is None:
        print("❌ Cannot proceed without forecast results")
        return
    
    # Test 4: Forecast Validation
    test_forecast_validation(app, df, results)
    
    # Summary Report
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Data: {df.shape[0]} records loaded successfully")
    print(f"✅ Models: {len(app.optimized_models)} models tested")
    print(f"✅ Forecasts: {len(results)} models generated forecasts")
    
    # Quality assessment
    good_models = 0
    for model_name, result in results.items():
        if (result['variation_ratio'] >= 0.1 and 
            result['variation_ratio'] <= 3.0 and 
            result['out_of_bounds'] == 0):
            good_models += 1
    
    print(f"✅ Quality: {good_models}/{len(results)} models producing good forecasts")
    
    if good_models == len(results):
        print("🎉 All models are working correctly!")
    elif good_models > 0:
        print("⚠️ Some models need attention")
    else:
        print("❌ All models need attention")
    
    print("\n🚀 Testing completed successfully!")

if __name__ == "__main__":
    run_comprehensive_test()
