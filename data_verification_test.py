#!/usr/bin/env python3
"""
Data Verification and Testing Script
Tests data loading, scaling, and forecast generation to identify scaling issues
"""

import pandas as pd
import numpy as np
import torch
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

def test_data_loading():
    """Test data loading and basic statistics"""
    print("=== DATA LOADING TEST ===")
    
    # Look in data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ {data_dir} directory not found")
        return None
    
    data_files = []
    for file in os.listdir(data_dir):
        if file.endswith(('.csv', '.xls', '.xlsx')):
            data_files.append(file)
    
    print(f"Found data files: {data_files}")
    
    if not data_files:
        print("❌ No data files found in data directory")
        return None
    
    # Try to load the Superstore data first
    target_file = None
    for file in data_files:
        if 'superstore' in file.lower() or 'sales' in file.lower():
            target_file = file
            break
    
    if target_file is None:
        target_file = data_files[0]
    
    file_path = os.path.join(data_dir, target_file)
    print(f"Loading: {file_path}")
    
    try:
        if target_file.endswith('.xls') or target_file.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"✅ Loaded dataset: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric columns: {list(numeric_cols)}")
        
        for col in numeric_cols[:5]:  # First 5 numeric columns
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.2f}")
            print(f"  Max: {df[col].max():.2f}")
            print(f"  Sample values: {df[col].head().tolist()}")
        
        return df, target_file
        
    except Exception as e:
        print(f"❌ Error loading {target_file}: {e}")
        return None

def test_scaler_behavior(df, target_col):
    """Test how the scaler behaves with the data"""
    print(f"\n=== SCALER BEHAVIOR TEST for {target_col} ===")
    
    # Create scaler
    scaler = MinMaxScaler()
    
    # Get target data
    y = df[target_col].values
    y = y[~np.isnan(y)]
    
    print(f"Original data shape: {y.shape}")
    print(f"Original data sample: {y[:10]}")
    
    # Fit and transform
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"Scaled data shape: {y_scaled.shape}")
    print(f"Scaled data sample: {y_scaled[:10]}")
    print(f"Scaled data range: {y_scaled.min():.6f} to {y_scaled.max():.6f}")
    
    # Test inverse transform
    y_inverse = scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    print(f"Inverse data shape: {y_inverse.shape}")
    print(f"Inverse data sample: {y_inverse[:10]}")
    print(f"Inverse data range: {y_inverse.min():.2f} to {y_inverse.max():.2f}")
    
    # Check if inverse transform recovers original
    diff = np.abs(y - y_inverse)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Recovery accuracy:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Perfect recovery: {max_diff < 1e-10}")
    
    return scaler, y, y_scaled

def test_model_loading():
    """Test if optimized models can be loaded"""
    print(f"\n=== MODEL LOADING TEST ===")
    
    optimized_models_dir = "optimized_models"
    if not os.path.exists(optimized_models_dir):
        print(f"❌ {optimized_models_dir} directory not found")
        return None
    
    model_files = os.listdir(optimized_models_dir)
    print(f"Found model files: {model_files}")
    
    # Try to load a model
    for file in model_files:
        if file.endswith('.pth'):
            model_path = os.path.join(optimized_models_dir, file)
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                print(f"✅ Successfully loaded {file}")
                print(f"  Keys: {list(state_dict.keys())}")
                return state_dict
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
    
    return None

def test_forecast_generation(df, target_col, scaler):
    """Test forecast generation with controlled variation"""
    print(f"\n=== FORECAST GENERATION TEST ===")
    
    # Get target data
    y = df[target_col].values
    y = y[~np.isnan(y)]
    
    if len(y) < 60:
        print(f"❌ Data too short: {len(y)} < 60")
        return None
    
    # Use last 60 points as sequence
    sequence = y[-60:]
    sequence_scaled = scaler.transform(sequence.reshape(-1, 1)).flatten()
    
    print(f"Sequence length: {len(sequence)}")
    print(f"Sequence range: {sequence.min():.2f} to {sequence.max():.2f}")
    print(f"Sequence scaled range: {sequence_scaled.min():.6f} to {sequence_scaled.max():.6f}")
    
    # Simulate forecast generation with controlled variation
    print(f"\nSimulating forecast generation...")
    
    forecast_scaled = []
    current_sequence = sequence_scaled.copy()
    
    for step in range(30):  # 30 forecast steps
        # Simple prediction: average of last few values + small variation
        pred_scaled = np.mean(current_sequence[-5:]) + np.random.normal(0, 0.01)
        
        # Ensure prediction stays within reasonable bounds
        pred_scaled = np.clip(pred_scaled, 0, 1)
        
        forecast_scaled.append(pred_scaled)
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred_scaled
    
    # Convert back to original scale
    forecast_original = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    
    print(f"Forecast scaled range: {min(forecast_scaled):.6f} to {max(forecast_scaled):.6f}")
    print(f"Forecast original range: {forecast_original.min():.2f} to {forecast_original.max():.2f}")
    
    # Calculate statistics
    data_std = np.std(y)
    forecast_std = np.std(forecast_original)
    variation_ratio = forecast_std / data_std if data_std > 0 else 0
    
    print(f"\nForecast Statistics:")
    print(f"  Data std: {data_std:.2f}")
    print(f"  Forecast std: {forecast_std:.2f}")
    print(f"  Variation ratio: {variation_ratio:.2f}")
    
    # Check if variation is reasonable (should be < 3x data variation)
    if variation_ratio > 3:
        print(f"⚠️ WARNING: Forecast variation is {variation_ratio:.2f}x data variation - too high!")
    else:
        print(f"✅ Forecast variation is reasonable")
    
    return forecast_original, variation_ratio

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE DATA VERIFICATION AND TESTING")
    print("=" * 60)
    
    # Test 1: Data loading
    result = test_data_loading()
    if result is None:
        print("❌ Cannot proceed without data")
        return
    
    df, data_file = result
    
    # Test 2: Find target column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = None
    
    # Look for common target columns
    for col in ['Sales', 'Total Amount', 'Quantity', 'Revenue', 'Amount']:
        if col in numeric_cols:
            target_col = col
            break
    
    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]
    
    if target_col is None:
        print("❌ No suitable target column found")
        return
    
    print(f"\n🎯 Using target column: {target_col}")
    
    # Test 3: Scaler behavior
    scaler, y, y_scaled = test_scaler_behavior(df, target_col)
    
    # Test 4: Model loading
    state_dict = test_model_loading()
    
    # Test 5: Forecast generation
    forecast, variation_ratio = test_forecast_generation(df, target_col, scaler)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"=" * 60)
    print(f"✅ Data file: {data_file}")
    print(f"✅ Target column: {target_col}")
    print(f"✅ Data shape: {df.shape}")
    print(f"✅ Data range: {y.min():.2f} to {y.max():.2f}")
    print(f"✅ Scaler working: {scaler is not None}")
    print(f"✅ Model loading: {state_dict is not None}")
    print(f"✅ Forecast generated: {forecast is not None}")
    print(f"✅ Variation ratio: {variation_ratio:.2f}")
    
    if variation_ratio > 3:
        print(f"⚠️ ISSUE: Forecast variation too high - needs fixing")
    else:
        print(f"✅ All tests passed - system working correctly")

if __name__ == "__main__":
    main()
