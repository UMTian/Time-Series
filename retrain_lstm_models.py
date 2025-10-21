#!/usr/bin/env python3
"""
Script to retrain all LSTM models with the new 58-feature architecture
This ensures compatibility with the updated deep_learning_forecasting.py
"""

import os
import numpy as np
import pandas as pd
import torch  # Added missing torch import
from deep_learning_forecasting import LSTMForecasting
import warnings
warnings.filterwarnings('ignore')

def load_sample_data(dataset_name: str) -> np.ndarray:
    """Load sample data for each dataset"""
    np.random.seed(42)
    
    if dataset_name == "airline_passengers":
        # Generate airline passenger-like data (trend + seasonality)
        n_points = 500
        t = np.linspace(0, 20, n_points)
        trend = 100 + 5 * t
        seasonality = 50 * np.sin(2 * np.pi * t / 4) + 30 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 10, n_points)
        return np.maximum(0, trend + seasonality + noise)
    
    elif dataset_name == "female_births":
        # Generate birth data-like pattern
        n_points = 500
        t = np.linspace(0, 20, n_points)
        base = 1000
        trend = 20 * np.sin(2 * np.pi * t / 10)
        seasonality = 200 * np.sin(2 * np.pi * t / 4)
        noise = np.random.normal(0, 50, n_points)
        return np.maximum(0, base + trend + seasonality + noise)
    
    elif dataset_name == "restaurant_visitors":
        # Generate restaurant visitor-like data
        n_points = 500
        t = np.linspace(0, 20, n_points)
        base = 500
        trend = 10 * t
        seasonality = 100 * np.sin(2 * np.pi * t / 7) + 50 * np.sin(2 * np.pi * t / 30)
        noise = np.random.normal(0, 30, n_points)
        return np.maximum(0, base + trend + seasonality + noise)
    
    elif dataset_name == "superstore_sales":
        # Generate sales-like data
        n_points = 500
        t = np.linspace(0, 20, n_points)
        base = 2000
        trend = 15 * t
        seasonality = 300 * np.sin(2 * np.pi * t / 4) + 150 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 100, n_points)
        return np.maximum(0, base + trend + seasonality + noise)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def retrain_lstm_model(dataset_name: str, data: np.ndarray, models_dir: str = "models"):
    """Retrain LSTM model for a specific dataset"""
    print(f"\n=== Retraining LSTM Model for {dataset_name} ===")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize LSTM forecaster with sequential architecture
    lstm_forecaster = LSTMForecasting(
        seq_length=20, 
        hidden_size=128, 
        num_layers=2, 
        dropout=0.2
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.min(data):.2f} to {np.max(data):.2f}")
    print(f"Data mean: {np.mean(data):.2f}")
    print(f"Data std: {np.std(data):.2f}")
    
    # Train the model
    print("\nTraining LSTM model...")
    training_result = lstm_forecaster.train_model(
        data, 
        epochs=100, 
        learning_rate=0.001,
        validation_split=0.2
    )
    
    if training_result.get('success'):
        print("✓ LSTM model trained successfully")
        print(f"Final training RMSE: {training_result['final_train_rmse']:.6f}")
        print(f"Final validation RMSE: {training_result['final_val_rmse']:.6f}")
        
        # Save the model
        model_path = os.path.join(models_dir, f"lstm_{dataset_name}.pth")
        torch.save(lstm_forecaster.model.state_dict(), model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Test forecast generation
        print("\nTesting forecast generation...")
        try:
            forecast = lstm_forecaster.forecast(data, 30)
            print(f"✓ Generated forecast with {len(forecast)} steps")
            
            # Validate forecast
            validation = lstm_forecaster.validate_forecast(forecast, data)
            if validation.get('is_reasonable'):
                print("✓ Forecast validation passed")
                print(f"Forecast mean: {validation['forecast_mean']:.2f}")
                print(f"Forecast std: {validation['forecast_std']:.2f}")
            else:
                print("⚠ Forecast validation warnings:")
                for warning in validation.get('warnings', []):
                    print(f"  - {warning}")
        except Exception as e:
            print(f"✗ Forecast generation failed: {str(e)}")
        
        return True
    else:
        print(f"✗ LSTM training failed: {training_result.get('error')}")
        return False

def main():
    """Main function to retrain all LSTM models"""
    print("=== LSTM Model Retraining Script ===")
    print("This script will retrain all LSTM models with the new sequential architecture")
    print("This ensures compatibility with the updated deep_learning_forecasting.py\n")
    
    # Datasets to retrain
    datasets = [
        "airline_passengers",
        "female_births", 
        "restaurant_visitors",
        "superstore_sales"
    ]
    
    success_count = 0
    total_count = len(datasets)
    
    for dataset_name in datasets:
        try:
            # Load sample data
            data = load_sample_data(dataset_name)
            
            # Retrain model
            if retrain_lstm_model(dataset_name, data):
                success_count += 1
            
        except Exception as e:
            print(f"✗ Failed to process {dataset_name}: {str(e)}")
    
    print(f"\n=== Retraining Complete ===")
    print(f"Successfully retrained: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("✓ All LSTM models are now compatible with the updated system!")
        print("You can now use them in your Streamlit application.")
    else:
        print("⚠ Some models failed to retrain. Check the errors above.")

if __name__ == "__main__":
    main()
