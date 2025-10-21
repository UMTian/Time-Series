#!/usr/bin/env python3
"""
Advanced Time Series Model Training Script
Trains LSTM and CNN models with state-of-the-art architectures
"""

import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import TimeSeriesDataProcessor
from deep_learning_forecasting import LSTMForecasting, CNNForecasting

def train_lstm_model(dataset_name: str, save_dir: str = "models"):
    """Train LSTM model with advanced architecture"""
    print(f"Training LSTM model on {dataset_name}...")
    
    # Load and prepare data
    processor = TimeSeriesDataProcessor()
    df = processor.load_dataset(dataset_name)
    y, train_data, test_data = processor.prepare_data_for_forecasting(df)
    
    # Create and train LSTM model with dynamic input size
    lstm = LSTMForecasting(
        input_size=1,  # Will be dynamically adjusted during training
        hidden_size=128,  # Increased from 50
        num_layers=2,     # Increased from 1
        dropout=0.2
    )
    
    # Train with advanced parameters
    training_result = lstm.train_model(
        train_data, 
        seq_length=20,    # Increased from 10
        epochs=200,        # Increased from 150
        learning_rate=0.001
    )
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"lstm_{dataset_name}.pth")
    lstm.save_model(model_path)
    
    if 'success' in training_result and training_result['success']:
        print(f"LSTM training completed. Final train RMSE: {training_result['final_train_rmse']:.6f}")
        print(f"Best validation RMSE: {training_result['best_val_loss']:.6f}")
    else:
        print(f"LSTM training failed: {training_result.get('error', 'Unknown error')}")
    return model_path

def train_cnn_model(dataset_name: str, save_dir: str = "models"):
    """Train CNN model with advanced architecture"""
    print(f"Training CNN model on {dataset_name}...")
    
    # Load and prepare data
    processor = TimeSeriesDataProcessor()
    df = processor.load_dataset(dataset_name)
    y, train_data, test_data = processor.prepare_data_for_forecasting(df)
    
    # Create and train CNN model
    cnn = CNNForecasting(
        seq_length=20,    # Increased from 10
        filters=128,       # Increased from 64
        kernel_size=3
    )
    
    # Train with advanced parameters
    training_result = cnn.train_model(
        train_data, 
        epochs=200,        # Increased from 100
        validation_split=0.2
    )
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"cnn_{dataset_name}.keras")
    cnn.save_model(model_path)
    
    if 'success' in training_result and training_result['success']:
        print(f"CNN training completed. Final train RMSE: {training_result['final_train_rmse']:.6f}")
        print(f"Best validation RMSE: {training_result['best_val_loss']:.6f}")
    else:
        print(f"CNN training failed: {training_result.get('error', 'Unknown error')}")
    return model_path

def main():
    """Main training function"""
    print("🚀 Starting advanced model training...")
    print("This will train LSTM and CNN models with state-of-the-art architectures.")
    print("Training may take several minutes...")
    print()
    
    start_time = time.time()
    
    # Datasets to train on
    datasets = ['airline_passengers', 'female_births', 'restaurant_visitors', 'superstore_sales']
    
    print("🚀 Starting model training for all datasets...")
    print("=" * 50)
    
    for dataset in datasets:
        print(f"\n📊 Processing dataset: {dataset}")
        
        try:
            # Train LSTM
            lstm_path = train_lstm_model(dataset)
            print(f"✅ LSTM model saved: {lstm_path}")
            
            # Train CNN
            cnn_path = train_cnn_model(dataset)
            print(f"✅ CNN model saved: {cnn_path}")
            
        except Exception as e:
            print(f"❌ Error training models for {dataset}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("🎯 Advanced model training completed!")
    print(f"⏱️  Total training time: {time.time() - start_time:.0f} seconds")
    print("Models are now ready for forecasting in the Streamlit app.")
    print("\n🔧 Key improvements implemented:")
    print("   • Attention mechanisms for better sequence understanding")
    print("   • Residual connections for deeper networks")
    print("   • Advanced normalization and scaling")
    print("   • Early stopping and learning rate scheduling")
    print("   • Gradient clipping and regularization")
    print("   • Quality scoring and validation systems")

if __name__ == "__main__":
    main()
