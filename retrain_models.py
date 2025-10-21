#!/usr/bin/env python3
"""
Comprehensive Model Retraining Script
Retrains all models with consistent parameters to ensure synchronization
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

from data_processor import TimeSeriesDataProcessor
from deep_learning_forecasting import LSTMForecasting, CNNForecasting
from prophet_forecasting import ProphetForecasting
from traditional_forecasting import TraditionalForecasting

class ModelRetrainer:
    """Comprehensive model retraining with consistent parameters"""
    
    def __init__(self):
        self.data_processor = TimeSeriesDataProcessor()
        self.lstm_forecaster = LSTMForecasting()
        self.cnn_forecaster = CNNForecasting()
        self.prophet_forecaster = ProphetForecasting()
        self.traditional_forecaster = TraditionalForecasting()
        
        # Consistent training parameters
        self.lstm_params = {
            'seq_length': 20,
            'epochs': 150,
            'learning_rate': 0.001,
            'validation_split': 0.2
        }
        
        self.cnn_params = {
            'seq_length': 20,
            'epochs': 200,
            'batch_size': 32,
            'validation_split': 0.2
        }
        
        self.datasets = ['airline_passengers', 'female_births', 'restaurant_visitors', 'superstore_sales']
    
    def retrain_lstm(self, dataset_name: str, train_data: np.ndarray) -> dict:
        """Retrain LSTM model with consistent parameters"""
        print(f"🧠 Retraining LSTM for {dataset_name}...")
        try:
            # Create new LSTM instance for each dataset
            lstm = LSTMForecasting(
                input_size=1,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            )
            
            # Train with consistent parameters
            result = lstm.train_model(
                train_data=train_data,
                seq_length=self.lstm_params['seq_length'],
                epochs=self.lstm_params['epochs'],
                learning_rate=self.lstm_params['learning_rate'],
                validation_split=self.lstm_params['validation_split']
            )
            
            if 'error' not in result:
                # Save model
                model_path = f"models/lstm_{dataset_name}.pth"
                lstm.save_model(model_path)
                print(f"✅ LSTM retrained and saved: {model_path}")
                return {'success': True, 'model_path': model_path}
            else:
                print(f"❌ LSTM training failed: {result['error']}")
                return {'error': result['error']}
                
        except Exception as e:
            print(f"❌ LSTM retraining error: {e}")
            return {'error': str(e)}
    
    def retrain_cnn(self, dataset_name: str, train_data: np.ndarray) -> dict:
        """Retrain CNN model with consistent parameters"""
        print(f"🔄 Retraining CNN for {dataset_name}...")
        try:
            # Create new CNN instance for each dataset
            cnn = CNNForecasting(seq_length=self.cnn_params['seq_length'])
            
            # Train with consistent parameters
            result = cnn.train_model(
                train_data=train_data,
                epochs=self.cnn_params['epochs'],
                batch_size=self.cnn_params['batch_size'],
                validation_split=self.cnn_params['validation_split']
            )
            
            if 'error' not in result:
                # Save model
                model_path = f"models/cnn_{dataset_name}.keras"
                cnn.save_model(model_path)
                print(f"✅ CNN retrained and saved: {model_path}")
                return {'success': True, 'model_path': model_path}
            else:
                print(f"❌ CNN training failed: {result['error']}")
                return {'error': result['error']}
                
        except Exception as e:
            print(f"❌ CNN retraining error: {e}")
            return {'error': str(e)}
    
    def retrain_prophet(self, dataset_name: str, df: pd.DataFrame) -> dict:
        """Retrain Prophet model with consistent parameters"""
        print(f"🎯 Retraining Prophet for {dataset_name}...")
        try:
            # Create new Prophet instance for each dataset
            prophet = ProphetForecasting()
            
            # Train with consistent parameters
            result = prophet.train_model(df)
            
            if result.get('success', False):
                # Save model
                model_path = f"models/prophet_{dataset_name}.json"
                prophet.save_model(model_path)
                print(f"✅ Prophet retrained and saved: {model_path}")
                return {'success': True, 'model_path': model_path}
            else:
                print(f"❌ Prophet training failed: {result.get('error', 'Unknown error')}")
                return {'error': result.get('error', 'Unknown error')}
                
        except Exception as e:
            print(f"❌ Prophet retraining error: {e}")
            return {'error': str(e)}
    
    def retrain_all_models(self):
        """Retrain all models for all datasets"""
        print("🚀 Starting comprehensive model retraining...")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        for dataset_name in self.datasets:
            print(f"\n{'='*60}")
            print(f"🔄 RETRAINING MODELS FOR: {dataset_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # Load dataset
                df = self.data_processor.load_dataset(dataset_name)
                y, train_data, test_data = self.data_processor.prepare_data_for_forecasting(df)
                
                print(f"📈 Dataset loaded: {len(y)} total points, {len(train_data)} train, {len(test_data)} test")
                
                dataset_results = {}
                
                # Retrain LSTM
                lstm_result = self.retrain_lstm(dataset_name, train_data)
                dataset_results['LSTM'] = lstm_result
                
                # Retrain CNN
                cnn_result = self.retrain_cnn(dataset_name, train_data)
                dataset_results['CNN'] = cnn_result
                
                # Retrain Prophet
                prophet_result = self.retrain_prophet(dataset_name, df)
                dataset_results['Prophet'] = prophet_result
                
                results[dataset_name] = dataset_results
                
                print(f"✅ Completed retraining for {dataset_name}")
                
            except Exception as e:
                print(f"❌ Failed to retrain models for {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        # Save retraining results
        self.save_retraining_results(results)
        
        print(f"\n🎉 Retraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return results
    
    def save_retraining_results(self, results: dict):
        """Save retraining results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"retraining_results_{timestamp}.txt"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE MODEL RETRAINING RESULTS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Retraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for dataset_name, dataset_results in results.items():
                    f.write(f"DATASET: {dataset_name.upper()}\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'error' in dataset_results:
                        f.write(f"Error: {dataset_results['error']}\n")
                    else:
                        for model_name, model_result in dataset_results.items():
                            if 'error' in model_result:
                                f.write(f"{model_name}: ❌ {model_result['error']}\n")
                            else:
                                f.write(f"{model_name}: ✅ Successfully retrained\n")
                    
                    f.write("\n")
                
                f.write("TRAINING PARAMETERS USED:\n")
                f.write("-" * 30 + "\n")
                f.write(f"LSTM: {self.lstm_params}\n")
                f.write(f"CNN: {self.cnn_params}\n")
            
            print(f"📝 Retraining results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save retraining results: {e}")

def main():
    """Main function"""
    print("🔄 COMPREHENSIVE MODEL RETRAINING")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Created models directory")
    
    # Initialize retrainer
    retrainer = ModelRetrainer()
    
    # Start retraining
    results = retrainer.retrain_all_models()
    
    # Print summary
    print("\n📊 RETRAINING SUMMARY")
    print("=" * 30)
    
    success_count = 0
    total_count = 0
    
    for dataset_name, dataset_results in results.items():
        if 'error' not in dataset_results:
            for model_name, model_result in dataset_results.items():
                total_count += 1
                if 'error' not in model_result:
                    success_count += 1
    
    print(f"✅ Successfully retrained: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("🎉 All models retrained successfully!")
    else:
        print("⚠️ Some models failed to retrain. Check the results above.")

if __name__ == "__main__":
    main()
