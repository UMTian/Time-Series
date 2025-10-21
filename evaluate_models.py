#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Tests all trained models and compares their performance across datasets
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

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self):
        self.data_processor = TimeSeriesDataProcessor()
        self.traditional_forecaster = TraditionalForecasting()
        self.lstm_forecaster = LSTMForecasting()
        self.cnn_forecaster = CNNForecasting()
        self.prophet_forecaster = ProphetForecasting()
        
        # Results storage
        self.results = {}
        
    def load_models(self, dataset_name: str):
        """Load all trained models for a specific dataset"""
        models_dir = "models"
        models_loaded = {}
        
        # Load LSTM model
        lstm_path = os.path.join(models_dir, f"lstm_{dataset_name}.pth")
        if os.path.exists(lstm_path):
            try:
                self.lstm_forecaster.load_model(lstm_path)
                models_loaded['LSTM'] = True
                print(f"✅ LSTM model loaded for {dataset_name}")
            except Exception as e:
                print(f"❌ Failed to load LSTM model: {e}")
                models_loaded['LSTM'] = False
        else:
            models_loaded['LSTM'] = False
        
        # Load CNN model
        cnn_path = os.path.join(models_dir, f"cnn_{dataset_name}.keras")
        if os.path.exists(cnn_path):
            try:
                self.cnn_forecaster.load_model(cnn_path)
                models_loaded['CNN'] = True
                print(f"✅ CNN model loaded for {dataset_name}")
            except Exception as e:
                print(f"❌ Failed to load CNN model: {e}")
                models_loaded['CNN'] = False
        else:
            models_loaded['CNN'] = False
        
        return models_loaded
    
    def evaluate_traditional_methods(self, data: np.ndarray, test_data: np.ndarray) -> dict:
        """Evaluate traditional forecasting methods"""
        print("🔍 Evaluating Traditional Methods...")
        results = {}
        
        # Holt-Winters
        try:
            hw_result = self.traditional_forecaster.holt_winters_forecast(data, forecast_steps=len(test_data))
            if 'error' not in hw_result:
                results['Holt-Winters'] = hw_result['metrics']
                print(f"✅ Holt-Winters: RMSE={hw_result['metrics']['rmse']:.4f}")
            else:
                results['Holt-Winters'] = {'error': hw_result['error']}
                print(f"❌ Holt-Winters failed: {hw_result['error']}")
        except Exception as e:
            results['Holt-Winters'] = {'error': str(e)}
            print(f"❌ Holt-Winters error: {e}")
        
        # ARIMA
        try:
            arima_result = self.traditional_forecaster.arima_forecast(data, forecast_steps=len(test_data))
            if 'error' not in arima_result:
                results['ARIMA'] = arima_result['metrics']
                print(f"✅ ARIMA: RMSE={arima_result['metrics']['rmse']:.4f}")
            else:
                results['ARIMA'] = {'error': arima_result['error']}
                print(f"❌ ARIMA failed: {arima_result['error']}")
        except Exception as e:
            results['ARIMA'] = {'error': str(e)}
            print(f"❌ ARIMA error: {e}")
        
        # Simple Moving Average
        try:
            sma_result = self.traditional_forecaster.simple_moving_average_forecast(data, forecast_steps=len(test_data))
            if 'error' not in sma_result:
                results['Simple Moving Average'] = sma_result['metrics']
                print(f"✅ Simple Moving Average: RMSE={sma_result['metrics']['rmse']:.4f}")
            else:
                results['Simple Moving Average'] = {'error': sma_result['error']}
                print(f"❌ Simple Moving Average failed: {sma_result['error']}")
        except Exception as e:
            results['Simple Moving Average'] = {'error': str(e)}
            print(f"❌ Simple Moving Average error: {e}")
        
        # Exponential Smoothing
        try:
            es_result = self.traditional_forecaster.exponential_smoothing_forecast(data, forecast_steps=len(test_data))
            if 'error' not in es_result:
                results['Exponential Smoothing'] = es_result['metrics']
                print(f"✅ Exponential Smoothing: RMSE={es_result['metrics']['rmse']:.4f}")
            else:
                results['Exponential Smoothing'] = {'error': es_result['error']}
                print(f"❌ Exponential Smoothing failed: {es_result['error']}")
        except Exception as e:
            results['Exponential Smoothing'] = {'error': str(e)}
            print(f"❌ Exponential Smoothing error: {e}")
        
        return results
    
    def evaluate_lstm(self, data: np.ndarray, test_data: np.ndarray) -> dict:
        """Evaluate LSTM model"""
        print("🧠 Evaluating LSTM Model...")
        try:
            if not self.lstm_forecaster.is_trained:
                return {'error': 'LSTM model not trained'}
            
            # Generate forecast using the new sequential architecture
            try:
                forecast = self.lstm_forecaster.forecast(data, len(test_data))
                if len(forecast) > 0:
                    print(f"✅ LSTM forecast successful")
                else:
                    return {'error': 'LSTM forecasting failed - empty forecast'}
            except Exception as e:
                print(f"❌ LSTM forecasting failed: {e}")
                return {'error': f'LSTM forecasting failed: {str(e)}'}
            
            if forecast is None or len(forecast) == 0:
                return {'error': 'LSTM forecasting failed with all sequence lengths'}
            
            # Validate forecast
            validation = self.lstm_forecaster.validate_forecast(forecast, data)
            
            # Calculate metrics
            if len(forecast) > 0 and len(test_data) > 0:
                min_len = min(len(forecast), len(test_data))
                forecast_compare = forecast[:min_len]
                test_compare = test_data[:min_len]
                
                mse = np.mean((test_compare - forecast_compare) ** 2)
                mae = np.mean(np.abs(test_compare - forecast_compare))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((test_compare - forecast_compare) / test_compare)) * 100
                
                results = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'validation': validation
                }
                
                print(f"✅ LSTM: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
                return results
            else:
                return {'error': 'Invalid forecast or test data'}
                
        except Exception as e:
            print(f"❌ LSTM evaluation error: {e}")
            return {'error': str(e)}
    
    def evaluate_cnn(self, data: np.ndarray, test_data: np.ndarray) -> dict:
        """Evaluate CNN model"""
        print("🔍 Evaluating CNN Model...")
        try:
            if not self.cnn_forecaster.is_trained:
                return {'error': 'CNN model not trained'}
            
            # Generate forecast
            forecast = self.cnn_forecaster.forecast(data, len(test_data))
            
            # Validate forecast
            validation = self.cnn_forecaster.validate_forecast(forecast, data)
            
            # Calculate metrics
            if len(forecast) > 0 and len(test_data) > 0:
                min_len = min(len(forecast), len(test_data))
                forecast_compare = forecast[:min_len]
                test_compare = test_data[:min_len]
                
                mse = np.mean((test_compare - forecast_compare) ** 2)
                mae = np.mean(np.abs(test_compare - forecast_compare))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((test_compare - forecast_compare) / test_compare)) * 100
                
                results = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'validation': validation
                }
                
                print(f"✅ CNN: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
                return results
            else:
                return {'error': 'Invalid forecast or test data'}
                
        except Exception as e:
            print(f"❌ CNN evaluation error: {e}")
            return {'error': str(e)}
    
    def evaluate_prophet(self, df: pd.DataFrame, test_data: np.ndarray) -> dict:
        """Evaluate Prophet model"""
        print("🎯 Evaluating Prophet Model...")
        try:
            # Train Prophet model
            train_result = self.prophet_forecaster.train_model(df)
            
            if not train_result.get('success', False):
                return {'error': f'Prophet training failed: {train_result.get("error", "Unknown error")}'}
            
            # Generate forecast
            forecast_result = self.prophet_forecaster.univariate_forecast(df, len(test_data), 'D')
            
            if 'error' in forecast_result:
                return {'error': f'Prophet forecasting failed: {forecast_result["error"]}'}
            
            # Calculate metrics
            forecast_values = forecast_result['forecast']['yhat'].values
            if len(forecast_values) > 0 and len(test_data) > 0:
                min_len = min(len(forecast_values), len(test_data))
                forecast_compare = forecast_values[:min_len]
                test_compare = test_data[:min_len]
                
                mse = np.mean((test_compare - forecast_compare) ** 2)
                mae = np.mean(np.abs(test_compare - forecast_compare))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((test_compare - forecast_compare) / test_compare)) * 100
                
                results = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'cv_rmse': train_result.get('cv_rmse', 'N/A'),
                    'cv_mae': train_result.get('cv_mae', 'N/A')
                }
                
                print(f"✅ Prophet: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
                return results
            else:
                return {'error': 'Invalid forecast or test data'}
                
        except Exception as e:
            print(f"❌ Prophet evaluation error: {e}")
            return {'error': str(e)}
    
    def evaluate_dataset(self, dataset_name: str) -> dict:
        """Evaluate all models on a specific dataset"""
        print(f"\n{'='*60}")
        print(f"📊 EVALUATING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            df = self.data_processor.load_dataset(dataset_name)
            y, train_data, test_data = self.data_processor.prepare_data_for_forecasting(df)
            
            print(f"📈 Dataset loaded: {len(y)} total points, {len(train_data)} train, {len(test_data)} test")
            
            # Load models
            models_loaded = self.load_models(dataset_name)
            
            # Evaluate all methods
            dataset_results = {}
            
            # Traditional methods
            traditional_results = self.evaluate_traditional_methods(train_data, test_data)
            # Add individual traditional methods to results
            for method_name, method_results in traditional_results.items():
                if 'error' not in method_results:
                    dataset_results[method_name] = method_results
                else:
                    dataset_results[method_name] = {'error': method_results['error']}
            
            # LSTM if available
            if models_loaded.get('LSTM', False):
                lstm_results = self.evaluate_lstm(y, test_data)
                dataset_results['LSTM'] = lstm_results
            
            # CNN if available
            if models_loaded.get('CNN', False):
                cnn_results = self.evaluate_cnn(y, test_data)
                dataset_results['CNN'] = cnn_results
            
            # Prophet
            prophet_results = self.evaluate_prophet(df, test_data)
            dataset_results['Prophet'] = prophet_results
            
            return dataset_results
            
        except Exception as e:
            print(f"❌ Error evaluating dataset {dataset_name}: {e}")
            return {'error': str(e)}
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE MODEL EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        report = []
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("=" * 50)
        
        # Collect all results
        all_models = set()
        for dataset_results in self.results.values():
            for method in dataset_results.keys():
                all_models.add(method)
        
        # Create comparison table
        for dataset_name, dataset_results in self.results.items():
            report.append(f"\n📊 {dataset_name.upper()}")
            report.append("-" * 30)
            
            for method, results in dataset_results.items():
                if 'error' in results:
                    report.append(f"{method:15} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>10}")
                else:
                    rmse = results.get('rmse', 'N/A')
                    mae = results.get('mae', 'N/A')
                    mape = results.get('mape', 'N/A')
                    
                    if isinstance(rmse, (int, float)):
                        rmse_str = f"{rmse:.4f}"
                    else:
                        rmse_str = str(rmse)
                    
                    if isinstance(mae, (int, float)):
                        mae_str = f"{mae:.4f}"
                    else:
                        mae_str = str(mae)
                    
                    if isinstance(mape, (int, float)):
                        mape_str = f"{mape:.2f}%"
                    else:
                        mape_str = str(mape)
                    
                    report.append(f"{method:15} | {rmse_str:>10} | {mae_str:>10} | {mape_str:>10}")
        
        # Add header
        header = f"{'Method':15} | {'RMSE':>10} | {'MAE':>10} | {'MAPE':>10}"
        report.insert(1, header)
        report.insert(2, "-" * 50)
        
        return "\n".join(report)
    
    def run_evaluation(self):
        """Run comprehensive evaluation on all datasets"""
        print("🚀 Starting comprehensive model evaluation...")
        print("This will test all trained models and compare their performance.")
        print()
        
        start_time = time.time()
        
        # Datasets to evaluate
        datasets = ['airline_passengers', 'female_births', 'restaurant_visitors', 'superstore_sales']
        
        # Evaluate each dataset
        for dataset in datasets:
            try:
                dataset_results = self.evaluate_dataset(dataset)
                self.results[dataset] = dataset_results
            except Exception as e:
                print(f"❌ Failed to evaluate {dataset}: {e}")
                self.results[dataset] = {'error': str(e)}
        
        # Generate summary report
        summary = self.generate_summary_report()
        print(summary)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(summary)
            f.write(f"\n\nDetailed Results:\n")
            f.write(str(self.results))
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"⏱️  Total evaluation time: {time.time() - start_time:.0f} seconds")
        
        return self.results

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()
    
    print("\n🎯 Evaluation completed!")
    print("Check the generated results file for detailed performance metrics.")

if __name__ == "__main__":
    main()
