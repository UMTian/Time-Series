#!/usr/bin/env python3
"""
Comprehensive Superstore Sales Model Testing & Evaluation
"""

import os
import numpy as np
import warnings
from deep_learning_forecasting import LSTMForecasting, CNNForecasting

warnings.filterwarnings('ignore')

def create_realistic_superstore_dataset(n_points=800):
    """Create realistic superstore sales dataset"""
    np.random.seed(42)
    t = np.linspace(0, 40, n_points)
    
    # Phase 1: Early growth (0-200 points)
    phase1 = 800 + 8 * t[:200] + 120 * np.sin(2 * np.pi * t[:200] / 4) + \
             80 * np.sin(2 * np.pi * t[:200] / 12) + np.random.normal(0, 40, 200)
    
    # Phase 2: Rapid expansion (200-500 points)
    phase2 = 1000 + 12 * (t[200:500] - t[200]) + 150 * np.sin(2 * np.pi * t[200:500] / 4) + \
             100 * np.sin(2 * np.pi * t[200:500] / 12) + np.random.normal(0, 50, 300)
    
    # Phase 3: Mature growth (500-800 points)
    phase3 = 1400 + 6 * (t[500:] - t[500]) + 180 * np.sin(2 * np.pi * t[500:] / 4) + \
             120 * np.sin(2 * np.pi * t[500:] / 12) + np.random.normal(0, 60, 300)
    
    # Combine phases
    data = np.concatenate([phase1, phase2, phase3])
    
    # Add business cycles and special events
    business_cycle = 50 * np.sin(2 * np.pi * t / 24)
    black_friday = 100 * np.exp(-((t % 12 - 11) ** 2) / 0.5)
    summer_sales = 80 * np.exp(-((t % 12 - 6) ** 2) / 1.0)
    
    data += business_cycle + black_friday + summer_sales
    return np.maximum(0, data)

def evaluate_forecast_quality(forecast, data):
    """Evaluate forecast quality"""
    try:
        if len(forecast) == 0 or len(data) < len(forecast):
            return {'quality_score': 0, 'quality_class': 'INVALID'}
        
        # Basic statistics
        forecast_std = np.std(forecast)
        data_std = np.std(data[-len(forecast):])
        variation_ratio = forecast_std / (data_std + 1e-8)
        
        # Simple quality scoring
        if variation_ratio > 0.8:
            quality_score = 85
            quality_class = "EXCELLENT"
        elif variation_ratio > 0.6:
            quality_score = 75
            quality_class = "GOOD"
        elif variation_ratio > 0.4:
            quality_score = 65
            quality_class = "ACCEPTABLE"
        else:
            quality_score = 45
            quality_class = "NEEDS_IMPROVEMENT"
        
        return {
            'quality_score': quality_score,
            'quality_class': quality_class,
            'variation_ratio': variation_ratio
        }
    except Exception as e:
        return {'quality_score': 0, 'quality_class': 'ERROR', 'error': str(e)}

def test_lstm_model(model_path, model_name, dataset):
    """Test LSTM model with flexible architecture detection"""
    try:
        print(f"\n🔍 Testing {model_name}...")
        
        # Try different LSTM configurations based on model name
        if 'ultra_precision' in model_name or 'mega_ensemble' in model_name:
            # Ultra-precision models use 6+ layers and larger hidden sizes
            lstm = LSTMForecasting(
                seq_length=40,
                hidden_size=768,  # Larger hidden size
                num_layers=6,     # More layers
                dropout=0.25
            )
        elif 'precision' in model_name or 'quality_focused' in model_name:
            # Precision models use 4-5 layers and medium hidden sizes
            lstm = LSTMForecasting(
                seq_length=40,
                hidden_size=640,  # Medium hidden size
                num_layers=4,     # 4 layers
                dropout=0.25
            )
        elif 'balanced' in model_name or 'enhanced' in model_name:
            # Balanced models use 3-4 layers and medium hidden sizes
            lstm = LSTMForecasting(
                seq_length=40,
                hidden_size=384,  # Medium hidden size
                num_layers=3,     # 3 layers
                dropout=0.25
            )
        elif 'cpu_optimized' in model_name:
            # CPU optimized models use smaller parameters
            lstm = LSTMForecasting(
                seq_length=30,
                hidden_size=128,  # Smaller hidden size
                num_layers=2,     # Fewer layers
                dropout=0.25
            )
        elif 'lightweight' in model_name:
            # Lightweight models use minimal parameters
            lstm = LSTMForecasting(
                seq_length=20,
                hidden_size=64,   # Small hidden size
                num_layers=1,     # Single layer
                dropout=0.25
            )
        else:
            # Default configuration for standard models
            lstm = LSTMForecasting(
                seq_length=40,
                hidden_size=256,
                num_layers=4,
                dropout=0.25
            )
        
        # Load model
        lstm.load_model(model_path)
        
        # Generate forecast
        forecast = lstm.forecast(dataset, forecast_steps=60)
        
        # Calculate metrics
        actual_values = dataset[-60:]
        rmse = np.sqrt(np.mean((forecast - actual_values) ** 2))
        mae = np.mean(np.abs(actual_values - forecast))
        mape = np.mean(np.abs((actual_values - forecast) / (actual_values + 1e-8))) * 100
        
        # Evaluate quality
        quality = evaluate_forecast_quality(forecast, dataset)
        
        print(f"✅ {model_name}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Quality: {quality['quality_class']} ({quality['quality_score']}/100)")
        print(f"   Variation Ratio: {quality['variation_ratio']:.3f}")
        
        return {
            'model_type': 'LSTM',
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'quality': quality,
            'forecast_std': np.std(forecast),
            'data_std': np.std(actual_values)
        }
        
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return {'error': str(e)}

def test_cnn_model(model_path, model_name, dataset):
    """Test CNN model"""
    try:
        print(f"\n🔍 Testing {model_name}...")
        
        # Initialize CNN with correct parameters
        cnn = CNNForecasting(
            seq_length=40,
            filters=[64, 128, 256],
            dropout=0.25
        )
        
        # Load model
        cnn.load_model(model_path)
        
        # Generate forecast
        forecast = cnn.forecast(dataset, forecast_steps=60)
        
        # Calculate metrics
        actual_values = dataset[-60:]
        rmse = np.sqrt(np.mean((forecast - actual_values) ** 2))
        mae = np.mean(np.abs(actual_values - forecast))
        mape = np.mean(np.abs((actual_values - forecast) / (actual_values + 1e-8))) * 100
        
        # Evaluate quality
        quality = evaluate_forecast_quality(forecast, dataset)
        
        print(f"✅ {model_name}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Quality: {quality['quality_class']} ({quality['quality_score']}/100)")
        print(f"   Variation Ratio: {quality['variation_ratio']:.3f}")
        
        return {
            'model_type': 'CNN',
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'quality': quality,
            'forecast_std': np.std(forecast),
            'data_std': np.std(actual_values)
        }
        
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return {'error': str(e)}

def main():
    """Main evaluation function"""
    print("🚀 Starting Comprehensive Superstore Sales Model Evaluation")
    print("=" * 70)
    
    # Create dataset
    print("📊 Creating realistic superstore sales dataset...")
    dataset = create_realistic_superstore_dataset(800)
    print(f"✅ Dataset created: {len(dataset)} points")
    print(f"   Range: ${np.min(dataset):.0f} - ${np.max(dataset):.0f}")
    print(f"   Mean: ${np.mean(dataset):.0f}")
    print(f"   Std: ${np.std(dataset):.0f}")
    
    # Find models
    models_dir = "models"
    lstm_models = []
    cnn_models = []
    
    for file in os.listdir(models_dir):
        if 'superstore_sales' in file:
            if file.endswith('.pth'):
                lstm_models.append(file)
            elif file.endswith('.keras'):
                cnn_models.append(file)
    
    print(f"\n📁 Found {len(lstm_models)} LSTM models and {len(cnn_models)} CNN models")
    
    # Test all models
    results = {}
    
    # Test LSTM models
    print("\n🧠 Testing LSTM Models:")
    for model_file in lstm_models:
        model_name = model_file.replace('.pth', '')
        model_path = os.path.join(models_dir, model_file)
        result = test_lstm_model(model_path, model_name, dataset)
        results[model_name] = result
    
    # Test CNN models
    print("\n🔄 Testing CNN Models:")
    for model_file in cnn_models:
        model_name = model_file.replace('.keras', '')
        model_path = os.path.join(models_dir, model_file)
        result = test_cnn_model(model_path, model_name, dataset)
        results[model_name] = result
    
    # Generate report
    print("\n" + "=" * 70)
    print("📊 EVALUATION REPORT")
    print("=" * 70)
    
    # Working models
    working_models = {k: v for k, v in results.items() if 'error' not in v}
    failed_models = {k: v for k, v in results.items() if 'error' in v}
    
    if working_models:
        # Sort by quality score
        sorted_models = sorted(working_models.items(), 
                             key=lambda x: x[1]['quality']['quality_score'], reverse=True)
        
        print(f"\n🏆 TOP PERFORMING MODELS:")
        print("-" * 50)
        
        for i, (model_name, result) in enumerate(sorted_models, 1):
            quality = result['quality']
            print(f"{i:2d}. {model_name}")
            print(f"    Quality: {quality['quality_score']}/100 ({quality['quality_class']})")
            print(f"    RMSE: ${result['rmse']:.2f}")
            print(f"    MAPE: {result['mape']:.2f}%")
            print(f"    Variation: {quality['variation_ratio']:.3f}")
        
        # Summary
        avg_quality = np.mean([m['quality']['quality_score'] for m in working_models.values()])
        avg_rmse = np.mean([m['rmse'] for m in working_models.values()])
        avg_mape = np.mean([m['mape'] for m in working_models.values()])
        
        print(f"\n📈 SUMMARY:")
        print(f"Average Quality: {avg_quality:.1f}/100")
        print(f"Average RMSE: ${avg_rmse:.2f}")
        print(f"Average MAPE: {avg_mape:.2f}%")
        print(f"Working Models: {len(working_models)}")
        
        # Best model
        best_model = sorted_models[0]
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Best Model: {best_model[0]}")
        print(f"Quality: {best_model[1]['quality']['quality_class']}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        print("-" * 50)
        for model_name, result in failed_models.items():
            print(f"• {model_name}: {result['error']}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")

if __name__ == "__main__":
    main()
