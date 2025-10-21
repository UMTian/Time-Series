#!/usr/bin/env python3
"""
Script to optimize LSTM performance for time series forecasting
Addresses the flat forecast line and high error metrics issues
"""

import numpy as np
import pandas as pd
from deep_learning_forecasting import LSTMForecasting
import warnings
warnings.filterwarnings('ignore')

class OptimizedLSTMForecaster:
    """Optimized LSTM forecaster with enhanced feature engineering"""
    
    def __init__(self, input_size: int = 30, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.is_trained = False
        
    def create_optimized_sequences(self, data: np.ndarray, seq_length: int) -> tuple:
        """Create sequences with optimized feature engineering for time series"""
        if len(data) < seq_length + 1:
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            # Extract sequence
            sequence = data[i:i + seq_length]
            target = data[i + seq_length]
            
            # Create optimized features
            features = self._create_optimized_features(sequence, i, data)
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def _create_optimized_features(self, sequence: np.ndarray, position: int, full_data: np.ndarray) -> np.ndarray:
        """Create optimized features specifically for time series forecasting"""
        features = np.zeros(self.input_size)
        seq_len = len(sequence)
        
        if seq_len == 0:
            return features
        
        # Core sequence features (first 20 features)
        seq_features = min(seq_len, 20)
        features[:seq_features] = sequence[-seq_features:]
        
        # Statistical features (features 20-25)
        if seq_len > 0:
            features[20] = np.mean(sequence)
            features[21] = np.std(sequence) if seq_len > 1 else 0.0
            features[22] = np.min(sequence)
            features[23] = np.max(sequence)
            features[24] = np.median(sequence)
            features[25] = np.ptp(sequence)  # range
        
        # Trend features (features 26-28)
        if seq_len > 2:
            # Linear trend
            x = np.arange(seq_len)
            z = np.polyfit(x, sequence, 1)
            features[26] = z[0]  # slope
            
            # Quadratic trend
            if seq_len > 3:
                z2 = np.polyfit(x, sequence, 2)
                features[27] = z2[1]  # linear coefficient
                features[28] = z2[2]  # quadratic coefficient
            else:
                features[27] = 0.0
                features[28] = 0.0
        
        # Seasonal features (features 29-30)
        if len(full_data) > 12 and position >= 12:
            features[29] = full_data[position - 12]  # 12-period seasonality
        
        return features
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data with robust scaling"""
        if len(data) == 0:
            return data
        
        try:
            # Use robust scaling to handle outliers
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized = (data - np.median(data)) / iqr
            else:
                normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Clip to reasonable range
            return np.clip(normalized, -3, 3)
        except Exception as e:
            print(f"Normalization failed: {e}")
            return data
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data"""
        if len(data) == 0:
            return data
        
        try:
            # Reverse the robust scaling
            q75, q25 = np.percentile(self.original_data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                denormalized = data * iqr + np.median(self.original_data)
            else:
                denormalized = data * (np.std(self.original_data) + 1e-8) + np.mean(self.original_data)
            return denormalized
        except Exception as e:
            print(f"Denormalization failed: {e}")
            return data
    
    def build_optimized_model(self):
        """Build optimized LSTM model for time series"""
        import torch
        import torch.nn as nn
        
        class OptimizedLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(OptimizedLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # LSTM layers
                self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, 
                                    batch_first=True, dropout=dropout)
                self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, 1, 
                                    batch_first=True, dropout=dropout)
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(hidden_size // 2, num_heads=4, 
                                                    dropout=dropout, batch_first=True)
                
                # Output layers
                self.fc1 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden_size // 4, 1)
                
                # Batch normalization
                self.bn1 = nn.BatchNorm1d(hidden_size // 4)
                
            def forward(self, x):
                # First LSTM layer
                lstm_out, _ = self.lstm1(x)
                
                # Second LSTM layer
                lstm_out2, _ = self.lstm2(lstm_out)
                
                # Attention mechanism
                attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
                
                # Global average pooling
                pooled = torch.mean(attn_out, dim=1)
                
                # Fully connected layers
                out = self.fc1(pooled)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                
                return out
        
        return OptimizedLSTM(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def train_optimized_model(self, train_data: np.ndarray, seq_length: int = 20, 
                             epochs: int = 150, learning_rate: float = 0.001, 
                             validation_split: float = 0.2) -> dict:
        """Train the optimized LSTM model"""
        if len(train_data) < seq_length + 1:
            return {'error': f'Insufficient data: Need at least {seq_length + 1} points, got {len(train_data)}'}
        
        try:
            # Store original data for denormalization
            self.original_data = train_data.copy()
            
            # Normalize data
            normalized_data = self.normalize_data(train_data)
            
            # Create sequences
            X, y = self.create_optimized_sequences(normalized_data, seq_length)
            
            if len(X) == 0:
                return {'error': 'Failed to create training sequences'}
            
            # Convert to tensors
            import torch
            X_tensor = torch.FloatTensor(X).unsqueeze(1).to('cpu')
            y_tensor = torch.FloatTensor(y).to('cpu')
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
            
            # Build and train model
            self.model = self.build_optimized_model().to('cpu')
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7, min_lr=1e-6)
            criterion = torch.nn.HuberLoss(delta=0.5)
            
            train_losses, val_losses = [], []
            best_val_loss = float('inf')
            patience, patience_counter = 25, 0
            
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss = loss.item()
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val)
                    val_losses.append(val_loss.item())
                
                scheduler.step(val_loss)
                self.model.train()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    self.model.load_state_dict(best_model_state)
                    break
            
            self.model.eval()
            self.is_fitted = True
            self.is_trained = True
            
            # Calculate final metrics
            with torch.no_grad():
                train_rmse = torch.sqrt(torch.mean((self.model(X_train).squeeze() - y_train) ** 2)).item()
                val_rmse = torch.sqrt(torch.mean((self.model(X_val).squeeze() - y_val) ** 2)).item()
            
            return {
                'success': True,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_train_rmse': train_rmse,
                'final_val_rmse': val_rmse
            }
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}
    
    def forecast(self, data: np.ndarray, forecast_steps: int, seq_length: int = 20) -> np.ndarray:
        """Generate optimized forecast"""
        if self.model is None or not self.is_fitted:
            raise ValueError("Model must be trained before forecasting")
        
        if len(data) < seq_length:
            raise ValueError(f"Data length ({len(data)}) must be at least {seq_length}")
        
        self.model.eval()
        normalized_data = self.normalize_data(data)
        forecasts = []
        
        import torch
        with torch.no_grad():
            current_input = normalized_data[-seq_length:].copy()
            
            for step in range(forecast_steps):
                # Create features with safe position indexing
                safe_position = min(len(normalized_data) - 1, len(normalized_data) + step - 1)
                features = self._create_optimized_features(current_input, safe_position, normalized_data)
                input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1).to('cpu')
                
                # Generate prediction
                prediction = self.model(input_tensor).item()
                
                # Smooth transition for first prediction
                if step == 0:
                    prediction = prediction * 0.2 + normalized_data[-1] * 0.8
                
                forecasts.append(prediction)
                
                # Update input sequence
                current_input = np.roll(current_input, -1)
                current_input[-1] = prediction
        
        # Denormalize forecasts
        return self.denormalize_data(np.array(forecasts))

def demonstrate_optimized_performance():
    """Demonstrate the optimized LSTM performance"""
    print("=== Optimized LSTM Performance Demonstration ===\n")
    
    # Generate realistic time series data
    np.random.seed(42)
    n_points = 1000
    t = np.linspace(0, 40, n_points)
    
    # Create data with trend, seasonality, and noise
    trend = 100 + 2 * t
    seasonality = 50 * np.sin(2 * np.pi * t / 4) + 25 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 15, n_points)
    data = trend + seasonality + noise
    
    print(f"Generated time series with {n_points} points")
    print(f"Data range: {np.min(data):.2f} to {np.max(data):.2f}")
    print(f"Data mean: {np.mean(data):.2f}")
    print(f"Data std: {np.std(data):.2f}\n")
    
    # Test optimized LSTM
    print("=== Testing Optimized LSTM ===")
    try:
        optimized_lstm = OptimizedLSTMForecaster(
            input_size=30, 
            hidden_size=64, 
            num_layers=2, 
            dropout=0.1
        )
        print("✓ Optimized LSTM model initialized")
        
        # Train the model
        print("\nTraining optimized LSTM model...")
        training_result = optimized_lstm.train_optimized_model(
            data, 
            seq_length=20, 
            epochs=150, 
            learning_rate=0.001
        )
        
        if training_result.get('success'):
            print("✓ Optimized LSTM model trained successfully")
            print(f"Final training RMSE: {training_result['final_train_rmse']:.6f}")
            print(f"Final validation RMSE: {training_result['final_val_rmse']:.6f}")
            
            # Generate forecast
            print("\nGenerating optimized forecast...")
            forecast = optimized_lstm.forecast(data, forecast_steps=50, seq_length=20)
            print(f"✓ Generated forecast with {len(forecast)} steps")
            
            # Calculate forecast metrics
            actual_values = data[-50:]  # Last 50 values for comparison
            forecast_rmse = np.sqrt(np.mean((forecast - actual_values) ** 2))
            forecast_mae = np.mean(np.abs(forecast - actual_values))
            forecast_mape = np.mean(np.abs((actual_values - forecast) / (actual_values + 1e-8))) * 100
            
            print(f"Forecast RMSE: {forecast_rmse:.2f}")
            print(f"Forecast MAE: {forecast_mae:.2f}")
            print(f"Forecast MAPE: {forecast_mape:.2f}%")
            
            # Check if forecast shows variation
            forecast_std = np.std(forecast)
            if forecast_std > 1.0:
                print("✓ Forecast shows good variation (not flat)")
            else:
                print("⚠ Forecast may be too flat")
                
        else:
            print(f"✗ Training failed: {training_result.get('error')}")
            
    except Exception as e:
        print(f"✗ Testing failed: {str(e)}")
    
    print("\n=== Demonstration Complete ===")

if __name__ == "__main__":
    demonstrate_optimized_performance()
