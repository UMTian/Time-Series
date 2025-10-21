#!/usr/bin/env python3
"""
Test seasonal decomposition functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from data_processor import TimeSeriesDataProcessor

def test_decomposition():
    """Test seasonal decomposition on superstore_sales dataset"""
    print("🔍 TESTING SEASONAL DECOMPOSITION")
    print("=" * 50)
    
    try:
        # Load the superstore_sales dataset
        processor = TimeSeriesDataProcessor()
        df = processor.load_dataset('superstore_sales')
        
        if df is None:
            print("❌ Failed to load superstore_sales dataset")
            return False
        
        print(f"📊 Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        
        # Get the numeric column for forecasting
        numeric_col = processor.get_numeric_column(df)
        if numeric_col is None:
            print("❌ No numeric column found")
            return False
        
        print(f"   Numeric column: {numeric_col}")
        
        # Get the data values
        y = df[numeric_col].values
        print(f"   Data values shape: {y.shape}")
        print(f"   Data range: [{np.min(y):.2f}, {np.max(y):.2f}]")
        print(f"   Data mean: {np.mean(y):.2f}")
        print(f"   Data std: {np.std(y):.2f}")
        
        # Test seasonal decomposition
        print(f"\n📈 Testing Seasonal Decomposition...")
        decomposition = processor.seasonal_decompose(y, period=12)
        
        if 'error' in decomposition:
            print(f"❌ Seasonal decomposition failed: {decomposition['error']}")
            return False
        
        print(f"✅ Seasonal decomposition successful")
        print(f"   Trend shape: {decomposition['trend'].shape}")
        print(f"   Seasonal shape: {decomposition['seasonal'].shape}")
        print(f"   Residual shape: {decomposition['residual'].shape}")
        print(f"   Period: {decomposition['period']}")
        
        # Check if trend has variability
        trend_data = decomposition['trend'][~np.isnan(decomposition['trend'])]
        if len(trend_data) > 0:
            trend_std = np.std(trend_data)
            print(f"   Trend std: {trend_std:.2f}")
            
            if trend_std > 100:
                print(f"   ✅ Trend has significant variability")
            else:
                print(f"   ⚠️ Trend has low variability")
        else:
            print(f"   ❌ Trend data is empty or all NaN")
        
        # Check seasonal component
        seasonal_data = decomposition['seasonal'][~np.isnan(decomposition['seasonal'])]
        if len(seasonal_data) > 0:
            seasonal_std = np.std(seasonal_data)
            print(f"   Seasonal std: {seasonal_std:.2f}")
            
            if seasonal_std > 50:
                print(f"   ✅ Seasonal component has significant variability")
            else:
                print(f"   ⚠️ Seasonal component has low variability")
        else:
            print(f"   ❌ Seasonal data is empty or all NaN")
        
        # Check residuals
        residual_data = decomposition['residual'][~np.isnan(decomposition['residual'])]
        if len(residual_data) > 0:
            residual_std = np.std(residual_data)
            print(f"   Residual std: {residual_std:.2f}")
            
            if residual_std > 200:
                print(f"   ✅ Residuals have significant variability")
            else:
                print(f"   ⚠️ Residuals have low variability")
        else:
            print(f"   ❌ Residual data is empty or all NaN")
        
        print(f"\n🎉 Seasonal decomposition test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_decomposition()
    if success:
        print("\n✅ Seasonal decomposition working correctly.")
    else:
        print("\n❌ Seasonal decomposition has issues.")
