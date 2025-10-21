#!/usr/bin/env python3
"""
Simple script to run model training
"""

if __name__ == "__main__":
    print("🚀 Starting model training...")
    print("This will train LSTM and CNN models for all datasets.")
    print("Training may take several minutes...")
    print()
    
    try:
        from train_models import main
        main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch tensorflow pandas numpy")
