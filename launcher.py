#!/usr/bin/env python3
"""
Simple launcher script for the Diabetes Prediction System
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import joblib
        import sklearn
        import xgboost
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model_exists():
    """Check if trained model exists"""
    return os.path.exists('diabetes_predection_model/best_model.pkl')

def train_model():
    """Train the machine learning model"""
    print("🔄 Training machine learning models...")
    try:
        result = subprocess.run([sys.executable, 'diabetes_prediction.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Diabetes Prediction System...")
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'run.py'])
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    print("🏥 Diabetes Prediction System Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if model exists
    if not check_model_exists():
        print("⚠️  No trained model found. Training new model...")
        if not train_model():
            print("❌ Failed to train model. Please check error messages above.")
            return
    else:
        print("✅ Trained model found")
        
        # Ask if user wants to retrain
        choice = input("Do you want to retrain the model? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            if not train_model():
                print("❌ Failed to retrain model. Using existing model.")
    
    # Launch the application
    launch_app()

if __name__ == "__main__":
    main()