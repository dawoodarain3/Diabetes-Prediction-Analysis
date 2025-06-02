
# Diabetes Risk Prediction System

A comprehensive machine learning system for predicting diabetes risk using statistical analysis and multiple ML algorithms. This project implements a complete end-to-end pipeline from data preprocessing to model deployment with a web interface.

## Project Features

- **Multiple ML Models**: Logistic Regression, Random Forest, SVM, and XGBoost
- **Data Preprocessing**: Handles missing values, outliers, and feature scaling
- **Statistical Analysis**: Correlation analysis and feature importance evaluation
- **Model Validation**: Cross-validation and performance comparison
- **Web Interface**: Streamlit-based interactive prediction tool
- **Synthetic Data Generation**: Creates medically realistic datasets when real data unavailable
- **Model Persistence**: Saves trained models for future use
- **Visualization**: Comprehensive plots for data analysis and model evaluation

## Overview

This project predicts diabetes risk using the Pima Indians Diabetes Dataset or synthetically generated medical data. It implements multiple machine learning models and provides an interactive web interface for real-time predictions. The system is designed with medical accuracy in mind, incorporating realistic data patterns and medical knowledge.

## Dataset Features

| Feature | Description | Medical Significance |
|---------|-------------|---------------------|
| Pregnancies | Number of times pregnant | Higher pregnancies increase diabetes risk |
| Glucose | Plasma glucose concentration (mg/dL) | Primary diabetes indicator |
| BloodPressure | Diastolic blood pressure (mm Hg) | Hypertension correlates with diabetes |
| SkinThickness | Triceps skin fold thickness (mm) | Body fat distribution indicator |
| Insulin | 2-Hour serum insulin (mu U/ml) | Insulin resistance marker |
| BMI | Body mass index (kg/m²) | Obesity is major diabetes risk factor |
| DiabetesPedigreeFunction | Genetic predisposition score | Family history importance |
| Age | Age in years | Risk increases with age |
| Outcome | Target variable (1: Diabetic, 0: Non-Diabetic) | Prediction target |

## Technical Architecture

### Data Processing Pipeline
1. **Data Validation**: Checks for medical consistency and outliers
2. **Missing Value Imputation**: Uses median/mean based on distribution
3. **Feature Engineering**: Handles zero values that represent missing data
4. **Normalization**: StandardScaler for linear models

### Machine Learning Models
- **Logistic Regression**: Baseline interpretable model with odds ratios
- **Random Forest**: Ensemble method with feature importance ranking
- **Support Vector Machine**: Non-linear classification with RBF kernel
- **XGBoost**: Gradient boosting for maximum performance

### Model Evaluation
- **Cross-Validation**: 5-fold stratified validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Medical Validation**: Tests with known high/low risk cases

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv diabetes_env
   source diabetes_env/bin/activate  # On Windows: diabetes_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create directories (if not exist):**
   ```bash
   mkdir -p data plots results diabetes_predection_model
   ```

## Custom Pre-trained Model

Download the Custom pre-trained model from Kaggle and skip training from scratch:
**[Diabetes Prediction Model](https://www.kaggle.com/models/dawoodarain/diabetes_predection_model)**

Extract and place the model files in the `diabetes_predection_model/` directory:
- `best_model.pkl` - Best performing model
- `scaler.pkl` - Feature scaler
- `model_metadata.json` - Model information and performance metrics

## Usage

### 1. Train Models (Fresh Training)
```bash
python diabetes_prediction.py
```
This will:
- Load/generate dataset
- Train all models
- Save best performing model
- Generate evaluation plots
- Save performance metrics

### 2. Web Application
```bash
streamlit run run.py
```
Access at `http://localhost:8501` for interactive predictions

### 3. Quick Launcher
```bash
python launcher.py
```
Provides menu-driven interface for all operations

### 4. Direct Prediction (Python API)
```python
from diabetes_prediction import DiabetesMLTrainer
import joblib
import numpy as np

# Load trained model
model = joblib.load('diabetes_predection_model/best_model.pkl')
scaler = joblib.load('diabetes_predection_model/scaler.pkl')

# Make prediction: [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]
features = np.array([[2, 120, 80, 25, 100, 28.5, 0.5, 35]])
features_scaled = scaler.transform(features)
risk_probability = model.predict_proba(features_scaled)[0][1]

print(f"Diabetes Risk: {risk_probability:.1%}")
```

## Model Performance

Based on synthetic medical data validation:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **0.799** | **0.711** | **0.676** | **0.693** | **0.863** |
| Random Forest | 0.792 | 0.697 | 0.647 | 0.671 | 0.851 |
| Logistic Regression | 0.771 | 0.659 | 0.618 | 0.638 | 0.834 |
| SVM | 0.760 | 0.641 | 0.588 | 0.614 | 0.823 |

## Key Medical Insights

- **Glucose levels** show strongest correlation with diabetes risk (r = 0.487)
- **BMI** significantly higher in diabetic patients (32.5 vs 30.1 average)
- **Age** demonstrates moderate positive correlation with diabetes outcome
- **Family history** (DiabetesPedigreeFunction) provides valuable genetic risk information
- **Multiple pregnancies** increase diabetes risk, especially after age 35

## Project Structure

```
├── diabetes_prediction.py     # Main ML training pipeline
├── run.py                    # Streamlit web application
├── launcher.py               # Interactive menu system
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── project_structure.md      # Detailed file descriptions
├── data/                     # Dataset storage
│   ├── diabetes_sample.csv   # Sample dataset
│   └── diabetes_synthetic.csv # Generated realistic data
├── diabetes_predection_model/ # Trained models
│   ├── best_model.pkl        # Best performing model
│   ├── scaler.pkl           # Feature scaler
│   ├── model_metadata.json  # Model information
│   └── *.pkl                # Individual model files
├── plots/                    # Generated visualizations
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   └── model_evaluation.png
└── results/                  # Analysis outputs
    └── model_performance.csv # Detailed metrics
```

## System Requirements

### Software Dependencies
- **Python 3.8+** (Recommended: 3.9-3.11)
- **Core Libraries**: scikit-learn, pandas, numpy
- **ML Libraries**: xgboost, scipy
- **Visualization**: matplotlib, seaborn
- **Web Interface**: streamlit
- **Model Persistence**: joblib

### Hardware Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 500MB for models and data
- **CPU**: Multi-core recommended for faster training

## Data Quality Features

### Synthetic Data Generation
When real medical data is unavailable, the system generates medically realistic synthetic data:
- **Age-correlated pregnancies**: Realistic pregnancy patterns by age group
- **BMI-glucose correlation**: Medically accurate relationships
- **Risk factor modeling**: Based on established medical research
- **Realistic distributions**: Mimics real-world medical data patterns

### Data Validation
- **Medical consistency checks**: Ensures biologically plausible values
- **Correlation validation**: Verifies expected medical relationships
- **Outlier detection**: Identifies and handles extreme values appropriately

## Advanced Features

### Model Interpretability
- **Feature importance ranking**: Identifies most predictive factors
- **Correlation analysis**: Shows relationships between variables
- **Medical validation**: Tests with known risk profiles

### Performance Optimization
- **Cross-validation**: Ensures robust model performance
- **Hyperparameter tuning**: Optimized for medical prediction accuracy
- **Ensemble methods**: Combines multiple models for better predictions

## Contact & Support

- **Email**: dawoodarain025@gmail.com
- **GitHub**: [dawoodarain3](https://github.com/dawoodarain)
- **LinkedIn**: [dawoodahmed](https://www.linkedin.com/in/dawood-ahmed-84776017b/)

## Disclaimer

**Important Medical Notice**: This tool is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. The predictions are based on statistical models and may not reflect individual medical circumstances.
