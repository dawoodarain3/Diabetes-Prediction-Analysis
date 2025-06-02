# Predicting Diabetes Risk Using Statistical and Machine Learning Methods

## 📋 Project Overview

This project develops a comprehensive predictive model to assess diabetes risk using statistical analysis and machine learning techniques. The solution combines traditional statistical methods with modern ML algorithms to provide accurate, interpretable predictions for diabetes risk assessment.

## 🎯 Objectives

- **Statistical Analysis**: Explore and preprocess medical data using rigorous statistical methods
- **Hypothesis Testing**: Apply statistical tests to validate relationships between features and diabetes risk
- **Feature Selection**: Use statistical techniques to identify the most significant predictors
- **Model Development**: Build and compare multiple machine learning models
- **Interpretability**: Provide statistical confidence intervals and significance metrics for clinical interpretation

## 📊 Dataset

The project utilizes the **Pima Indians Diabetes Dataset**, containing health-related variables for female patients. The dataset includes:

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)²) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | Target variable (1: Diabetic, 0: Non-Diabetic) |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv diabetes_env
   source diabetes_env/bin/activate  # On Windows: diabetes_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**
   ```bash
   mkdir data plots results models
   ```

5. **Download the dataset**
   - Download the Pima Indians Diabetes Dataset from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
   - Place it in the `data/` directory as `diabetes.csv`
   - Or run the script without the dataset - it will generate sample data automatically

## 🏃‍♂️ Usage

### Quick Start
```bash
python main.py
```

### Step-by-Step Analysis
```python
from main import DiabetesPredictionAnalysis

# Initialize analyzer
analyzer = DiabetesPredictionAnalysis()

# Run complete analysis pipeline
analyzer.run_complete_analysis('data/diabetes.csv')
```

### Individual Components
```python
# Load and explore data
analyzer.load_data('data/diabetes.csv')
analyzer.exploratory_data_analysis()

# Statistical analysis
analyzer.hypothesis_testing()
analyzer.feature_selection()

# Machine learning
analyzer.build_models()
analyzer.evaluate_models()
```

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Mean, median, standard deviation, quartiles
- **Distribution Analysis**: Histograms, box plots for identifying skewness and outliers
- **Correlation Analysis**: Heatmaps to understand feature relationships
- **Target Variable Analysis**: Class distribution and imbalance assessment

### 2. Data Preprocessing
- **Missing Value Treatment**: Mean/median imputation based on distribution
- **Outlier Detection**: IQR and Z-score methods for anomaly identification
- **Feature Scaling**: Standardization for algorithm compatibility
- **Data Validation**: Statistical tests for data quality assurance

### 3. Statistical Hypothesis Testing
- **Two-Sample T-Tests**: Compare feature means between diabetic and non-diabetic groups
- **Significance Testing**: P-value analysis with α = 0.05
- **Effect Size Calculation**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for population parameter estimation

### 4. Feature Selection
- **Correlation Analysis**: Pearson correlation with target variable
- **Chi-Square Tests**: Independence tests for categorical features
- **Statistical Significance**: P-value based feature ranking
- **Domain Knowledge**: Medical literature-informed feature selection

### 5. Model Development
- **Logistic Regression**: Baseline interpretable model with statistical outputs
- **Random Forest**: Ensemble method with feature importance
- **Support Vector Machine**: Non-linear classification with RBF kernel
- **XGBoost**: Gradient boosting for maximum predictive performance

### 6. Model Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Validation**: 5-fold stratified CV for robust performance estimation
- **Statistical Tests**: McNemar's test for model comparison
- **Confidence Intervals**: Bootstrap CI for performance metrics

### 7. Statistical Interpretation
- **Coefficient Analysis**: Odds ratios and confidence intervals from logistic regression
- **P-values**: Statistical significance of each feature
- **Model Diagnostics**: Residual analysis and goodness-of-fit tests
- **Clinical Interpretation**: Translating statistical results to medical insights

## 📈 Results and Outputs

The analysis generates comprehensive outputs in the following directories:

### `/plots/`
- `feature_distributions.png` - Histograms of all features
- `boxplots_by_outcome.png` - Box plots comparing diabetic vs non-diabetic groups
- `correlation_heatmap.png` - Feature correlation matrix
- `model_evaluation.png` - ROC curves and performance comparison

### `/results/`
- `model_performance.csv` - Detailed performance metrics for all models
- `processed_data.csv` - Cleaned and preprocessed dataset
- `statistical_summary.txt` - Hypothesis testing results and statistical insights

### Console Output
- Descriptive statistics and data quality reports
- Hypothesis testing results with p-values and significance levels
- Model performance comparison table
- Statistical interpretation of logistic regression coefficients

## 🔧 Configuration

### Model Parameters
Modify hyperparameters in `main.py`:

```python
self.models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        C=1.0  # Regularization strength
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    ),
    # ... other models
}
```

### Statistical Significance Level
Change the alpha level for hypothesis testing:
```python
alpha = 0.05  # Default significance level
significant = p_value < alpha
```

## 📊 Sample Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.771 | 0.659 | 0.618 | 0.638 | 0.834 |
| Random Forest | 0.792 | 0.697 | 0.647 | 0.671 | 0.851 |
| SVM | 0.760 | 0.641 | 0.588 | 0.614 | 0.823 |
| XGBoost | 0.799 | 0.711 | 0.676 | 0.693 | 0.863 |

### Statistical Insights
- **Glucose** levels show the strongest correlation with diabetes risk (r = 0.487, p < 0.001)
- **BMI** is significantly higher in diabetic patients (32.5 vs 30.1, p < 0.001)
- **Age** demonstrates moderate positive correlation with diabetes outcome
- **Insulin** levels require careful interpretation due to missing value patterns

## 🚀 Deployment Options

### Streamlit Web App
```bash
streamlit run app.py
```

### Flask API
```bash
python flask_app.py
```

### Jupyter Notebook
```bash
jupyter notebook diabetes_analysis.ipynb
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for any changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

2. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

3. American Diabetes Association. (2023). Standards of Medical Care in Diabetes. Diabetes Care, 46(Supplement_1).

## 🔍 Troubleshooting

### Common Issues

**Issue**: ImportError for statsmodels or xgboost
```bash
pip install --upgrade statsmodels xgboost
```

**Issue**: Memory error with large datasets
- Reduce dataset size or use sampling
- Consider using Dask for larger-than-memory processing

**Issue**: Plots not displaying
```python
import matplotlib
matplotlib.use('Agg')  # For headless environments
```

### Performance Optimization
- Use `n_jobs=-1` for parallel processing in sklearn models
- Consider feature selection to reduce dimensionality
- Use early stopping for XGBoost to prevent overfitting

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)
- **LinkedIn**: [yourprofile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Scikit-learn community for excellent ML tools
- Statsmodels developers for statistical computing capabilities
- Matplotlib and Seaborn teams for visualization tools

---

**Note**: This project is for educational and research purposes. For clinical applications, please consult with healthcare professionals and follow appropriate regulatory guidelines.