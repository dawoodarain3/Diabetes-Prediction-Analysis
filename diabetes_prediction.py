"""
Diabetes Prediction - Clean ML Model Training Script
Professional training pipeline without unnecessary complexity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

class DiabetesMLTrainer:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('data', exist_ok=True)

    def create_realistic_data(self):
        """Create medically realistic diabetes dataset"""
        print("Creating realistic diabetes dataset...")
        np.random.seed(42)
        n_samples = 768
        
        # Generate age distribution (more realistic for diabetes study)
        age = np.random.choice(
            range(21, 81), 
            size=n_samples, 
            p=self.get_age_probabilities()
        )
        
        # Generate pregnancies based on age
        pregnancies = np.array([self.generate_pregnancies(a) for a in age])
        
        # Generate BMI with age correlation
        bmi_base = 25 + (age - 30) * 0.1
        bmi = np.random.normal(bmi_base, 6)
        bmi = np.clip(bmi, 15, 50)
        
        # Generate glucose with realistic distribution
        glucose_base = 85 + (age - 21) * 0.3 + (bmi - 25) * 0.8
        glucose = glucose_base + np.random.normal(0, 15, n_samples)
        glucose = np.clip(glucose, 60, 200)
        
        # Generate blood pressure
        bp_base = 65 + (age - 21) * 0.2 + (bmi - 25) * 0.3
        blood_pressure = bp_base + np.random.normal(0, 8, n_samples)
        blood_pressure = np.clip(blood_pressure, 50, 120)
        
        # Generate skin thickness
        skin_base = 15 + (bmi - 25) * 0.4
        skin_thickness = np.random.normal(skin_base, 5)
        skin_thickness = np.clip(skin_thickness, 5, 50)
        
        # Generate insulin (many zeros, some high values)
        insulin = np.array([self.generate_insulin() for _ in range(n_samples)])
        
        # Generate diabetes pedigree function
        dpf = np.random.gamma(2, 0.3, n_samples)
        dpf = np.clip(dpf, 0.1, 2.5)
        
        # Calculate diabetes risk based on medical factors
        diabetes_risk = self.calculate_diabetes_risk(
            age, glucose, bmi, blood_pressure, pregnancies, dpf, insulin
        )
        
        # Convert to binary outcome (approximately 35% positive)
        threshold = np.percentile(diabetes_risk, 65)
        outcome = (diabetes_risk > threshold).astype(int)
        
        # Create DataFrame with proper data types
        self.data = pd.DataFrame({
            'Pregnancies': pregnancies.astype(int),
            'Glucose': np.round(glucose).astype(int),
            'BloodPressure': np.round(blood_pressure).astype(int),
            'SkinThickness': np.round(skin_thickness).astype(int),
            'Insulin': insulin.astype(int),
            'BMI': np.round(bmi, 1),
            'DiabetesPedigreeFunction': np.round(dpf, 3),
            'Age': age.astype(int),
            'Outcome': outcome.astype(int)
        })
        
        # Save the data
        self.data.to_csv('data/diabetes_synthetic.csv', index=False)
        
        print(f"Dataset created: {self.data.shape}")
        print(f"Diabetes prevalence: {self.data['Outcome'].mean():.1%}")
        
        # Validate the data
        self.validate_data()

    def get_age_probabilities(self):
        """Get realistic age distribution"""
        ages = range(21, 81)
        # Higher probability for middle-aged adults
        probs = [np.exp(-0.5 * ((age - 45) / 15) ** 2) for age in ages]
        total = sum(probs)
        return [p / total for p in probs]

    def generate_pregnancies(self, age):
        """Generate realistic pregnancy count based on age"""
        if age < 25:
            return max(0, np.random.poisson(0.5))
        elif age < 35:
            return max(0, np.random.poisson(2))
        elif age < 45:
            return max(0, np.random.poisson(3))
        else:
            return max(0, np.random.poisson(3.5))

    def generate_insulin(self):
        """Generate realistic insulin values"""
        if np.random.random() < 0.3:  # 30% have zero insulin
            return 0
        else:
            return min(int(np.random.lognormal(mean=4, sigma=0.8)), 700)

    def calculate_diabetes_risk(self, age, glucose, bmi, bp, pregnancies, dpf, insulin):
        """Calculate realistic diabetes risk"""
        # Convert to numpy arrays to ensure proper operations
        age = np.array(age)
        glucose = np.array(glucose)
        bmi = np.array(bmi)
        bp = np.array(bp)
        pregnancies = np.array(pregnancies)
        dpf = np.array(dpf)
        insulin = np.array(insulin)
        
        risk = np.zeros(len(age))
        
        # Glucose effects (strongest predictor)
        risk += np.where(glucose >= 126, 3.0,  # Diabetic range
                np.where(glucose >= 100, 1.5,   # Prediabetic range
                        0.0))                    # Normal range
        
        # BMI effects
        risk += np.where(bmi >= 35, 2.0,        # Severe obesity
                np.where(bmi >= 30, 1.2,        # Obesity
                np.where(bmi >= 25, 0.3,        # Overweight
                        0.0)))                   # Normal weight
        
        # Age effects
        risk += np.where(age >= 65, 1.5,
                np.where(age >= 45, 0.8,
                np.where(age >= 35, 0.3,
                        0.0)))
        
        # Blood pressure effects
        risk += np.where(bp >= 90, 0.5, 0.0)
        
        # Pregnancy effects
        risk += np.where(pregnancies >= 5, 0.4,
                np.where(pregnancies >= 3, 0.2, 0.0))
        
        # Family history effects
        risk += dpf * 0.8
        
        # Insulin effects
        risk += np.where(insulin > 200, 0.3, 0.0)
        
        # Add noise
        risk += np.random.normal(0, 0.2, len(age))
        
        return risk

    def validate_data(self):
        """Validate that generated data makes medical sense"""
        print("\nData Validation:")
        
        # Check correlations
        corr_glucose = self.data['Glucose'].corr(self.data['Outcome'])
        corr_bmi = self.data['BMI'].corr(self.data['Outcome'])
        corr_age = self.data['Age'].corr(self.data['Outcome'])
        
        print(f"Glucose-Diabetes correlation: {corr_glucose:.3f}")
        print(f"BMI-Diabetes correlation: {corr_bmi:.3f}")
        print(f"Age-Diabetes correlation: {corr_age:.3f}")
        
        # Check means
        diabetic = self.data[self.data['Outcome'] == 1]
        non_diabetic = self.data[self.data['Outcome'] == 0]
        
        print(f"Mean Glucose - Diabetic: {diabetic['Glucose'].mean():.1f}, Non-diabetic: {non_diabetic['Glucose'].mean():.1f}")
        print(f"Mean BMI - Diabetic: {diabetic['BMI'].mean():.1f}, Non-diabetic: {non_diabetic['BMI'].mean():.1f}")

    def load_data(self, file_path='data/diabetes.csv'):
        """Load data from file or create sample data"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded from {file_path}: {self.data.shape}")
        except FileNotFoundError:
            print(f"File {file_path} not found. Creating synthetic data...")
            self.create_realistic_data()

    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\nData Preprocessing:")
        
        # Handle zero values that should be NaN
        zero_columns = ['Glucose', 'BloodPressure', 'BMI']
        for col in zero_columns:
            if col in self.data.columns:
                zeros_count = (self.data[col] == 0).sum()
                if zeros_count > 0:
                    print(f"Replacing {zeros_count} zero values in {col}")
                    self.data[col] = self.data[col].replace(0, np.nan)

        # Impute missing values
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if col in ['Insulin']:
                    fill_value = self.data[col].median()
                else:
                    fill_value = self.data[col].mean()
                
                self.data[col].fillna(fill_value, inplace=True)
                print(f"Filled {col} missing values with {fill_value:.2f}")

        print("Data preprocessing completed")

    def prepare_features(self):
        """Prepare features for machine learning"""
        self.X = self.data[self.feature_names]
        self.y = self.data['Outcome']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

    def train_models(self):
        """Train multiple ML models"""
        print("\nModel Training:")
        
        # Define models with regularization to prevent overfitting
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True,
                C=1.0
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
        }

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"Training {name}...")

            # Use scaled data for linear models
            if name in ['SVM', 'Logistic Regression']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }

            self.results[name] = metrics
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

    def test_model_predictions(self):
        """Test model with known cases"""
        print("\nModel Validation Tests:")
        
        # Test cases: [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]
        test_cases = {
            'Low Risk Case': [1, 95, 70, 22, 85, 22.8, 0.25, 28],
            'High Risk Case': [5, 140, 95, 35, 200, 32.0, 1.2, 55]
        }
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['roc_auc'])
        best_model = self.models[best_model_name]
        
        for case_name, features in test_cases.items():
            input_data = np.array([features])
            
            if best_model_name in ['SVM', 'Logistic Regression']:
                input_scaled = self.scaler.transform(input_data)
                prediction = best_model.predict(input_scaled)[0]
                probability = best_model.predict_proba(input_scaled)[0]
            else:
                prediction = best_model.predict(input_data)[0]
                probability = best_model.predict_proba(input_data)[0]
            
            risk_prob = probability[1] * 100
            print(f"{case_name}: {risk_prob:.1f}% risk probability")

    def evaluate_models(self):
        """Evaluate and compare model performance"""
        print("\nModel Evaluation:")
        
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}")
        print("-" * 70)

        best_roc_auc = 0
        best_model_name = ""

        for name, metrics in self.results.items():
            print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['roc_auc']:<10.3f}")

            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_model_name = name

        print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_roc_auc:.3f})")
        self.best_model = self.models[best_model_name]

        return best_model_name

    def save_models(self):
        """Save trained models and metadata"""
        print("\nSaving Models:")
        
        # Save best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['roc_auc'])
        joblib.dump(self.best_model, 'diabetes_predection_model/best_model.pkl')
        print(f"Best model ({best_model_name}) saved")

        # Save scaler
        joblib.dump(self.scaler, 'diabetes_predection_model/scaler.pkl')
        print("Scaler saved")

        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'best_roc_auc': self.results[best_model_name]['roc_auc'],
            'feature_names': self.feature_names,
            'model_results': self.results
        }

        with open('diabetes_predection_model/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('results/model_performance.csv')
        print("Results saved")

    def create_plots(self):
        """Create evaluation plots"""
        plt.style.use('default')
        
        # Feature distributions
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.feature_names, 1):
            plt.subplot(2, 4, i)
            self.data[feature].hist(bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'{feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ROC curves
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for name, model in self.models.items():
            if name in ['SVM', 'Logistic Regression']:
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = self.results[name]['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()

        # Feature importance
        plt.subplot(2, 2, 2)
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.bar(range(len(importances)), importances[indices])
        plt.title('Feature Importance')
        plt.xticks(range(len(importances)),
                  [self.feature_names[i] for i in indices], rotation=45)

        plt.tight_layout()
        plt.savefig('plots/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("DIABETES PREDICTION MODEL TRAINING")
        print("=" * 50)

        # Load and preprocess data
        self.load_data()
        self.preprocess_data()

        # Prepare features and train models
        self.prepare_features()
        self.train_models()

        # Evaluate and test
        best_model_name = self.evaluate_models()
        self.test_model_predictions()

        # Save everything
        self.create_plots()
        self.save_models()

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Best Model: {best_model_name}")
        print("Files saved in: diabetes_predection_model/, plots/, results/")
        print("\nRun the frontend: streamlit run app.py")

def main():
    trainer = DiabetesMLTrainer()
    trainer.run_complete_training()

if __name__ == "__main__":
    main()