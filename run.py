import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# --- CSS for dark/light theme and fixed navbar ---
def load_css():
    return """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-color: #6366f1;
        --primary-hover: #4f46e5;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --bg-primary: #111827;
        --bg-secondary: #1f2937;
        --bg-card: #374151;
        --border-color: #4b5563;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 1px 2px 0 rgba(0, 0, 0, 0.2);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
        --transition: all 0.3s ease;
    }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* App styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: var(--transition);
    }

    /* Fixed top navbar */
    .top-navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 60px;
        background-color: var(--bg-secondary);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        box-shadow: var(--shadow-lg);
        z-index: 9999;
        user-select: none;
    }

    .app-title {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: -0.02em;
    }

    .nav-tabs {
        display: flex;
        gap: 1rem;
    }

    .nav-tab {
        background: transparent;
        border: none;
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 1rem;
        cursor: pointer;
        padding: 0.5rem 1.25rem;
        border-radius: 0.5rem;
        transition: var(--transition);
        outline: none;
    }

    .nav-tab:hover {
        background-color: var(--bg-card);
        color: var(--primary-color);
        box-shadow: var(--shadow);
    }

    .nav-tab.active {
        background-color: var(--primary-color);
        color: var(--bg-primary);
        box-shadow: var(--shadow-lg);
    }

    /* Add top padding so content not hidden behind navbar */
    .stApp > main {
        padding-top: 70px !important;
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }

    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--text-primary);
    }

    /* Prediction result boxes */
    .prediction-result {
        text-align: center;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        animation: slideUp 0.6s ease;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-high-risk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
        border: 2px solid rgba(239, 68, 68, 0.3);
        color: var(--danger-color);
        font-weight: 700;
    }

    .result-low-risk {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
        border: 2px solid rgba(16, 185, 129, 0.3);
        color: var(--success-color);
        font-weight: 700;
    }

    .result-title {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }

    .result-percentage {
        font-size: 3rem;
        margin: 0.25rem 0;
    }

    .result-confidence {
        font-size: 1rem;
        opacity: 0.8;
    }

    /* Risk factors */
    .risk-factor {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid;
        animation: slideIn 0.4s ease;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .risk-high {
        background: rgba(239, 68, 68, 0.1);
        border-color: var(--danger-color);
        color: var(--danger-color);
    }

    .risk-medium {
        background: rgba(245, 158, 11, 0.1);
        border-color: var(--warning-color);
        color: var(--warning-color);
    }

    .risk-low {
        background: rgba(16, 185, 129, 0.1);
        border-color: var(--success-color);
        color: var(--success-color);
    }

    /* Responsive */
    @media (max-width: 768px) {
        .top-navbar {
            flex-direction: column;
            height: auto;
            padding: 1rem 2rem;
            gap: 0.5rem;
        }
        .nav-tabs {
            justify-content: center;
            flex-wrap: wrap;
            width: 100%;
        }
    }
    </style>
    """

# Apply the CSS and set default dark theme
def apply_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    # Force dark theme vars
    st.markdown(f"""
    <script>
        document.documentElement.setAttribute('data-theme', '{st.session_state.theme}');
    </script>
    """, unsafe_allow_html=True)

class DiabetesMLSystem:
    def __init__(self):
        self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        self.model = None
        self.scaler = None
        self.model_metadata = {}
        self.model_type = "Not Available"
        for directory in ['models', 'data', 'results', 'plots']:
            Path(directory).mkdir(exist_ok=True)
    
    def create_synthetic_data(self):
        np.random.seed(42)
        n_samples = 768
        age = np.random.choice(range(21, 81), size=n_samples, p=self.get_age_probabilities())
        pregnancies = np.array([max(0, np.random.poisson(2 if a < 35 else 3)) for a in age])
        bmi = np.clip(np.random.normal(25 + (age - 30) * 0.1, 6), 15, 50)
        glucose = np.clip(85 + (age - 21) * 0.3 + (bmi - 25) * 0.8 + np.random.normal(0, 15, n_samples), 60, 200)
        blood_pressure = np.clip(65 + (age - 21) * 0.2 + (bmi - 25) * 0.3 + np.random.normal(0, 8, n_samples), 50, 120)
        skin_thickness = np.clip(15 + (bmi - 25) * 0.4 + np.random.normal(0, 5, n_samples), 5, 50)
        insulin = np.array([0 if np.random.random() < 0.3 else min(int(np.random.lognormal(4, 0.8)), 700) for _ in range(n_samples)])
        dpf = np.clip(np.random.gamma(2, 0.3, n_samples), 0.1, 2.5)
        
        risk = (
            np.where(glucose >= 126, 3.0, np.where(glucose >= 100, 1.5, 0.0)) +
            np.where(bmi >= 35, 2.0, np.where(bmi >= 30, 1.2, np.where(bmi >= 25, 0.3, 0.0))) +
            np.where(age >= 65, 1.5, np.where(age >= 45, 0.8, np.where(age >= 35, 0.3, 0.0))) +
            np.where(blood_pressure >= 90, 0.5, 0.0) +
            np.where(pregnancies >= 5, 0.4, np.where(pregnancies >= 3, 0.2, 0.0)) +
            dpf * 0.8 +
            np.where(insulin > 200, 0.3, 0.0) +
            np.random.normal(0, 0.2, n_samples)
        )
        
        threshold = np.percentile(risk, 65)
        outcome = (risk > threshold).astype(int)
        
        data = pd.DataFrame({
            'Pregnancies': pregnancies.astype(int),
            'Glucose': np.round(glucose).astype(int),
            'BloodPressure': np.round(blood_pressure).astype(int),
            'SkinThickness': np.round(skin_thickness).astype(int),
            'Insulin': insulin.astype(int),
            'BMI': np.round(bmi, 1),
            'DiabetesPedigreeFunction': np.round(dpf, 3),
            'Age': age.astype(int),
            'Outcome': outcome
        })
        data.to_csv('data/diabetes_synthetic.csv', index=False)
        return data
    
    def get_age_probabilities(self):
        ages = range(21, 81)
        probs = [np.exp(-0.5 * ((age - 45) / 15) ** 2) for age in ages]
        total = sum(probs)
        return [p / total for p in probs]
    
    def train_models(self, data):
        X = data[self.feature_names]
        y = data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'SVM': SVC(random_state=42, probability=True, C=1.0),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
        }
        
        results = {}
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            results[name] = metrics
            
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model
                self.model_type = name
        
        self.model = best_model
        joblib.dump(best_model, 'diabetes_predection_model/best_model.pkl')
        joblib.dump(self.scaler, 'diabetes_predection_model/scaler.pkl')
        
        self.model_metadata = {
            'best_model': self.model_type,
            'best_roc_auc': best_score,
            'feature_names': self.feature_names,
            'model_results': results
        }
        with open('diabetes_predection_model/model_metadata.json', 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        return results
    
    def load_models(self):
        try:
            if os.path.exists('diabetes_predection_model/best_model.pkl'):
                self.model = joblib.load('diabetes_predection_model/best_model.pkl')
                self.model_type = type(self.model).__name__
                
                if os.path.exists('diabetes_predection_model/scaler.pkl'):
                    self.scaler = joblib.load('diabetes_predection_model/scaler.pkl')
                
                if os.path.exists('diabetes_predection_model/model_metadata.json'):
                    with open('diabetes_predection_model/model_metadata.json', 'r') as f:
                        self.model_metadata = json.load(f)
                
                return True
            return False
        except Exception:
            return False
    
    def predict_risk(self, input_data):
        try:
            input_array = np.array(input_data).reshape(1, -1)
            
            # Models needing scaling by name
            needs_scaling = self.model_type in ['SVC', 'LogisticRegression']
            if needs_scaling and self.scaler:
                input_processed = self.scaler.transform(input_array)
            else:
                input_processed = input_array
            
            prediction = self.model.predict(input_processed)[0]
            probabilities = self.model.predict_proba(input_processed)[0]
            
            return prediction, probabilities
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

@st.cache_resource
def get_ml_system():
    return DiabetesMLSystem()

def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="⚕️", layout="wide")
    
    st.markdown(load_css(), unsafe_allow_html=True)
    apply_theme()

    ml_system = get_ml_system()

    # Navbar with app name left and tabs right
    st.markdown(f"""
    <div class="top-navbar">
        <div class="app-title">Diabetes Risk Predictor</div>
        <nav class="nav-tabs">
            <button class="nav-tab {'active' if st.session_state.get('page','prediction')=='prediction' else ''}" id="tab-prediction">Risk Prediction</button>
            <button class="nav-tab {'active' if st.session_state.get('page','prediction')=='analytics' else ''}" id="tab-analytics">Model Analytics</button>
            <button class="nav-tab {'active' if st.session_state.get('page','prediction')=='insights' else ''}" id="tab-insights">Data Insights</button>
        </nav>
    </div>
    <script>
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {{
        tab.onclick = () => {{
            window.parent.postMessage({{func: "setPage", page: tab.id.replace('tab-', '')}}, "*");
        }};
    }});
    </script>
    """, unsafe_allow_html=True)

    # JS message handler to switch page state in Streamlit
    if "page" not in st.session_state:
        st.session_state.page = "prediction"

    # Listen for postMessage from JS and update page state
    # (Streamlit cannot directly listen to JS events, so workaround needed)
    # For now, we use buttons below as fallback for older browsers
    
    # Buttons fallback for accessibility
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Risk Prediction", use_container_width=True):
            st.session_state.page = "prediction"
    with col2:
        if st.button("Model Analytics", use_container_width=True):
            st.session_state.page = "analytics"
    with col3:
        if st.button("Data Insights", use_container_width=True):
            st.session_state.page = "insights"
    
    # If model not loaded, train it first
    if not ml_system.load_models():
        st.markdown("""
        <div class="card">
            <div class="card-header">Initial Setup Required</div>
            <div class="risk-factor risk-medium">
                Training machine learning models...
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Training models... please wait..."):
            data = ml_system.create_synthetic_data()
            ml_system.train_models(data)
        st.experimental_rerun()

    roc_auc = ml_system.model_metadata.get('best_roc_auc', 0)
    st.markdown(f"""
    <div class="card">
        <div class="risk-factor risk-low">
            Model ready: {ml_system.model_type} | ROC-AUC: {roc_auc:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Page routing
    if st.session_state.page == "prediction":
        show_prediction_page(ml_system)
    elif st.session_state.page == "analytics":
        show_analytics_page(ml_system)
    elif st.session_state.page == "insights":
        show_insights_page(ml_system)

def show_prediction_page(ml_system):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card"><div class="card-header">Patient Information</div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose (mg/dL)", 50, 300, 95)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 40, 200, 70)
            skin_thickness = st.number_input("Skin Thickness (mm)", 5, 60, 22)
            insulin = st.number_input("Insulin (μU/mL)", 0, 800, 85)
            bmi = st.number_input("BMI (kg/m²)", 15.0, 60.0, 22.8, 0.1)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.25, 0.01)
            age = st.number_input("Age (years)", 18, 100, 28)
            submitted = st.form_submit_button("Analyze Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if submitted:
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            prediction, probabilities = ml_system.predict_risk(input_data)
            if prediction is not None:
                risk_percentage = probabilities[1] * 100
                confidence = max(probabilities) * 100
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-result result-high-risk">
                        <div class="result-title">High Risk</div>
                        <div class="result-percentage">{risk_percentage:.1f}%</div>
                        <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result result-low-risk">
                        <div class="result-title">Low Risk</div>
                        <div class="result-percentage">{risk_percentage:.1f}%</div>
                        <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="card"><div class="card-header">Risk Factor Analysis</div>', unsafe_allow_html=True)
                factors = analyze_risk_factors(glucose, bmi, age, blood_pressure, dpf, pregnancies)
                for factor in factors:
                    st.markdown(f"""
                    <div class="risk-factor risk-{factor['level']}">
                        {factor['text']}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

def analyze_risk_factors(glucose, bmi, age, bp, dpf, pregnancies):
    factors = []
    if glucose >= 126:
        factors.append({"level": "high", "text": "Glucose in diabetic range (≥126 mg/dL)"})
    elif glucose >= 100:
        factors.append({"level": "medium", "text": "Glucose in prediabetic range (100-125 mg/dL)"})
    else:
        factors.append({"level": "low", "text": "Normal glucose levels (<100 mg/dL)"})
    if bmi >= 30:
        factors.append({"level": "high", "text": "Obesity (BMI ≥30)"})
    elif bmi >= 25:
        factors.append({"level": "medium", "text": "Overweight (BMI 25-29.9)"})
    else:
        factors.append({"level": "low", "text": "Normal BMI (<25)"})
    if age >= 45:
        factors.append({"level": "high", "text": "Age ≥45 years"})
    else:
        factors.append({"level": "low", "text": "Younger age (<45 years)"})
    if bp > 90:
        factors.append({"level": "high", "text": "High blood pressure (>90 mm Hg)"})
    else:
        factors.append({"level": "low", "text": "Normal blood pressure (≤90 mm Hg)"})
    if dpf > 1.0:
        factors.append({"level": "high", "text": "Strong family history"})
    elif dpf <= 0.5:
        factors.append({"level": "low", "text": "Low family history"})
    if pregnancies > 4:
        factors.append({"level": "medium", "text": "Multiple pregnancies (>4)"})
    return factors

def show_analytics_page(ml_system):
    st.markdown('<div class="card"><div class="card-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    if 'model_results' in ml_system.model_metadata:
        results_df = pd.DataFrame(ml_system.model_metadata['model_results']).T
        fig = go.Figure()
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            if metric in results_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=results_df.index,
                    y=results_df[metric],
                    text=results_df[metric].round(3),
                    textposition='auto'
                ))
        fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=500,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results_df.round(3), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if hasattr(ml_system.model, 'feature_importances_'):
        st.markdown('<div class="card"><div class="card-header">Feature Importance</div>', unsafe_allow_html=True)
        importance_df = pd.DataFrame({
            'Feature': ml_system.feature_names,
            'Importance': ml_system.model.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Feature Importance in Risk Assessment",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_insights_page(ml_system):
    if os.path.exists('data/diabetes_synthetic.csv'):
        data = pd.read_csv('data/diabetes_synthetic.csv')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", len(data))
        with col2:
            diabetic_count = data['Outcome'].sum()
            st.metric("Diabetic Patients", diabetic_count)
        with col3:
            diabetic_rate = (diabetic_count / len(data)) * 100
            st.metric("Diabetes Rate", f"{diabetic_rate:.1f}%")
        
        st.markdown('<div class="card"><div class="card-header">Feature Analysis</div>', unsafe_allow_html=True)
        selected_feature = st.selectbox(
            "Select feature to analyze:",
            [col for col in data.columns if col != 'Outcome']
        )
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                data, 
                x=selected_feature, 
                color='Outcome',
                title=f'Distribution of {selected_feature}',
                barmode='overlay',
                template='plotly_dark'
            )
            fig_hist.update_traces(opacity=0.7)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(
                data, 
                x='Outcome', 
                y=selected_feature,
                title=f'{selected_feature} by Diabetes Status',
                template='plotly_dark'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-header">Feature Correlations</div>', unsafe_allow_html=True)
        corr_matrix = data.corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            template='plotly_dark'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
