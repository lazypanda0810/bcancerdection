"""
Streamlit Web Application for Breast Cancer Detection
===================================================
This is a user-friendly web interface built with Streamlit for breast cancer prediction.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from sklearn.datasets import load_breast_cancer

# Configure page
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-benign {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .result-malignant {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_components():
    """Load the trained model and preprocessing components"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, scaler, feature_names, model_info, True
    except FileNotFoundError:
        return None, None, None, None, False

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.Series(data.target, name='target')
    return df, target

def create_feature_importance_chart(model, feature_names):
    """Create feature importance visualization"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        fig = px.bar(importance_df, x='importance', y='feature', 
                     title="Top 15 Most Important Features",
                     color='importance',
                     color_continuous_scale='viridis')
        fig.update_layout(height=600)
        return fig
    return None

def create_prediction_gauge(probability, prediction):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Malignancy Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def main():
    # Load model components
    model, scaler, feature_names, model_info, model_loaded = load_model_components()
    
    # Main header
    st.markdown('<h1 class="main-header">🩺 Breast Cancer Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered diagnostic tool for breast tumor classification</p>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if not model_loaded:
        st.error("❌ **Model not found!** Please run `python model_training.py` first to train and save the model.")
        st.info("💡 **Quick Start:** Run the model training script to get started with predictions.")
        
        # Show sample data exploration while model is not available
        st.markdown("---")
        st.subheader("📊 Dataset Exploration")
        
        df_sample, target_sample = load_sample_data()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df_sample))
        with col2:
            st.metric("Features", len(df_sample.columns))
        with col3:
            st.metric("Benign Cases", sum(target_sample == 0))
        with col4:
            st.metric("Malignant Cases", sum(target_sample == 1))
        
        # Show sample data
        if st.checkbox("Show Sample Data"):
            st.dataframe(df_sample.head())
        
        return
    
    # Sidebar - Model Information
    with st.sidebar:
        st.markdown("## 🤖 Model Information")
        st.markdown(f"**Model:** {model_info['model_name']}")
        st.markdown(f"**Accuracy:** {model_info['accuracy']:.4f}")
        st.markdown(f"**F1-Score:** {model_info['f1_score']:.4f}")
        st.markdown(f"**ROC-AUC:** {model_info['roc_auc']:.4f}")
        
        st.markdown("---")
        st.markdown("## 📋 Quick Actions")
        
        # Sample data buttons
        if st.button("🔬 Load Benign Sample"):
            benign_sample = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
            for i, value in enumerate(benign_sample):
                st.session_state[f'feature_{i}'] = value
            st.success("Benign sample loaded!")
            
        if st.button("⚠️ Load Malignant Sample"):
            malignant_sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            for i, value in enumerate(malignant_sample):
                st.session_state[f'feature_{i}'] = value
            st.success("Malignant sample loaded!")
        
        if st.button("🗑️ Clear All Fields"):
            for i in range(len(feature_names)):
                if f'feature_{i}' in st.session_state:
                    del st.session_state[f'feature_{i}']
            st.success("All fields cleared!")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Analysis", "📖 Information"])
    
    with tab1:
        st.markdown("### 📋 Enter Patient Data")
        
        # Create input form
        with st.form("prediction_form"):
            # Organize features into groups
            mean_features = [f for f in feature_names if 'mean' in f]
            se_features = [f for f in feature_names if 'se' in f]
            worst_features = [f for f in feature_names if 'worst' in f]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Mean Values**")
                mean_values = []
                for i, feature in enumerate(mean_features):
                    idx = feature_names.index(feature)
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=st.session_state.get(f'feature_{idx}', 0.0),
                        key=f'feature_{idx}',
                        format="%.4f"
                    )
                    mean_values.append(value)
            
            with col2:
                st.markdown("**Standard Error Values**")
                se_values = []
                for feature in se_features:
                    idx = feature_names.index(feature)
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=st.session_state.get(f'feature_{idx}', 0.0),
                        key=f'feature_{idx}',
                        format="%.4f"
                    )
                    se_values.append(value)
            
            with col3:
                st.markdown("**Worst Values**")
                worst_values = []
                for feature in worst_features:
                    idx = feature_names.index(feature)
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=st.session_state.get(f'feature_{idx}', 0.0),
                        key=f'feature_{idx}',
                        format="%.4f"
                    )
                    worst_values.append(value)
            
            # Prediction button
            predict_button = st.form_submit_button("🔍 Analyze & Predict", use_container_width=True)
        
        # Make prediction
        if predict_button:
            # Collect all feature values
            features = []
            for i in range(len(feature_names)):
                features.append(st.session_state.get(f'feature_{i}', 0.0))
            
            # Validate input
            if all(f == 0.0 for f in features):
                st.warning("⚠️ Please enter some feature values before making a prediction.")
            else:
                # Make prediction
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0][1]
                confidence = probability if prediction == 1 else (1 - probability)
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown(f'''
                        <div class="result-malignant">
                            <h2>⚠️ MALIGNANT</h2>
                            <p>The tumor is predicted to be malignant (cancerous)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="result-benign">
                            <h2>✅ BENIGN</h2>
                            <p>The tumor is predicted to be benign (non-cancerous)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with col2:
                    # Gauge chart
                    fig_gauge = create_prediction_gauge(probability, prediction)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Prediction", "Malignant" if prediction == 1 else "Benign")
                with col2:
                    st.metric("Probability", f"{probability:.4f}")
                with col3:
                    st.metric("Confidence", f"{confidence:.4f}")
                with col4:
                    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Additional information
                st.markdown("---")
                st.markdown("### 📋 Clinical Recommendation")
                
                if prediction == 1:
                    st.error("🚨 **Immediate Action Required:** This result suggests malignancy. Please consult with an oncologist immediately for further evaluation and treatment planning.")
                else:
                    st.success("✅ **Good News:** This result suggests the tumor is benign. However, regular monitoring and follow-up with your healthcare provider is still recommended.")
                
                st.info("💡 **Important Note:** This AI prediction is a diagnostic aid and should not replace professional medical judgment. Always consult with qualified healthcare professionals for proper diagnosis and treatment.")
    
    with tab2:
        st.markdown("### 📊 Model Performance Analysis")
        
        # Feature importance
        fig_importance = create_feature_importance_chart(model, feature_names)
        if fig_importance:
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance visualization is only available for tree-based models.")
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>Accuracy</h3>
                <h2>{model_info['accuracy']:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>F1-Score</h3>
                <h2>{model_info['f1_score']:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <h2>{model_info['roc_auc']:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{model_info['feature_count']}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        # Training information
        st.markdown("### 🏋️ Training Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", model_info['training_samples'])
        with col2:
            st.metric("Test Samples", model_info['test_samples'])
    
    with tab3:
        st.markdown("### 📖 About This Application")
        
        st.markdown("""
        #### 🎯 **Purpose**
        This application uses machine learning to predict whether a breast tumor is benign (non-cancerous) or malignant (cancerous) based on clinical features extracted from diagnostic images.
        
        #### 📊 **Dataset**
        - **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset
        - **Features:** 30 numerical features computed from digitized images of breast mass
        - **Samples:** 569 samples (357 benign, 212 malignant)
        
        #### 🧠 **Model Features**
        The model analyzes 30 different characteristics of cell nuclei:
        
        **Mean Values:**
        - Radius, Texture, Perimeter, Area, Smoothness
        - Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
        
        **Standard Error (SE) Values:**
        - Standard error of the above 10 features
        
        **Worst Values:**
        - Largest (worst) values of the above 10 features
        
        #### 🔬 **How It Works**
        1. **Feature Extraction:** Clinical features are extracted from breast mass images
        2. **Preprocessing:** Features are standardized using the same scaler used during training
        3. **Prediction:** The trained ML model analyzes the features and makes a prediction
        4. **Result:** Binary classification (Benign/Malignant) with confidence probability
        
        #### ⚠️ **Important Disclaimer**
        This tool is for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for proper medical evaluation and treatment.
        
        #### 🏆 **Model Performance**
        - High accuracy on test data
        - Validated using cross-validation
        - Optimized for both sensitivity and specificity
        
        #### 👥 **Perfect for Academic Projects**
        This project demonstrates:
        - Machine Learning classification
        - Data preprocessing and feature engineering
        - Model evaluation and validation
        - Web application development
        - Medical AI applications
        """)

if __name__ == "__main__":
    main()
