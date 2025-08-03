# 🩺 Breast Cancer Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/lazypanda0810/bcancerdection)

## 🧠 **Project Overview**

This is a comprehensive machine learning project that implements an AI-powered diagnostic tool for breast cancer detection. The system uses multiple machine learning algorithms to predict whether a breast tumor is **benign** (non-cancerous) or **malignant** (cancerous) based on clinical diagnostic features.

**🎯 Achieved Accuracy: 98.25%** with Logistic Regression model.

## 🏥 **Medical Context**

Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate detection is crucial for successful treatment. This AI system assists healthcare professionals by analyzing digitized features of breast mass cells to provide rapid, accurate predictions.

## 📊 **Dataset Information**

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Origin**: University of Wisconsin Hospitals, Madison
- **Features**: 30 numerical features computed from digitized images
- **Samples**: 569 patient records
- **Classes**: Binary (Benign: 357, Malignant: 212)
- **Feature Categories**:
  - **Mean values**: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
  - **Standard error values**: SE of the above 10 features  
  - **Worst values**: Largest values of the above 10 features

## 🏗️ **Project Architecture**

```
📁 Breast Cancer Detection/
├── 🤖 model_training.py          # ML pipeline & model training
├── 🎨 streamlit_app.py           # Interactive web application  
├── 🧪 test_project.py            # Comprehensive testing suite
├── 📊 *.pkl                      # Trained model artifacts
├── 📈 *.png                      # Data visualizations
├── 📋 requirements.txt           # Python dependencies
├── ⚙️ setup.bat                  # Windows setup script
├── 📖 README.md                  # Project documentation
├── 📁 backend/
│   └── ⚡ app.py                # Flask REST API
├── 📁 frontend/
│   └── 💻 index.html            # HTML web interface
└── 📁 .venv/                     # Virtual environment
```

## 🚀 **Quick Start Guide**

### **Option 1: Automatic Setup (Windows)**
```bash
# Double-click or run in Command Prompt
setup.bat
```

### **Option 2: Manual Setup**

#### **Step 1: Clone & Navigate**
```bash
git clone https://github.com/lazypanda0810/bcancerdection.git
cd bcancerdection
```

#### **Step 2: Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **Step 4: Train the Model**
```bash
python model_training.py
```

#### **Step 5: Run Applications**

**🎨 Streamlit App (Recommended)**
```bash
streamlit run streamlit_app.py
# Opens automatically at: http://localhost:8501
```

**⚡ Flask API**
```bash
cd backend
python app.py
# Available at: http://localhost:5000
```

**💻 HTML Frontend**
```bash
# Open frontend/index.html in your browser
# Requires Flask API to be running
```

## 🤖 **Machine Learning Pipeline**

### **Algorithms Implemented**
1. **🏆 Logistic Regression** - Best Performance (98.25% accuracy)
2. **🔍 Support Vector Machine** - High precision classifier  
3. **🧠 Neural Network (MLP)** - Deep learning approach
4. **🌳 Random Forest** - Ensemble method
5. **📊 K-Nearest Neighbors** - Instance-based learning
6. **🌿 Decision Tree** - Interpretable model

### **Model Evaluation Metrics**
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate (cancer detection reliability)
- **Recall**: Sensitivity (cancer case detection rate)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

### **Cross-Validation**
- 5-fold cross-validation for robust performance estimation
- Stratified sampling to maintain class distribution

## 🌐 **Web Applications**

### **1. 🎨 Streamlit Application**
**Features:**
- Interactive web interface with modern UI
- Real-time predictions with confidence gauges
- Model performance dashboard
- Sample data loading buttons
- Comprehensive visualizations
- Educational information panels

**Access:** http://localhost:8501

### **2. ⚡ Flask REST API**
**Endpoints:**
- `GET /` - API documentation homepage
- `GET /health` - System health check
- `POST /predict` - Make cancer prediction
- `GET /features` - Get required feature list
- `GET /model-info` - Model performance metrics

**Features:**
- RESTful architecture
- JSON request/response format
- Comprehensive error handling
- CORS enabled for web integration
- Input validation and sanitization

**Access:** http://localhost:5000

### **3. 💻 HTML Frontend**
**Features:**
- Professional responsive design
- 30-feature input form with validation
- Real-time prediction display
- Sample data presets (benign/malignant)
- Beautiful animations and styling
- Mobile-friendly interface

## 📊 **Model Performance Results**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **🏆 Logistic Regression** | **98.25%** | **98.61%** | **98.61%** | **98.61%** | **99.54%** |
| Support Vector Machine | 98.25% | 98.61% | 98.61% | 98.61% | 99.50% |
| Neural Network | 96.49% | 98.57% | 95.83% | 97.18% | 99.40% |
| K-Nearest Neighbors | 95.61% | 95.89% | 97.22% | 96.55% | 97.88% |
| Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 99.39% |
| Decision Tree | 91.23% | 95.59% | 90.28% | 92.86% | 91.57% |

### **Clinical Significance**
- **High Sensitivity**: 98.61% - Excellent at detecting malignant cases
- **High Specificity**: 98.25% - Minimizes false positive diagnoses  
- **Low False Negative Rate**: Critical for cancer detection
- **ROC-AUC > 99%**: Outstanding discriminative ability

## 🧪 **Testing & Validation**

### **Run Comprehensive Tests**
```bash
python test_project.py
```

**Test Coverage:**
- Model loading and prediction accuracy
- API endpoint functionality  
- Input validation and error handling
- Performance benchmarking
- Integration testing

## 🎓 **Educational Value**

### **Perfect for Academic Projects (BCA/NTCC)**

**Demonstrates Proficiency In:**
- **Machine Learning**: Algorithm selection, training, evaluation
- **Data Science**: EDA, preprocessing, feature engineering
- **Web Development**: Full-stack application development
- **API Design**: RESTful services and documentation
- **Software Engineering**: Testing, documentation, deployment
- **Healthcare AI**: Real-world medical applications

**Learning Outcomes:**
- Understanding of classification algorithms
- Experience with scikit-learn ecosystem
- Web framework knowledge (Flask, Streamlit)
- API development and integration
- Data visualization techniques
- Model deployment strategies

## 🔧 **Technical Requirements**

### **System Requirements**
- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 500MB free space

### **Python Dependencies**
```txt
scikit-learn>=1.2.0    # Machine learning algorithms
pandas>=2.0.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
matplotlib>=3.5.0      # Static plotting
seaborn>=0.11.0        # Statistical visualization
flask>=2.0.0           # Web framework
flask-cors>=4.0.0      # Cross-origin resource sharing
streamlit>=1.20.0      # Interactive web apps
joblib>=1.1.0          # Model serialization
plotly>=5.0.0          # Interactive visualizations
```

## 📱 **Usage Examples**

### **Making Predictions via Python**
```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Example patient data (30 features)
patient_data = [13.54, 14.36, 87.46, 566.3, 0.09779, ...]

# Make prediction
scaled_data = scaler.transform(np.array(patient_data).reshape(1, -1))
prediction = model.predict(scaled_data)[0]
probability = model.predict_proba(scaled_data)[0][1]

print(f"Diagnosis: {'Malignant' if prediction == 1 else 'Benign'}")
print(f"Confidence: {probability:.4f}")
```

### **API Request Example**
```python
import requests
import json

url = "http://localhost:5000/predict"
data = {
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, ...]
}

response = requests.post(url, json=data)
result = response.json()

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Risk Level: {result['risk_level']}")
```

## 🚀 **Deployment Options**

### **Local Development**
- Streamlit: `streamlit run streamlit_app.py`
- Flask: `python backend/app.py`

### **Cloud Deployment**
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Flask app deployment
- **AWS EC2**: Custom server deployment
- **Docker**: Containerized deployment

### **Production Considerations**
- Use production WSGI server (Gunicorn/uWSGI)
- Implement proper logging and monitoring
- Add authentication and authorization
- Set up SSL certificates
- Configure load balancing for high traffic

## ⚠️ **Important Medical Disclaimer**

**This system is designed for educational and research purposes only.**

- 🚫 **NOT** a substitute for professional medical diagnosis
- 🚫 **NOT** intended for direct clinical use without validation
- 🩺 Always consult qualified healthcare professionals
- 📋 Requires proper clinical validation before medical use
- ⚖️ Healthcare regulations must be followed for any clinical application

## 👥 **Contributing**

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Dataset**: UCI Machine Learning Repository
- **Original Research**: W.N. Street, W.H. Wolberg and O.L. Mangasarian
- **Libraries**: Scikit-learn, Flask, Streamlit, and the Python community
- **Inspiration**: Healthcare AI and early cancer detection research

## 📞 **Support & Contact**

For questions, issues, or contributions:

- 📧 **Email**: [Your Email]
- 💬 **Issues**: GitHub Issues page
- 📚 **Documentation**: Check the wiki
- 🆘 **Support**: Open a support ticket

---

**⭐ If this project helped you, please consider giving it a star!**

**🎯 Built with ❤️ for healthcare AI and machine learning education**
