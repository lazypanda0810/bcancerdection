"""
Flask Backend API for Breast Cancer Detection
============================================
This Flask application provides a REST API for breast cancer prediction
using the trained machine learning model.
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for model components
model = None
scaler = None
feature_names = None
model_info = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, scaler, feature_names, model_info
    
    try:
        # Load model components
        model_path = os.path.join('..', 'model.pkl')
        scaler_path = os.path.join('..', 'scaler.pkl')
        features_path = os.path.join('..', 'feature_names.pkl')
        info_path = os.path.join('..', 'model_info.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        model_info = joblib.load(info_path)
        
        print("✅ Model components loaded successfully!")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Accuracy: {model_info['accuracy']:.4f}")
        print(f"   Features: {model_info['feature_count']}")
        
        return True
    except FileNotFoundError as e:
        print(f"❌ Error loading model components: {e}")
        print("Please run model_training.py first to train and save the model.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# Load model components on startup
model_loaded = load_model_components()

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breast Cancer Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            code { background: #f8f9fa; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🩺 Breast Cancer Detection API</h1>
                <p>AI-powered breast cancer prediction service</p>
            </div>
            
            {% if model_status %}
            <div class="status success">
                <strong>✅ Model Status:</strong> Loaded and Ready<br>
                <strong>Model:</strong> {{ model_info.model_name }}<br>
                <strong>Accuracy:</strong> {{ "%.2f"|format(model_info.accuracy * 100) }}%<br>
                <strong>Features:</strong> {{ model_info.feature_count }}
            </div>
            {% else %}
            <div class="status error">
                <strong>❌ Model Status:</strong> Not Loaded<br>
                Please run <code>python model_training.py</code> first to train the model.
            </div>
            {% endif %}
            
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/</strong><br>
                <em>This page - API documentation</em>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong><br>
                <em>Check API health status</em>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/predict</strong><br>
                <em>Make breast cancer prediction</em><br><br>
                <strong>Request Body (JSON):</strong><br>
                <code>{"features": [30 numerical values]}</code><br><br>
                <strong>Response:</strong><br>
                <code>{"prediction": 0 or 1, "probability": float, "diagnosis": "Benign" or "Malignant"}</code>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/features</strong><br>
                <em>Get list of required feature names</em>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/model-info</strong><br>
                <em>Get model information and performance metrics</em>
            </div>
            
            <h2>🧪 Example Usage</h2>
            <pre style="background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Python example
import requests
import json

url = "http://localhost:5000/predict"
data = {
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}

response = requests.post(url, json=data)
result = response.json()
print(f"Diagnosis: {result['diagnosis']}")
print(f"Probability: {result['probability']:.4f}")
            </pre>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                model_status=model_loaded, 
                                model_info=model_info if model_loaded else None)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_loaded:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "message": "API is running and model is loaded"
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "message": "Model not loaded. Please train the model first."
        }), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction endpoint"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "Please run model_training.py first to train the model"
        }), 503
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                "error": "Invalid input",
                "message": "Please provide 'features' in JSON format"
            }), 400
        
        features = data['features']
        
        # Validate feature count
        if len(features) != len(feature_names):
            return jsonify({
                "error": "Invalid feature count",
                "message": f"Expected {len(feature_names)} features, got {len(features)}",
                "expected_features": len(feature_names)
            }), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]  # Probability of malignant
        
        # Convert prediction to diagnosis
        diagnosis = "Malignant" if prediction == 1 else "Benign"
        confidence = probability if prediction == 1 else (1 - probability)
        
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": float(confidence),
            "diagnosis": diagnosis,
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
        })
        
    except ValueError as e:
        return jsonify({
            "error": "Invalid feature values",
            "message": f"All features must be numeric: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get feature names endpoint"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "Please run model_training.py first"
        }), 503
    
    return jsonify({
        "features": feature_names,
        "feature_count": len(feature_names),
        "description": "List of required features for prediction"
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get model information endpoint"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "Please run model_training.py first"
        }), 503
    
    return jsonify({
        "model_name": model_info['model_name'],
        "accuracy": model_info['accuracy'],
        "f1_score": model_info['f1_score'],
        "roc_auc": model_info['roc_auc'],
        "feature_count": model_info['feature_count'],
        "training_samples": model_info['training_samples'],
        "test_samples": model_info['test_samples']
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/predict", "/features", "/model-info"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Breast Cancer Detection API...")
    print("📍 API will be available at: http://localhost:5000")
    print("📖 Visit http://localhost:5000 for API documentation")
    
    if not model_loaded:
        print("⚠️  Warning: Model not loaded. Some endpoints will not work.")
        print("   Please run 'python model_training.py' first.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
