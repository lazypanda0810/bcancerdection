"""
Test Script for Breast Cancer Detection Project
==============================================
This script demonstrates the functionality of the trained model
"""

import joblib
import numpy as np
import requests
import json

def test_local_model():
    """Test the local model directly"""
    print("🧪 Testing Local Model...")
    
    # Load model components
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    model_info = joblib.load('model_info.pkl')
    
    print(f"✅ Model loaded: {model_info['model_name']}")
    print(f"✅ Accuracy: {model_info['accuracy']:.4f}")
    print(f"✅ Features: {len(feature_names)}")
    
    # Test with sample data (malignant case)
    malignant_sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    
    # Make prediction
    features_scaled = scaler.transform(np.array(malignant_sample).reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    diagnosis = "Malignant" if prediction == 1 else "Benign"
    print(f"🎯 Prediction: {diagnosis}")
    print(f"📊 Probability: {probability:.4f}")
    print(f"✅ Expected: Malignant (this is a known malignant sample)")
    print()

def test_api_endpoint():
    """Test the Flask API endpoint"""
    print("🌐 Testing Flask API...")
    
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:5000/health")
        if health_response.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return
        
        # Test prediction endpoint
        malignant_sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        
        prediction_data = {"features": malignant_sample}
        response = requests.post("http://localhost:5000/predict", json=prediction_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 API Prediction: {result['diagnosis']}")
            print(f"📊 API Probability: {result['probability']:.4f}")
            print(f"🔒 API Confidence: {result['confidence']:.4f}")
            print(f"⚠️ Risk Level: {result['risk_level']}")
            print("✅ API test successful")
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Flask app is running on localhost:5000")
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print()

def main():
    print("🩺 BREAST CANCER DETECTION PROJECT TEST")
    print("=" * 50)
    print()
    
    # Test local model
    test_local_model()
    
    # Test API
    test_api_endpoint()
    
    print("🎉 All tests completed!")
    print()
    print("📍 Access Points:")
    print("   Streamlit App: http://localhost:8501")
    print("   Flask API: http://localhost:5000")
    print("   HTML Frontend: Open frontend/index.html in browser")

if __name__ == "__main__":
    main()
