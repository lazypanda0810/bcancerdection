@echo off
echo ========================================
echo Breast Cancer Detection Setup Script
echo ========================================
echo.

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Training the machine learning model...
python model_training.py

echo.
echo Setup complete! 
echo.
echo To run the application, choose one of:
echo 1. Streamlit App: streamlit run streamlit_app.py
echo 2. Flask API: cd backend && python app.py
echo 3. Open frontend\index.html in your browser (after starting Flask API)
echo.
pause
