"""
Breast Cancer Detection Model Training Script
============================================
This script trains multiple machine learning models on the Breast Cancer Wisconsin dataset
and saves the best performing model for deployment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost model.")

class BreastCancerDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_explore_data(self):
        """Load and explore the breast cancer dataset"""
        print("🔍 Loading Breast Cancer Wisconsin Dataset...")
        
        # Load the dataset
        data = load_breast_cancer()
        self.X = pd.DataFrame(data.data, columns=data.feature_names)
        self.y = pd.Series(data.target)
        
        print(f"📊 Dataset Shape: {self.X.shape}")
        print(f"📈 Features: {len(self.X.columns)}")
        print(f"🎯 Target Distribution:")
        print(f"   Malignant (1): {sum(self.y == 1)} samples")
        print(f"   Benign (0): {sum(self.y == 0)} samples")
        
        # Basic statistics
        print("\n📋 Dataset Info:")
        print(f"   Missing values: {self.X.isnull().sum().sum()}")
        print(f"   Data types: {self.X.dtypes.value_counts().to_dict()}")
        
        return self.X, self.y
    
    def visualize_data(self):
        """Create visualizations for data exploration"""
        print("\n📊 Creating data visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Target distribution
        axes[0, 0].pie(self.y.value_counts(), labels=['Benign', 'Malignant'], 
                       autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Target Distribution')
        
        # 2. Feature correlation heatmap (top 10 features)
        corr_matrix = self.X.iloc[:, :10].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    ax=axes[0, 1], fmt='.2f')
        axes[0, 1].set_title('Feature Correlation (Top 10)')
        
        # 3. Box plot of mean features
        mean_features = [col for col in self.X.columns if 'mean' in col][:5]
        self.X[mean_features].boxplot(ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Mean Features')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Feature importance preview (using a quick random forest)
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(self.X, self.y)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[1, 1].set_title('Top 10 Most Important Features')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for training"""
        print("\n🛠️ Preprocessing data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {self.X_train_scaled.shape}")
        print(f"✅ Test set: {self.X_test_scaled.shape}")
        
    def initialize_models(self):
        """Initialize all machine learning models"""
        print("\n🤖 Initializing ML models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
        
        print(f"✅ Initialized {len(self.models)} models")
        
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n🚀 Training and evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            results[name] = {
                'Model': model,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std()
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   ✅ F1-Score: {f1:.4f}")
            print(f"   ✅ ROC-AUC: {roc_auc:.4f}")
        
        self.results = results
        return results
    
    def display_results(self):
        """Display model comparison results"""
        print("\n📊 MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            model_name: {
                'Accuracy': data['Accuracy'],
                'Precision': data['Precision'],
                'Recall': data['Recall'],
                'F1-Score': data['F1-Score'],
                'ROC-AUC': data['ROC-AUC'],
                'CV Mean': data['CV Mean'],
                'CV Std': data['CV Std']
            }
            for model_name, data in self.results.items()
        }).T
        
        # Sort by F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(results_df.round(4))
        
        # Find best model
        self.best_model_name = results_df.index[0]
        self.best_model = self.results[self.best_model_name]['Model']
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1-Score: {results_df.loc[self.best_model_name, 'F1-Score']:.4f}")
        print(f"   Accuracy: {results_df.loc[self.best_model_name, 'Accuracy']:.4f}")
        
    def plot_model_comparison(self):
        """Plot model comparison charts"""
        print("\n📈 Creating model comparison visualizations...")
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Individual metric comparisons
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            values = [self.results[name][metric] for name in model_names]
            
            bars = axes[row, col].bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
            axes[row, col].set_title(f'{metric} Comparison')
            axes[row, col].set_ylim(0, 1)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Overall comparison (radar chart style)
        ax = axes[1, 2]
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(['Accuracy', 'F1-Score', 'ROC-AUC']):
            values = [self.results[name][metric] for name in model_names]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Overall Model Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix for the best model"""
        print(f"\n🎯 Creating confusion matrix for {self.best_model_name}...")
        
        y_pred = self.best_model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print(f"\n📋 Classification Report for {self.best_model_name}:")
        print(classification_report(self.y_test, y_pred, target_names=['Benign', 'Malignant']))
        
    def save_model(self):
        """Save the best model and scaler"""
        print(f"\n💾 Saving the best model ({self.best_model_name})...")
        
        # Save model and scaler
        joblib.dump(self.best_model, 'model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        # Save feature names
        feature_names = list(self.X.columns)
        joblib.dump(feature_names, 'feature_names.pkl')
        
        # Save model metadata
        model_info = {
            'model_name': self.best_model_name,
            'accuracy': self.results[self.best_model_name]['Accuracy'],
            'f1_score': self.results[self.best_model_name]['F1-Score'],
            'roc_auc': self.results[self.best_model_name]['ROC-AUC'],
            'feature_count': len(feature_names),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        joblib.dump(model_info, 'model_info.pkl')
        
        print("✅ Model saved successfully!")
        print("   - model.pkl (trained model)")
        print("   - scaler.pkl (feature scaler)")
        print("   - feature_names.pkl (feature names)")
        print("   - model_info.pkl (model metadata)")

def main():
    """Main function to run the complete pipeline"""
    print("🎯 BREAST CANCER DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Initialize detector
    detector = BreastCancerDetector()
    
    # Load and explore data
    detector.load_and_explore_data()
    detector.visualize_data()
    
    # Preprocess data
    detector.preprocess_data()
    
    # Initialize and train models
    detector.initialize_models()
    detector.train_and_evaluate_models()
    
    # Display and visualize results
    detector.display_results()
    detector.plot_model_comparison()
    detector.plot_confusion_matrix()
    
    # Save the best model
    detector.save_model()
    
    print("\n✅ Model training completed successfully!")
    print("🚀 You can now run the Streamlit app: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
