# ==============================================
# Enhanced Credit Card Fraud Detection Hybrid Model
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering for credit card transactions"""
    
    def __init__(self):
        self.amount_stats = {}
        self.time_stats = {}
        
    def fit(self, X, y=None):
        # Calculate statistics for feature engineering
        if 'Amount' in X.columns:
            self.amount_stats['mean'] = X['Amount'].mean()
            self.amount_stats['std'] = X['Amount'].std()
            self.amount_stats['median'] = X['Amount'].median()
            
        if 'Time' in X.columns:
            self.time_stats['mean'] = X['Time'].mean()
            self.time_stats['std'] = X['Time'].std()
            
        return self
        
    def transform(self, X):
        X_new = X.copy()
        
        # Amount-based features
        if 'Amount' in X_new.columns:
            X_new['Amount_log'] = np.log1p(X_new['Amount'])
            X_new['Amount_sqrt'] = np.sqrt(X_new['Amount'])
            X_new['Amount_zscore'] = (X_new['Amount'] - self.amount_stats['mean']) / self.amount_stats['std']
            X_new['Amount_vs_median'] = X_new['Amount'] / self.amount_stats['median']
            X_new['Amount_high'] = (X_new['Amount'] > X_new['Amount'].quantile(0.95)).astype(int)
            X_new['Amount_low'] = (X_new['Amount'] < X_new['Amount'].quantile(0.05)).astype(int)
            
        # Time-based features
        if 'Time' in X_new.columns:
            X_new['Hour'] = (X_new['Time'] % 86400) // 3600
            X_new['Day_of_week'] = (X_new['Time'] // 86400) % 7
            X_new['Is_weekend'] = (X_new['Day_of_week'] >= 5).astype(int)
            X_new['Is_night'] = ((X_new['Hour'] >= 22) | (X_new['Hour'] <= 6)).astype(int)
            X_new['Is_business_hours'] = ((X_new['Hour'] >= 9) & (X_new['Hour'] <= 17)).astype(int)
            
        # V-feature interactions (assuming V1-V28 exist)
        v_columns = [col for col in X_new.columns if col.startswith('V')]
        if len(v_columns) >= 2:
            X_new['V_sum'] = X_new[v_columns].sum(axis=1)
            X_new['V_mean'] = X_new[v_columns].mean(axis=1)
            X_new['V_std'] = X_new[v_columns].std(axis=1)
            X_new['V_max'] = X_new[v_columns].max(axis=1)
            X_new['V_min'] = X_new[v_columns].min(axis=1)
            
        return X_new

class HybridFraudDetector:
    """Advanced hybrid fraud detection system"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.models = {}
        self.ensemble = None
        self.is_trained = False
        self.feature_importance = {}
        
    def initialize_models(self):
        """Initialize all models with optimized hyperparameters"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                learning_rate=0.1,
                max_depth=6,
                n_estimators=150,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=300,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'logistic': LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                random_state=42,
                max_iter=500
            ),
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
    def preprocess_data(self, X, y=None, fit_transform=True):
        """Advanced preprocessing pipeline"""
        if fit_transform:
            # Feature engineering
            X_engineered = self.feature_engineer.fit_transform(X)
            
            # Scaling
            # Use robust scaler for Amount and Time, standard scaler for others
            amount_time_cols = ['Amount', 'Time'] + [col for col in X_engineered.columns 
                                                   if 'Amount' in col or 'Time' in col or 'Hour' in col]
            amount_time_cols = [col for col in amount_time_cols if col in X_engineered.columns]
            
            other_cols = [col for col in X_engineered.columns if col not in amount_time_cols]
            
            if amount_time_cols:
                X_engineered[amount_time_cols] = self.robust_scaler.fit_transform(X_engineered[amount_time_cols])
            if other_cols:
                X_engineered[other_cols] = self.scaler.fit_transform(X_engineered[other_cols])
                
        else:
            # Transform only
            X_engineered = self.feature_engineer.transform(X)
            
            amount_time_cols = ['Amount', 'Time'] + [col for col in X_engineered.columns 
                                                   if 'Amount' in col or 'Time' in col or 'Hour' in col]
            amount_time_cols = [col for col in amount_time_cols if col in X_engineered.columns]
            
            other_cols = [col for col in X_engineered.columns if col not in amount_time_cols]
            
            if amount_time_cols:
                X_engineered[amount_time_cols] = self.robust_scaler.transform(X_engineered[amount_time_cols])
            if other_cols:
                X_engineered[other_cols] = self.scaler.transform(X_engineered[other_cols])
        
        return X_engineered
        
    def train(self, X, y, use_smote=True, test_size=0.3):
        """Train the hybrid model"""
        print("Starting training process...")
        
        # Initialize models
        self.initialize_models()
        
        # Preprocessing
        print("Preprocessing data...")
        X_processed = self.preprocess_data(X, y, fit_transform=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Handle class imbalance
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            
        print(f"Training samples: {X_train_balanced.shape[0]}")
        print(f"Class distribution: {np.bincount(y_train_balanced)}")
        
        # Train individual models
        print("Training individual models...")
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'svm':
                # Train SVM on subset for computational efficiency
                if len(X_train_balanced) > 10000:
                    sample_indices = np.random.choice(len(X_train_balanced), 10000, replace=False)
                    X_svm = X_train_balanced.iloc[sample_indices]
                    y_svm = y_train_balanced.iloc[sample_indices]
                else:
                    X_svm, y_svm = X_train_balanced, y_train_balanced
                    
                model.fit(X_svm, y_svm)
                
            elif name == 'isolation_forest':
                # Train on normal transactions only
                normal_indices = y_train_balanced == 0
                X_normal = X_train_balanced[normal_indices]
                model.fit(X_normal)
                
            else:
                model.fit(X_train_balanced, y_train_balanced)
                
            trained_models[name] = model
            
        # Create ensemble (excluding isolation forest for voting)
        ensemble_models = [(name, model) for name, model in trained_models.items() 
                          if name != 'isolation_forest']
        
        self.ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            weights=[2, 3, 1, 1, 1]  # Give more weight to RF and XGB
        )
        
        print("Training ensemble...")
        self.ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Store trained models
        self.models = trained_models
        self.is_trained = True
        
        # Evaluate on test set
        print("Evaluating models...")
        self.evaluate_models(X_test, y_test)
        
        # Feature importance
        self.calculate_feature_importance(X_processed.columns)
        
        return X_test, y_test
        
    def predict(self, X, return_proba=False):
        """Make predictions using the hybrid approach"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
            
        # Preprocess
        X_processed = self.preprocess_data(X, fit_transform=False)
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict(X_processed)
        ensemble_proba = self.ensemble.predict_proba(X_processed)[:, 1]
        
        # Isolation Forest prediction
        iso_pred = self.models['isolation_forest'].predict(X_processed)
        iso_pred_binary = np.where(iso_pred == -1, 1, 0)
        
        # Hybrid decision: If isolation forest detects anomaly, increase fraud probability
        hybrid_proba = ensemble_proba.copy()
        anomaly_mask = iso_pred_binary == 1
        hybrid_proba[anomaly_mask] = np.minimum(hybrid_proba[anomaly_mask] + 0.2, 1.0)
        
        # Final prediction
        final_pred = (hybrid_proba > 0.5).astype(int)
        
        if return_proba:
            return final_pred, hybrid_proba
        return final_pred
        
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Preprocess test data
        X_test_processed = self.preprocess_data(X_test, fit_transform=False)
        
        # Individual model evaluation
        results = {}
        for name, model in self.models.items():
            if name == 'isolation_forest':
                y_pred = model.predict(X_test_processed)
                y_pred_binary = np.where(y_pred == -1, 1, 0)
                y_proba = np.zeros_like(y_pred_binary, dtype=float)
                y_proba[y_pred_binary == 1] = 0.8  # Assign high probability to anomalies
            else:
                y_pred_binary = model.predict(X_test_processed)
                y_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)
            auc = roc_auc_score(y_test, y_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_score': auc,
                'predictions': y_pred_binary,
                'probabilities': y_proba
            }
            
            print(f"\n{name.upper()}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC Score: {auc:.4f}")
        
        # Ensemble evaluation
        ensemble_pred = self.ensemble.predict(X_test_processed)
        ensemble_proba = self.ensemble.predict_proba(X_test_processed)[:, 1]
        
        print(f"\nENSEMBLE")
        print(f"Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, ensemble_pred):.4f}")
        print(f"AUC Score: {roc_auc_score(y_test, ensemble_proba):.4f}")
        
        # Hybrid evaluation
        hybrid_pred, hybrid_proba = self.predict(X_test, return_proba=True)
        
        print(f"\nHYBRID (Final Model)")
        print(f"Accuracy: {accuracy_score(y_test, hybrid_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, hybrid_pred):.4f}")
        print(f"AUC Score: {roc_auc_score(y_test, hybrid_proba):.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report (Hybrid Model):")
        print(classification_report(y_test, hybrid_pred, digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, hybrid_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'], 
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title('Confusion Matrix - Hybrid Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, hybrid_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Hybrid Model (AUC = {roc_auc_score(y_test, hybrid_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Hybrid Model')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, hybrid_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Hybrid Model')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Hybrid Model')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return results
        
    def calculate_feature_importance(self, feature_names):
        """Calculate and display feature importance"""
        print("\nCalculating feature importance...")
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['random_forest'] = feature_importance_df
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 20 Feature Importance (Random Forest)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'ensemble': self.ensemble,
            'feature_engineer': self.feature_engineer,
            'scaler': self.scaler,
            'robust_scaler': self.robust_scaler,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.ensemble = model_data['ensemble']
        self.feature_engineer = model_data['feature_engineer']
        self.scaler = model_data['scaler']
        self.robust_scaler = model_data['robust_scaler']
        self.is_trained = model_data['is_trained']
        self.feature_importance = model_data.get('feature_importance', {})
        print(f"Model loaded from {filepath}")
        
    def predict_single_transaction(self, transaction_data):
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
            
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            transaction_df = pd.DataFrame([transaction_data])
        else:
            transaction_df = transaction_data
            
        # Make prediction
        prediction, probability = self.predict(transaction_df, return_proba=True)
        
        result = {
            'is_fraud': bool(prediction[0]),
            'fraud_probability': float(probability[0]),
            'risk_level': self._get_risk_level(probability[0]),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'

# Example usage
if __name__ == "__main__":
    # Load your dataset
    # df = pd.read_csv("creditcard.csv")
    
    # For demonstration, let's create sample data
    print("Creating sample data for demonstration...")
    
    # Generate sample credit card data
    np.random.seed(42)
    n_samples = 50000
    
    # Create features similar to credit card dataset
    data = {}
    
    # Time and Amount
    data['Time'] = np.random.randint(0, 172800, n_samples)  # 48 hours
    data['Amount'] = np.random.exponential(88.35, n_samples)  # Average transaction amount
    
    # V1 to V28 (PCA features)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    # Create target (fraud)
    fraud_rate = 0.001  # 0.1% fraud rate
    target = np.random.binomial(1, fraud_rate, n_samples)
    
    # Make fraudulent transactions distinctive
    fraud_indices = np.where(target == 1)[0]
    for idx in fraud_indices:
        data['Amount'][idx] *= np.random.uniform(5, 20)  # Higher amounts
        for i in range(1, 15):  # Modify first 14 V features
            data[f'V{i}'][idx] *= np.random.uniform(2, 5)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['Class'] = target
    
    print(f"Dataset created with {n_samples} samples")
    print(f"Fraud rate: {target.mean():.4f}")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Initialize and train the detector
    detector = HybridFraudDetector()
    X_test, y_test = detector.train(X, y)
    
    # Save the model
    detector.save_model("enhanced_fraud_detector.joblib")
    
    # Example single transaction prediction
    print("\n" + "="*60)
    print("SINGLE TRANSACTION PREDICTION EXAMPLE")
    print("="*60)
    
    # Create a sample transaction
    sample_transaction = {
        'Time': 12345,
        'Amount': 1500.50,
        **{f'V{i}': np.random.randn() for i in range(1, 29)}
    }
    
    result = detector.predict_single_transaction(sample_transaction)
    
    print(f"Transaction Amount: ${sample_transaction['Amount']:.2f}")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Timestamp: {result['timestamp']}")
    
    print("\nTraining and evaluation completed successfully!")