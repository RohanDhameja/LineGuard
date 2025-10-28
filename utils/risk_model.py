"""
Risk Assessment Model for Fire App
ML-based wildfire risk prediction using Random Forest and Gradient Boosting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PowerLineRiskModel:
    """
    ML-based risk assessment for power line vegetation monitoring.
    Predicts wildfire risk based on vegetation, distance, and environmental factors.
    """
    
    def __init__(self, model_type='both'):
        """
        Initialize risk model.
        
        Args:
            model_type: 'classifier', 'regressor', or 'both'
        """
        self.model_type = model_type
        
        # Models
        self.rf_classifier = None
        self.rf_regressor = None
        self.gb_classifier = None
        self.gb_regressor = None
        
        # Best models
        self.best_classifier = None
        self.best_regressor = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature info
        self.feature_names = []
        self.feature_importance = {}
        
        # Metrics
        self.metrics = {'classifier': {}, 'regressor': {}}
        
        # Risk categories
        self.risk_levels = ['Low', 'Moderate', 'High', 'Critical']
        
        # Metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'trained_at': None,
            'n_samples': 0,
            'n_features': 0
        }
    
    def create_features(self, data_dict):
        """
        Create feature vector from zone data.
        
        Args:
            data_dict: {
                'vegetation_height': float,
                'clearance': float,
                'ndvi': float (optional),
                'temperature': float (optional),
                'humidity': float (optional),
                'wind_speed': float (optional),
                'line_voltage': float (optional),
                'line_age': int (optional)
            }
            
        Returns:
            pd.DataFrame: Feature vector
        """
        features = {}
        
        # === VEGETATION FEATURES ===
        veg_height = data_dict.get('vegetation_height', 0.5)
        features['vegetation_height'] = veg_height
        features['veg_height_squared'] = veg_height ** 2
        
        ndvi = data_dict.get('ndvi', 0.3)
        features['ndvi'] = ndvi
        features['ndvi_squared'] = ndvi ** 2
        
        # Vegetation risk score
        features['vegetation_density'] = min(1.0, veg_height / 5.0)
        features['vegetation_health_risk'] = ndvi * features['vegetation_density']
        
        # === DISTANCE FEATURES ===
        clearance = data_dict.get('clearance', 8.0)
        features['clearance'] = clearance
        features['clearance_inverse'] = 1.0 / (clearance + 0.1)  # Prevent division by zero
        
        # Proximity risk (exponential decay with distance)
        features['proximity_risk'] = np.exp(-clearance / 5.0)
        features['clearance_violation'] = 1.0 if clearance < 3.0 else 0.0
        
        # Height vs clearance interaction
        features['encroachment_risk'] = max(0, (veg_height - clearance / 2.0) / 10.0)
        
        # === ENVIRONMENTAL FEATURES ===
        temperature = data_dict.get('temperature', 22.0)
        features['temperature'] = temperature
        features['temperature_risk'] = max(0, (temperature - 20) / 20.0)
        
        humidity = data_dict.get('humidity', 50.0)
        features['humidity'] = humidity
        features['dryness_index'] = 1.0 - (humidity / 100.0)
        
        wind_speed = data_dict.get('wind_speed', 3.0)
        features['wind_speed'] = wind_speed
        features['wind_risk'] = min(1.0, wind_speed / 15.0)
        
        # Fire danger rating (simplified)
        features['fire_danger_rating'] = (
            0.3 * features['temperature_risk'] +
            0.4 * features['dryness_index'] +
            0.3 * features['wind_risk']
        )
        
        # === INFRASTRUCTURE FEATURES ===
        voltage = data_dict.get('line_voltage', 115.0)
        features['voltage'] = voltage / 500.0  # Normalized
        
        line_age = data_dict.get('line_age', 20)
        features['line_age'] = min(1.0, line_age / 50.0)
        
        # === TEMPORAL FEATURES ===
        month = data_dict.get('month', datetime.now().month)
        features['month'] = month
        features['is_fire_season'] = 1.0 if 6 <= month <= 10 else 0.0
        
        # Seasonal risk factor
        features['seasonal_risk'] = 0.5 + 0.5 * np.cos(2 * np.pi * (month - 8) / 12)
        
        # === COMPOSITE FEATURES ===
        # Vegetation + Distance interaction
        features['veg_distance_interaction'] = features['vegetation_density'] * features['proximity_risk']
        
        # Weather + Vegetation interaction
        features['dry_vegetation_risk'] = features['dryness_index'] * features['vegetation_density']
        
        # Temperature + Wind interaction
        features['extreme_weather_risk'] = features['temperature_risk'] * features['wind_risk']
        
        # Overall composite risk
        features['composite_risk'] = (
            0.35 * features['proximity_risk'] +
            0.25 * features['vegetation_density'] +
            0.20 * features['fire_danger_rating'] +
            0.10 * features['encroachment_risk'] +
            0.10 * features['seasonal_risk']
        )
        
        return pd.DataFrame([features])
    
    def generate_training_data(self, n_samples=3000, random_state=42):
        """
        Generate synthetic training data for model development.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            tuple: (features_df, risk_labels, risk_scores)
        """
        np.random.seed(random_state)
        
        print(f"🔄 Generating {n_samples} synthetic training samples...")
        
        all_features = []
        risk_scores = []
        risk_labels = []
        
        for i in range(n_samples):
            # Generate different risk scenarios
            scenario_type = np.random.choice(
                ['low_risk', 'moderate_risk', 'high_risk', 'critical_risk'],
                p=[0.4, 0.3, 0.2, 0.1]
            )
            
            if scenario_type == 'low_risk':
                data = {
                    'vegetation_height': np.random.uniform(0.1, 0.5),
                    'clearance': np.random.uniform(8.0, 15.0),
                    'ndvi': np.random.uniform(0.1, 0.4),
                    'temperature': np.random.uniform(15.0, 25.0),
                    'humidity': np.random.uniform(50.0, 80.0),
                    'wind_speed': np.random.uniform(1.0, 5.0),
                    'line_voltage': np.random.choice([69.0, 115.0]),
                    'line_age': np.random.randint(5, 25),
                    'month': np.random.choice([1, 2, 3, 4, 11, 12])
                }
                risk_score = np.random.uniform(0.0, 0.3)
                risk_label = 'Low'
                
            elif scenario_type == 'moderate_risk':
                data = {
                    'vegetation_height': np.random.uniform(0.5, 1.2),
                    'clearance': np.random.uniform(5.0, 8.0),
                    'ndvi': np.random.uniform(0.4, 0.6),
                    'temperature': np.random.uniform(22.0, 30.0),
                    'humidity': np.random.uniform(35.0, 55.0),
                    'wind_speed': np.random.uniform(4.0, 10.0),
                    'line_voltage': np.random.choice([115.0, 230.0]),
                    'line_age': np.random.randint(20, 40),
                    'month': np.random.choice([5, 6, 9, 10])
                }
                risk_score = np.random.uniform(0.3, 0.6)
                risk_label = 'Moderate'
                
            elif scenario_type == 'high_risk':
                data = {
                    'vegetation_height': np.random.uniform(1.2, 2.5),
                    'clearance': np.random.uniform(3.0, 5.0),
                    'ndvi': np.random.uniform(0.5, 0.75),
                    'temperature': np.random.uniform(28.0, 38.0),
                    'humidity': np.random.uniform(20.0, 40.0),
                    'wind_speed': np.random.uniform(8.0, 18.0),
                    'line_voltage': np.random.choice([230.0, 345.0]),
                    'line_age': np.random.randint(35, 60),
                    'month': np.random.choice([7, 8, 9])
                }
                risk_score = np.random.uniform(0.6, 0.85)
                risk_label = 'High'
                
            else:  # critical_risk
                data = {
                    'vegetation_height': np.random.uniform(2.5, 5.0),
                    'clearance': np.random.uniform(1.0, 3.0),
                    'ndvi': np.random.uniform(0.65, 0.85),
                    'temperature': np.random.uniform(35.0, 45.0),
                    'humidity': np.random.uniform(10.0, 25.0),
                    'wind_speed': np.random.uniform(15.0, 30.0),
                    'line_voltage': np.random.choice([345.0, 500.0]),
                    'line_age': np.random.randint(50, 80),
                    'month': np.random.choice([7, 8, 9])
                }
                risk_score = np.random.uniform(0.85, 1.0)
                risk_label = 'Critical'
            
            # Create features
            features = self.create_features(data)
            all_features.append(features)
            risk_scores.append(risk_score)
            risk_labels.append(risk_label)
        
        # Combine all features
        features_df = pd.concat(all_features, ignore_index=True)
        
        print(f"✅ Generated {len(features_df)} samples with {len(features_df.columns)} features")
        print(f"📊 Risk distribution: {pd.Series(risk_labels).value_counts().to_dict()}")
        
        return features_df, risk_labels, risk_scores
    
    def train_models(self, features_df, risk_labels, risk_scores, test_size=0.2, random_state=42):
        """
        Train Random Forest and Gradient Boosting models.
        
        Args:
            features_df: Feature matrix
            risk_labels: Classification labels
            risk_scores: Regression targets (0-1)
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            dict: Training results
        """
        print("\n" + "="*70)
        print("🚀 TRAINING RISK ASSESSMENT MODELS")
        print("="*70)
        
        self.metadata['n_samples'] = len(features_df)
        self.metadata['n_features'] = len(features_df.columns)
        self.feature_names = features_df.columns.tolist()
        
        print(f"\n📊 Dataset: {len(features_df)} samples, {len(features_df.columns)} features")
        
        # Scale features
        print("⚙️  Preprocessing data...")
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(risk_labels)
        
        # Split data
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_encoded, risk_scores,
            test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        results = {}
        
        # === TRAIN CLASSIFIERS ===
        if self.model_type in ['classifier', 'both']:
            print("\n🎯 Training Classification Models...")
            
            # Random Forest
            self.rf_classifier = RandomForestClassifier(
                n_estimators=150, max_depth=15, min_samples_split=10,
                random_state=random_state, n_jobs=-1
            )
            self.rf_classifier.fit(X_train, y_train_class)
            rf_test_score = self.rf_classifier.score(X_test, y_test_class)
            
            # Gradient Boosting
            self.gb_classifier = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=7,
                random_state=random_state
            )
            self.gb_classifier.fit(X_train, y_train_class)
            gb_test_score = self.gb_classifier.score(X_test, y_test_class)
            
            # Select best
            if gb_test_score > rf_test_score:
                self.best_classifier = self.gb_classifier
                best_classifier_name = "Gradient Boosting"
                best_pred = self.gb_classifier.predict(X_test)
            else:
                self.best_classifier = self.rf_classifier
                best_classifier_name = "Random Forest"
                best_pred = self.rf_classifier.predict(X_test)
            
            print(f"🏆 Best Classifier: {best_classifier_name}")
            
            # Metrics
            accuracy = accuracy_score(y_test_class, best_pred)
            precision = precision_score(y_test_class, best_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_class, best_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_class, best_pred, average='weighted', zero_division=0)
            
            print(f"   Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f} | F1-Score: {f1:.4f}")
            
            self.metrics['classifier'] = {
                'best_model': best_classifier_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Feature importance
            if hasattr(self.best_classifier, 'feature_importances_'):
                self.feature_importance['classifier'] = dict(zip(
                    self.feature_names,
                    self.best_classifier.feature_importances_
                ))
            
            results['classifier'] = {
                'predictions': best_pred,
                'true_labels': y_test_class
            }
        
        # === TRAIN REGRESSORS ===
        if self.model_type in ['regressor', 'both']:
            print("\n📊 Training Regression Models...")
            
            # Random Forest
            self.rf_regressor = RandomForestRegressor(
                n_estimators=150, max_depth=15, min_samples_split=10,
                random_state=random_state, n_jobs=-1
            )
            self.rf_regressor.fit(X_train, y_train_reg)
            rf_test_r2 = self.rf_regressor.score(X_test, y_test_reg)
            rf_pred = self.rf_regressor.predict(X_test)
            rf_mae = mean_absolute_error(y_test_reg, rf_pred)
            
            # Gradient Boosting
            self.gb_regressor = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=7,
                random_state=random_state
            )
            self.gb_regressor.fit(X_train, y_train_reg)
            gb_test_r2 = self.gb_regressor.score(X_test, y_test_reg)
            gb_pred = self.gb_regressor.predict(X_test)
            gb_mae = mean_absolute_error(y_test_reg, gb_pred)
            
            # Select best
            if gb_test_r2 > rf_test_r2:
                self.best_regressor = self.gb_regressor
                best_regressor_name = "Gradient Boosting"
                best_pred_reg = gb_pred
                best_mae = gb_mae
                best_r2 = gb_test_r2
            else:
                self.best_regressor = self.rf_regressor
                best_regressor_name = "Random Forest"
                best_pred_reg = rf_pred
                best_mae = rf_mae
                best_r2 = rf_test_r2
            
            print(f"🏆 Best Regressor: {best_regressor_name}")
            print(f"   R² Score: {best_r2:.4f} | MAE: {best_mae:.4f}")
            
            self.metrics['regressor'] = {
                'best_model': best_regressor_name,
                'r2_score': best_r2,
                'mae': best_mae
            }
            
            # Feature importance
            if hasattr(self.best_regressor, 'feature_importances_'):
                self.feature_importance['regressor'] = dict(zip(
                    self.feature_names,
                    self.best_regressor.feature_importances_
                ))
            
            results['regressor'] = {
                'predictions': best_pred_reg,
                'true_scores': y_test_reg
            }
        
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        print("\n✅ MODEL TRAINING COMPLETE")
        print("="*70)
        
        return results
    
    def predict(self, data_dict):
        """
        Predict risk for new zone data.
        
        Args:
            data_dict: Zone data dictionary
            
        Returns:
            dict: Prediction results
        """
        # Create features
        features = self.create_features(data_dict)
        
        # Scale
        X_scaled = self.scaler.transform(features)
        
        results = {}
        
        # Classification
        if self.best_classifier:
            pred_encoded = self.best_classifier.predict(X_scaled)[0]
            pred_proba = self.best_classifier.predict_proba(X_scaled)[0]
            
            risk_level = self.label_encoder.inverse_transform([pred_encoded])[0]
            confidence = np.max(pred_proba)
            
            risk_probabilities = dict(zip(
                self.label_encoder.classes_,
                pred_proba
            ))
            
            results['classification'] = {
                'risk_level': risk_level,
                'confidence': confidence,
                'probabilities': risk_probabilities
            }
        
        # Regression
        if self.best_regressor:
            risk_score = self.best_regressor.predict(X_scaled)[0]
            risk_score = np.clip(risk_score, 0, 1)
            
            results['regression'] = {
                'risk_score': risk_score,
                'risk_percentage': risk_score * 100
            }
        
        return results
    
    def save_models(self, filepath_prefix='models/fire_app_risk_model'):
        """Save trained models."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save models
            if self.best_classifier:
                joblib.dump(self.best_classifier, f'{filepath_prefix}_classifier_{timestamp}.pkl')
            
            if self.best_regressor:
                joblib.dump(self.best_regressor, f'{filepath_prefix}_regressor_{timestamp}.pkl')
            
            # Save preprocessors
            joblib.dump(self.scaler, f'{filepath_prefix}_scaler_{timestamp}.pkl')
            joblib.dump(self.label_encoder, f'{filepath_prefix}_label_encoder_{timestamp}.pkl')
            
            # Save metadata
            metadata = {
                'metadata': self.metadata,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance
            }
            
            with open(f'{filepath_prefix}_metadata_{timestamp}.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Models saved with timestamp: {timestamp}")
            return timestamp
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return None
    
    def load_models(self, filepath_prefix='models/fire_app_risk_model', timestamp=None):
        """Load trained models."""
        try:
            if not timestamp:
                # Find most recent
                import glob
                import os
                files = glob.glob(f'{filepath_prefix}_metadata_*.json')
                if not files:
                    return False
                latest_file = max(files, key=os.path.getctime)
                # Extract timestamp properly (everything after last occurrence of prefix + '_metadata_')
                timestamp = latest_file.replace(filepath_prefix + '_metadata_', '').replace('.json', '')
            
            # Load models
            try:
                self.best_classifier = joblib.load(f'{filepath_prefix}_classifier_{timestamp}.pkl')
            except:
                pass
            
            try:
                self.best_regressor = joblib.load(f'{filepath_prefix}_regressor_{timestamp}.pkl')
            except:
                pass
            
            # Load preprocessors
            self.scaler = joblib.load(f'{filepath_prefix}_scaler_{timestamp}.pkl')
            self.label_encoder = joblib.load(f'{filepath_prefix}_label_encoder_{timestamp}.pkl')
            
            # Load metadata
            with open(f'{filepath_prefix}_metadata_{timestamp}.json', 'r') as f:
                metadata = json.load(f)
                self.metadata = metadata['metadata']
                self.metrics = metadata['metrics']
                self.feature_names = metadata['feature_names']
                self.feature_importance = metadata['feature_importance']
            
            print(f"✅ Models loaded from timestamp: {timestamp}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False

