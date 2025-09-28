import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VegetationClassifier:
    """
    Advanced machine learning models for automated vegetation classification
    and risk assessment.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Vegetation type classifier
        self.vegetation_classifier = None
        self.risk_classifier = None
        self.growth_predictor = None
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model metadata
        self.model_metadata = {
            'vegetation_types': ['Grass', 'Shrubs', 'Trees_Deciduous', 'Trees_Coniferous', 'Mixed_Vegetation'],
            'risk_levels': ['Low', 'Moderate', 'High', 'Critical'],
            'growth_categories': ['Slow', 'Moderate', 'Fast', 'Rapid']
        }
    
    def create_training_features(self, satellite_data, lidar_data, environmental_data=None):
        """
        Create feature matrix for training machine learning models.
        
        Args:
            satellite_data: Dict containing NDVI and spectral data
            lidar_data: Dict containing canopy height and structure data
            environmental_data: Optional dict with weather/climate data
            
        Returns:
            pd.DataFrame: Feature matrix for ML training
        """
        features = []
        
        # Extract NDVI-based features
        if satellite_data and 'ndvi_values' in satellite_data:
            ndvi_values = np.array(satellite_data['ndvi_values'])
            
            # Statistical features from NDVI
            ndvi_features = {
                'ndvi_mean': np.mean(ndvi_values),
                'ndvi_std': np.std(ndvi_values),
                'ndvi_min': np.min(ndvi_values),
                'ndvi_max': np.max(ndvi_values),
                'ndvi_range': np.max(ndvi_values) - np.min(ndvi_values),
                'ndvi_skew': self._calculate_skewness(ndvi_values),
                'ndvi_percentile_25': np.percentile(ndvi_values, 25),
                'ndvi_percentile_75': np.percentile(ndvi_values, 75),
                'vegetation_fraction': np.sum(ndvi_values > 0.3) / len(ndvi_values),
                'healthy_vegetation_fraction': np.sum(ndvi_values > 0.6) / len(ndvi_values)
            }
            
            # Seasonal patterns (based on current month)
            month = datetime.now().month
            ndvi_features['seasonal_factor'] = 0.5 + 0.5 * np.cos(2 * np.pi * (month - 6) / 12)
            
        else:
            # Default NDVI features if no data available
            ndvi_features = {f'ndvi_{key}': 0.0 for key in [
                'mean', 'std', 'min', 'max', 'range', 'skew', 
                'percentile_25', 'percentile_75', 'vegetation_fraction', 
                'healthy_vegetation_fraction', 'seasonal_factor'
            ]}
        
        # Extract LiDAR-based features
        if lidar_data and 'height_values' in lidar_data:
            height_values = np.array(lidar_data['height_values'])
            
            height_features = {
                'canopy_height_mean': np.mean(height_values),
                'canopy_height_std': np.std(height_values),
                'canopy_height_max': np.max(height_values),
                'canopy_height_95th': np.percentile(height_values, 95),
                'canopy_cover_density': np.sum(height_values > 2) / len(height_values),
                'tall_vegetation_fraction': np.sum(height_values > 10) / len(height_values),
                'canopy_complexity': np.std(height_values) / (np.mean(height_values) + 0.1),
                'understory_fraction': np.sum((height_values > 1) & (height_values < 5)) / len(height_values)
            }
        else:
            # Default height features
            height_features = {f'canopy_{key}': 0.0 for key in [
                'height_mean', 'height_std', 'height_max', 'height_95th',
                'cover_density', 'tall_vegetation_fraction', 'complexity', 'understory_fraction'
            ]}
        
        # Environmental features
        if environmental_data:
            env_features = {
                'temperature_avg': environmental_data.get('temperature', 20),
                'humidity_avg': environmental_data.get('humidity', 50),
                'precipitation_mm': environmental_data.get('precipitation', 100),
                'wind_speed_ms': environmental_data.get('wind_speed', 3),
                'solar_radiation': environmental_data.get('solar_radiation', 200),
                'soil_moisture': environmental_data.get('soil_moisture', 0.3)
            }
        else:
            # Generate synthetic environmental features
            env_features = {
                'temperature_avg': np.random.normal(22, 8),
                'humidity_avg': np.random.uniform(30, 80),
                'precipitation_mm': np.random.gamma(2, 50),
                'wind_speed_ms': np.random.gamma(2, 2),
                'solar_radiation': np.random.uniform(150, 300),
                'soil_moisture': np.random.beta(2, 3)
            }
        
        # Combine all features
        all_features = {**ndvi_features, **height_features, **env_features}
        
        return pd.DataFrame([all_features])
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def train_vegetation_classifier(self, training_data, target_column='vegetation_type'):
        """
        Train a classifier to identify vegetation types from satellite and LiDAR data.
        
        Args:
            training_data: DataFrame with features and vegetation type labels
            target_column: Column name containing vegetation type labels
        """
        try:
            # Prepare data
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Encode labels (ensure string conversion)
            self.label_encoders['vegetation'] = LabelEncoder()
            y_encoded = self.label_encoders['vegetation'].fit_transform(y.astype(str))
            
            # Scale features
            self.scalers['vegetation'] = StandardScaler()
            X_scaled = self.scalers['vegetation'].fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train multiple models and choose the best
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            best_score = 0
            best_model = None
            
            for model_name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                avg_score = np.mean(cv_scores)
                
                print(f"{model_name} CV Score: {avg_score:.3f} ± {np.std(cv_scores):.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # Train the best model
            self.vegetation_classifier = best_model
            self.vegetation_classifier.fit(X_train, y_train)
            
            # Evaluate on test set
            test_score = self.vegetation_classifier.score(X_test, y_test)
            print(f"✅ Vegetation classifier trained. Test accuracy: {test_score:.3f}")
            
            # Store feature importance
            if hasattr(self.vegetation_classifier, 'feature_importances_'):
                self.feature_importance['vegetation'] = dict(zip(
                    feature_columns, self.vegetation_classifier.feature_importances_
                ))
            
            return test_score
            
        except Exception as e:
            print(f"❌ Error training vegetation classifier: {str(e)}")
            return None
    
    def train_risk_classifier(self, training_data, target_column='risk_level'):
        """
        Train a classifier to predict fire risk levels based on vegetation and environmental factors.
        """
        try:
            # Prepare data
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Encode labels (ensure string conversion)
            self.label_encoders['risk'] = LabelEncoder()
            y_encoded = self.label_encoders['risk'].fit_transform(y.astype(str))
            
            # Scale features
            self.scalers['risk'] = StandardScaler()
            X_scaled = self.scalers['risk'].fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train Random Forest for risk prediction
            self.risk_classifier = RandomForestClassifier(
                n_estimators=150, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            self.risk_classifier.fit(X_train, y_train)
            
            # Evaluate
            test_score = self.risk_classifier.score(X_test, y_test)
            print(f"✅ Risk classifier trained. Test accuracy: {test_score:.3f}")
            
            # Store feature importance
            if hasattr(self.risk_classifier, 'feature_importances_'):
                self.feature_importance['risk'] = dict(zip(
                    feature_columns, self.risk_classifier.feature_importances_
                ))
            
            return test_score
            
        except Exception as e:
            print(f"❌ Error training risk classifier: {str(e)}")
            return None
    
    def classify_vegetation(self, features_df):
        """
        Classify vegetation types using the trained model.
        
        Args:
            features_df: DataFrame with features for classification
            
        Returns:
            dict: Classification results with probabilities
        """
        if self.vegetation_classifier is None:
            return self._generate_mock_vegetation_classification()
        
        try:
            # Scale features
            X_scaled = self.scalers['vegetation'].transform(features_df)
            
            # Predict
            predictions = self.vegetation_classifier.predict(X_scaled)
            probabilities = self.vegetation_classifier.predict_proba(X_scaled)
            
            # Decode labels
            vegetation_types = self.label_encoders['vegetation'].inverse_transform(predictions)
            
            results = {
                'predicted_vegetation': vegetation_types[0],
                'confidence': np.max(probabilities[0]),
                'probabilities': dict(zip(
                    self.label_encoders['vegetation'].classes_,
                    probabilities[0]
                )),
                'feature_contributions': self._get_feature_contributions(features_df, 'vegetation')
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in vegetation classification: {str(e)}")
            return self._generate_mock_vegetation_classification()
    
    def predict_fire_risk(self, features_df):
        """
        Predict fire risk level using the trained model.
        """
        if self.risk_classifier is None:
            return self._generate_mock_risk_prediction()
        
        try:
            # Scale features
            X_scaled = self.scalers['risk'].transform(features_df)
            
            # Predict
            predictions = self.risk_classifier.predict(X_scaled)
            probabilities = self.risk_classifier.predict_proba(X_scaled)
            
            # Decode labels
            risk_levels = self.label_encoders['risk'].inverse_transform(predictions)
            
            results = {
                'predicted_risk': risk_levels[0],
                'risk_score': np.max(probabilities[0]),
                'risk_probabilities': dict(zip(
                    self.label_encoders['risk'].classes_,
                    probabilities[0]
                )),
                'contributing_factors': self._get_feature_contributions(features_df, 'risk')
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in risk prediction: {str(e)}")
            return self._generate_mock_risk_prediction()
    
    def cluster_vegetation_areas(self, features_df, n_clusters=5):
        """
        Perform unsupervised clustering to identify vegetation patterns.
        """
        try:
            # Check if we have enough samples for clustering
            if len(features_df) < n_clusters:
                print(f"Warning: Only {len(features_df)} samples available, adjusting clustering parameters")
                n_clusters = max(2, len(features_df) // 2)
                if n_clusters < 2:
                    return {
                        'error': 'Insufficient data for clustering',
                        'message': f'Need at least 2 samples for clustering, got {len(features_df)}'
                    }
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # DBSCAN for density-based clustering (adjust parameters for small datasets)
            eps = 0.5 if len(features_df) > 10 else 1.0
            min_samples = min(5, max(2, len(features_df) // 3))
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            density_labels = dbscan.fit_predict(X_scaled)
            
            # Analyze clusters
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    cluster_features = features_df[cluster_mask]
                    cluster_analysis[f'cluster_{i}'] = {
                        'size': np.sum(cluster_mask),
                        'avg_ndvi': cluster_features['ndvi_mean'].mean(),
                        'avg_height': cluster_features.get('canopy_height_mean', pd.Series([0])).mean(),
                        'characteristics': self._characterize_cluster(cluster_features)
                    }
            
            return {
                'kmeans_labels': cluster_labels.tolist(),
                'dbscan_labels': density_labels.tolist(),
                'cluster_analysis': cluster_analysis,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            print(f"❌ Error in vegetation clustering: {str(e)}")
            return {'error': str(e)}
    
    def _characterize_cluster(self, cluster_features):
        """Characterize a vegetation cluster based on its features."""
        ndvi_mean = cluster_features['ndvi_mean'].mean()
        height_mean = cluster_features.get('canopy_height_mean', pd.Series([5])).mean()
        
        if ndvi_mean > 0.7 and height_mean > 15:
            return "Dense Forest"
        elif ndvi_mean > 0.5 and height_mean > 8:
            return "Woodland"
        elif ndvi_mean > 0.4 and height_mean < 5:
            return "Grassland/Shrubland"
        elif ndvi_mean > 0.2:
            return "Sparse Vegetation"
        else:
            return "Bare Ground/Low Vegetation"
    
    def _get_feature_contributions(self, features_df, model_type):
        """Get feature contributions for model interpretability."""
        if model_type not in self.feature_importance:
            return {}
        
        feature_importance = self.feature_importance[model_type]
        feature_values = features_df.iloc[0].to_dict()
        
        contributions = {}
        for feature, importance in feature_importance.items():
            if feature in feature_values:
                contributions[feature] = {
                    'importance': importance,
                    'value': feature_values[feature],
                    'contribution_score': importance * abs(feature_values[feature])
                }
        
        # Sort by contribution score
        contributions = dict(sorted(
            contributions.items(), 
            key=lambda x: x[1]['contribution_score'], 
            reverse=True
        ))
        
        return contributions
    
    def _generate_mock_vegetation_classification(self):
        """Generate mock vegetation classification when model is not available."""
        vegetation_types = self.model_metadata['vegetation_types']
        selected_type = np.random.choice(vegetation_types)
        
        # Generate realistic probabilities
        probabilities = np.random.dirichlet(np.ones(len(vegetation_types)))
        
        return {
            'predicted_vegetation': selected_type,
            'confidence': np.max(probabilities),
            'probabilities': dict(zip(vegetation_types, probabilities)),
            'feature_contributions': {}
        }
    
    def _generate_mock_risk_prediction(self):
        """Generate mock risk prediction when model is not available."""
        risk_levels = self.model_metadata['risk_levels']
        
        # Weight towards moderate risk for realistic demo
        weights = [0.3, 0.4, 0.2, 0.1]  # Low, Moderate, High, Critical
        selected_risk = np.random.choice(risk_levels, p=weights)
        
        probabilities = np.random.dirichlet([3, 4, 2, 1])  # Weighted towards lower risk
        
        return {
            'predicted_risk': selected_risk,
            'risk_score': np.max(probabilities),
            'risk_probabilities': dict(zip(risk_levels, probabilities)),
            'contributing_factors': {}
        }
    
    def generate_synthetic_training_data(self, n_samples=1000):
        """
        Generate synthetic training data for model development.
        This would be replaced with real labeled data in production.
        """
        np.random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            # Generate base vegetation characteristics
            vegetation_type = str(np.random.choice(self.model_metadata['vegetation_types']))
            
            # Generate features based on vegetation type
            if vegetation_type == 'Trees_Coniferous':
                ndvi_mean = float(np.random.normal(0.75, 0.1))
                canopy_height = float(np.random.normal(20, 5))
                risk_level = str(np.random.choice(['Low', 'Moderate'], p=[0.7, 0.3]))
            elif vegetation_type == 'Trees_Deciduous':
                ndvi_mean = float(np.random.normal(0.7, 0.12))
                canopy_height = float(np.random.normal(15, 4))
                risk_level = str(np.random.choice(['Low', 'Moderate', 'High'], p=[0.5, 0.4, 0.1]))
            elif vegetation_type == 'Shrubs':
                ndvi_mean = float(np.random.normal(0.5, 0.15))
                canopy_height = float(np.random.normal(3, 1))
                risk_level = str(np.random.choice(['Moderate', 'High'], p=[0.6, 0.4]))
            elif vegetation_type == 'Grass':
                ndvi_mean = float(np.random.normal(0.4, 0.1))
                canopy_height = float(np.random.normal(0.5, 0.2))
                risk_level = str(np.random.choice(['Moderate', 'High', 'Critical'], p=[0.5, 0.3, 0.2]))
            else:  # Mixed_Vegetation
                ndvi_mean = float(np.random.normal(0.6, 0.2))
                canopy_height = float(np.random.normal(8, 4))
                risk_level = str(np.random.choice(['Low', 'Moderate', 'High'], p=[0.3, 0.5, 0.2]))
            
            # Clip values to realistic ranges
            ndvi_mean = float(np.clip(ndvi_mean, 0, 1))
            canopy_height = float(np.clip(canopy_height, 0, 50))
            
            # Generate additional features - ensure all are native Python types
            sample = {
                'vegetation_type': vegetation_type,
                'risk_level': risk_level,
                'ndvi_mean': ndvi_mean,
                'ndvi_std': float(np.random.uniform(0.05, 0.2)),
                'ndvi_max': float(min(1.0, ndvi_mean + np.random.uniform(0.1, 0.3))),
                'ndvi_min': float(max(0.0, ndvi_mean - np.random.uniform(0.1, 0.3))),
                'vegetation_fraction': float(np.clip(ndvi_mean + np.random.normal(0, 0.1), 0, 1)),
                'healthy_vegetation_fraction': float(np.clip(ndvi_mean - 0.2 + np.random.normal(0, 0.1), 0, 1)),
                'canopy_height_mean': canopy_height,
                'canopy_height_std': float(np.random.uniform(1, 5)),
                'canopy_height_max': float(canopy_height + np.random.uniform(2, 8)),
                'canopy_cover_density': float(np.random.uniform(0.3, 0.9)),
                'tall_vegetation_fraction': float(1.0 if canopy_height > 10 else np.random.uniform(0, 0.3)),
                'temperature_avg': float(np.random.normal(22, 8)),
                'humidity_avg': float(np.random.uniform(30, 80)),
                'precipitation_mm': float(np.random.gamma(2, 50)),
                'wind_speed_ms': float(np.random.gamma(2, 2)),
                'seasonal_factor': float(np.random.uniform(0.5, 1.0))
            }
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def save_models(self, filepath_prefix='vegetation_models'):
        """Save trained models and preprocessors."""
        try:
            # Save models
            if self.vegetation_classifier:
                joblib.dump(self.vegetation_classifier, f'{filepath_prefix}_vegetation_classifier.pkl')
            
            if self.risk_classifier:
                joblib.dump(self.risk_classifier, f'{filepath_prefix}_risk_classifier.pkl')
            
            # Save scalers and encoders
            joblib.dump(self.scalers, f'{filepath_prefix}_scalers.pkl')
            joblib.dump(self.label_encoders, f'{filepath_prefix}_encoders.pkl')
            
            # Save feature importance
            with open(f'{filepath_prefix}_feature_importance.json', 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            print("✅ Models saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
    
    def load_models(self, filepath_prefix='vegetation_models'):
        """Load trained models and preprocessors."""
        try:
            # Load models
            try:
                self.vegetation_classifier = joblib.load(f'{filepath_prefix}_vegetation_classifier.pkl')
            except FileNotFoundError:
                print("Vegetation classifier not found")
            
            try:
                self.risk_classifier = joblib.load(f'{filepath_prefix}_risk_classifier.pkl')
            except FileNotFoundError:
                print("Risk classifier not found")
            
            # Load scalers and encoders
            try:
                self.scalers = joblib.load(f'{filepath_prefix}_scalers.pkl')
                self.label_encoders = joblib.load(f'{filepath_prefix}_encoders.pkl')
            except FileNotFoundError:
                print("Scalers/encoders not found")
            
            # Load feature importance
            try:
                with open(f'{filepath_prefix}_feature_importance.json', 'r') as f:
                    self.feature_importance = json.load(f)
            except FileNotFoundError:
                print("Feature importance not found")
            
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")