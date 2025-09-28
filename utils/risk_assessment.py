import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RiskAssessment:
    """
    Handles risk assessment calculations combining vegetation detection,
    distance analysis, and environmental factors to predict fire risk.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'vegetation_density', 'min_distance_to_powerline', 'avg_vegetation_height',
            'ndvi_mean', 'ndvi_std', 'slope', 'aspect', 'wind_exposure',
            'moisture_content', 'temperature_avg', 'humidity_avg'
        ]
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
    
    def perform_analysis(self, satellite_data, lidar_data, powerline_data, parameters):
        """
        Perform comprehensive risk analysis combining all data sources.
        
        Args:
            satellite_data: Processed satellite imagery data
            lidar_data: LiDAR/canopy height data
            powerline_data: Power line infrastructure data
            parameters: Analysis parameters from UI
            
        Returns:
            dict: Comprehensive analysis results
        """
        results = {
            'analysis_timestamp': pd.Timestamp.now(),
            'parameters': parameters,
            'data_sources': {
                'satellite': satellite_data is not None,
                'lidar': lidar_data is not None,
                'powerlines': powerline_data is not None and not powerline_data.empty
            }
        }
        
        try:
            # Generate analysis grid
            analysis_grid = self._create_analysis_grid(
                parameters['lat'], parameters['lon'], parameters['radius_km']
            )
            
            # Extract features for each grid cell
            features_df = self._extract_features(
                analysis_grid, satellite_data, lidar_data, powerline_data, parameters
            )
            
            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(features_df)
            features_df['risk_score'] = risk_scores
            
            # Generate statistics
            results.update(self._generate_statistics(features_df, parameters))
            
            # Identify priority areas
            results['priority_areas'] = self._identify_priority_areas(
                features_df, parameters['clearance_threshold']
            )
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(features_df, parameters)
            
            # Create risk areas for mapping
            results['risk_areas'] = self._create_risk_areas(features_df)
            
            return results
            
        except Exception as e:
            print(f"Error in risk analysis: {str(e)}")
            return self._generate_fallback_results(parameters)
    
    def _create_analysis_grid(self, center_lat, center_lon, radius_km, grid_size=100):
        """Create a grid of analysis points within the study area."""
        # Create grid points
        lat_range = radius_km / 111.0  # Rough conversion to degrees
        lon_range = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        # Generate grid
        n_points = int(np.sqrt(grid_size))
        lats = np.linspace(center_lat - lat_range, center_lat + lat_range, n_points)
        lons = np.linspace(center_lon - lon_range, center_lon + lon_range, n_points)
        
        grid_points = []
        for lat in lats:
            for lon in lons:
                # Check if point is within radius
                distance = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2) * 111.0
                if distance <= radius_km:
                    grid_points.append({
                        'lat': lat,
                        'lon': lon,
                        'point_id': len(grid_points)
                    })
        
        return pd.DataFrame(grid_points)
    
    def _extract_features(self, grid_points, satellite_data, lidar_data, powerline_data, parameters):
        """Extract features for each grid point."""
        features = []
        
        for idx, point in grid_points.iterrows():
            feature_vector = self._extract_point_features(
                point['lat'], point['lon'], satellite_data, lidar_data, powerline_data
            )
            feature_vector['lat'] = point['lat']
            feature_vector['lon'] = point['lon']
            feature_vector['point_id'] = point['point_id']
            features.append(feature_vector)
        
        return pd.DataFrame(features)
    
    def _extract_point_features(self, lat, lon, satellite_data, lidar_data, powerline_data):
        """Extract features for a single point."""
        features = {}
        
        # Satellite-derived features
        if satellite_data:
            ndvi_values = satellite_data.get('ndvi_values', [])
            if ndvi_values:
                # Simulate local NDVI by sampling from distribution
                local_ndvi = np.random.choice(ndvi_values)
                features['ndvi_mean'] = local_ndvi
                features['ndvi_std'] = satellite_data['ndvi_stats'].get('std', 0.1)
                features['vegetation_density'] = 1.0 if local_ndvi > 0.3 else 0.0
            else:
                features['ndvi_mean'] = 0.2
                features['ndvi_std'] = 0.1
                features['vegetation_density'] = 0.0
        else:
            features['ndvi_mean'] = 0.2
            features['ndvi_std'] = 0.1
            features['vegetation_density'] = 0.0
        
        # LiDAR-derived features
        if lidar_data:
            height_values = lidar_data.get('height_values', [])
            if height_values:
                local_height = np.random.choice(height_values)
                features['avg_vegetation_height'] = max(0, local_height)
            else:
                features['avg_vegetation_height'] = lidar_data['canopy_height_stats'].get('mean', 10.0)
        else:
            features['avg_vegetation_height'] = 8.0  # Default vegetation height
        
        # Distance to nearest power line
        if powerline_data is not None and not powerline_data.empty:
            min_distance = self._calculate_distance_to_powerlines(lat, lon, powerline_data)
            features['min_distance_to_powerline'] = min_distance
        else:
            features['min_distance_to_powerline'] = 100.0  # Default safe distance
        
        # Topographic features (simplified)
        features['slope'] = abs(np.random.normal(5, 3))  # Slope in degrees
        features['aspect'] = np.random.uniform(0, 360)    # Aspect in degrees
        
        # Environmental features (would normally come from weather APIs)
        features['wind_exposure'] = np.random.uniform(0.3, 1.0)
        features['moisture_content'] = np.random.uniform(0.1, 0.8)
        features['temperature_avg'] = np.random.normal(22, 8)  # Celsius
        features['humidity_avg'] = np.random.uniform(30, 80)   # Percentage
        
        return features
    
    def _calculate_distance_to_powerlines(self, lat, lon, powerline_data):
        """Calculate minimum distance from point to any power line."""
        from shapely.geometry import Point
        
        point = Point(lon, lat)
        min_distance = float('inf')
        
        for idx, line_row in powerline_data.iterrows():
            distance = point.distance(line_row.geometry) * 111000  # Convert to meters approximately
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_risk_scores(self, features_df):
        """Calculate fire risk scores using a combination of factors."""
        risk_scores = []
        
        for idx, row in features_df.iterrows():
            # Base risk from vegetation
            vegetation_risk = row['vegetation_density'] * (row['ndvi_mean'] - 0.3) / 0.7
            vegetation_risk = max(0, vegetation_risk)
            
            # Distance risk (higher risk when closer to power lines)
            distance_risk = 1.0 / (1.0 + row['min_distance_to_powerline'] / 10.0)
            
            # Height risk (taller vegetation = higher risk)
            height_risk = min(1.0, row['avg_vegetation_height'] / 20.0)
            
            # Environmental factors
            dryness_risk = 1.0 - row['moisture_content']
            temperature_risk = max(0, (row['temperature_avg'] - 20) / 20.0)
            wind_risk = row['wind_exposure']
            slope_risk = min(1.0, row['slope'] / 30.0)
            
            # Combine factors with weights
            risk_score = (
                0.25 * vegetation_risk +
                0.30 * distance_risk +
                0.15 * height_risk +
                0.10 * dryness_risk +
                0.10 * temperature_risk +
                0.05 * wind_risk +
                0.05 * slope_risk
            )
            
            risk_scores.append(min(1.0, max(0.0, risk_score)))
        
        return risk_scores
    
    def _generate_statistics(self, features_df, parameters):
        """Generate summary statistics from the analysis."""
        risk_scores = features_df['risk_score']
        
        stats = {
            'overall_risk': risk_scores.mean(),
            'risk_std': risk_scores.std(),
            'high_risk_count': len(risk_scores[risk_scores > parameters['risk_threshold']]),
            'vegetation_density': features_df['vegetation_density'].mean(),
            'avg_clearance': features_df['min_distance_to_powerline'].mean(),
            'min_clearance': features_df['min_distance_to_powerline'].min(),
            'risk_distribution': risk_scores.tolist(),
            'ndvi_stats': {
                'mean': features_df['ndvi_mean'].mean(),
                'max': features_df['ndvi_mean'].max(),
                'min': features_df['ndvi_mean'].min(),
                'vegetation_percentage': (features_df['vegetation_density'].mean() * 100)
            },
            'vegetation_health': {
                'Healthy': len(features_df[features_df['ndvi_mean'] > 0.6]),
                'Moderate': len(features_df[(features_df['ndvi_mean'] > 0.3) & (features_df['ndvi_mean'] <= 0.6)]),
                'Stressed': len(features_df[(features_df['ndvi_mean'] > 0.1) & (features_df['ndvi_mean'] <= 0.3)]),
                'Sparse': len(features_df[features_df['ndvi_mean'] <= 0.1])
            },
            'proximity_stats': {
                'critical_segments': len(features_df[features_df['min_distance_to_powerline'] < parameters['clearance_threshold']]),
                'avg_distance': features_df['min_distance_to_powerline'].mean(),
                'min_distance': features_df['min_distance_to_powerline'].min()
            },
            'ndvi_histogram': {
                'bins': np.histogram(features_df['ndvi_mean'], bins=10)[1].tolist(),
                'counts': np.histogram(features_df['ndvi_mean'], bins=10)[0].tolist()
            }
        }
        
        return stats
    
    def _identify_priority_areas(self, features_df, clearance_threshold):
        """Identify areas requiring immediate attention."""
        priority_areas = []
        
        # Filter high-risk areas
        high_risk_areas = features_df[
            (features_df['risk_score'] > 0.7) | 
            (features_df['min_distance_to_powerline'] < clearance_threshold)
        ].copy()
        
        # Sort by risk score and distance
        high_risk_areas = high_risk_areas.sort_values(['risk_score', 'min_distance_to_powerline'], 
                                                     ascending=[False, True])
        
        for idx, area in high_risk_areas.head(20).iterrows():  # Top 20 priority areas
            priority_areas.append({
                'lat': area['lat'],
                'lon': area['lon'],
                'risk_score': area['risk_score'],
                'clearance_distance': area['min_distance_to_powerline'],
                'vegetation_height': area['avg_vegetation_height'],
                'ndvi': area['ndvi_mean'],
                'priority_rank': len(priority_areas) + 1,
                'risk_category': self._categorize_risk(area['risk_score']),
                'action_required': self._determine_action(area)
            })
        
        return priority_areas
    
    def _categorize_risk(self, risk_score):
        """Categorize risk score into levels."""
        if risk_score >= self.risk_thresholds['critical']:
            return 'Critical'
        elif risk_score >= self.risk_thresholds['high']:
            return 'High'
        elif risk_score >= self.risk_thresholds['moderate']:
            return 'Moderate'
        else:
            return 'Low'
    
    def _determine_action(self, area):
        """Determine recommended action for a priority area."""
        risk_score = area['risk_score']
        distance = area['min_distance_to_powerline']
        
        if distance < 0.5:
            return 'Immediate vegetation removal required'
        elif distance < 1.0 and risk_score > 0.8:
            return 'Urgent vegetation trimming needed'
        elif risk_score > 0.7:
            return 'Schedule vegetation management'
        else:
            return 'Monitor vegetation growth'
    
    def _generate_recommendations(self, features_df, parameters):
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # High risk area count
        high_risk_count = len(features_df[features_df['risk_score'] > 0.7])
        if high_risk_count > 0:
            recommendations.append(
                f"Immediate attention required for {high_risk_count} high-risk areas"
            )
        
        # Critical clearance violations
        critical_clearance = len(features_df[
            features_df['min_distance_to_powerline'] < parameters['clearance_threshold']
        ])
        if critical_clearance > 0:
            recommendations.append(
                f"Vegetation clearance violations detected in {critical_clearance} locations"
            )
        
        # Vegetation density
        avg_vegetation = features_df['vegetation_density'].mean()
        if avg_vegetation > 0.6:
            recommendations.append(
                "High vegetation density detected - consider enhanced monitoring"
            )
        
        # Environmental factors
        avg_moisture = features_df['moisture_content'].mean()
        if avg_moisture < 0.3:
            recommendations.append(
                "Low moisture content increases fire risk - implement enhanced precautions"
            )
        
        # Seasonal recommendations
        recommendations.append(
            "Schedule regular vegetation surveys during peak growth seasons"
        )
        
        return recommendations
    
    def _create_risk_areas(self, features_df):
        """Create risk area markers for mapping."""
        risk_areas = []
        
        high_risk_areas = features_df[features_df['risk_score'] > 0.6]
        
        for idx, area in high_risk_areas.iterrows():
            risk_areas.append({
                'lat': area['lat'],
                'lon': area['lon'],
                'risk_score': area['risk_score'],
                'risk_category': self._categorize_risk(area['risk_score'])
            })
        
        return risk_areas
    
    def _generate_fallback_results(self, parameters):
        """Generate basic results when full analysis fails."""
        return {
            'analysis_timestamp': pd.Timestamp.now(),
            'parameters': parameters,
            'data_sources': {'satellite': False, 'lidar': False, 'powerlines': False},
            'overall_risk': 0.3,
            'high_risk_count': 0,
            'vegetation_density': 0.4,
            'avg_clearance': 15.0,
            'risk_distribution': [0.2, 0.3, 0.4, 0.3, 0.2],
            'priority_areas': [],
            'recommendations': [
                "Unable to perform full analysis - check data sources",
                "Consider manual inspection of the area"
            ],
            'risk_areas': []
        }
    
    def train_risk_model(self, training_data):
        """
        Train a machine learning model for risk prediction.
        This would be used with historical fire incident data.
        """
        if training_data is None or len(training_data) == 0:
            print("No training data available - using rule-based approach")
            return
        
        # Prepare features and target
        X = training_data[self.feature_columns]
        y = training_data['fire_occurred']  # Binary target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model performance - MSE: {mse:.3f}, R²: {r2:.3f}")
