import requests
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

class WeatherService:
    """
    Weather data service for real-time weather conditions and fire risk assessment.
    Integrates with weather APIs to provide environmental data for vegetation analysis.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        
        # Base URLs for different weather services
        self.openweather_base = "http://api.openweathermap.org/data/2.5"
        
        # Fire weather indices and thresholds
        self.fire_danger_thresholds = {
            'low': {'temp_max': 25, 'humidity_min': 60, 'wind_max': 15},
            'moderate': {'temp_max': 30, 'humidity_min': 40, 'wind_max': 25},
            'high': {'temp_max': 35, 'humidity_min': 25, 'wind_max': 35},
            'extreme': {'temp_max': float('inf'), 'humidity_min': 0, 'wind_max': float('inf')}
        }
    
    def get_current_weather(self, lat, lon):
        """
        Get current weather conditions for a specific location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            dict: Current weather data
        """
        if not self.api_key:
            return self._generate_synthetic_weather(lat, lon)
        
        try:
            url = f"{self.openweather_base}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_current_weather(data)
            else:
                print(f"Weather API error: {response.status_code}")
                return self._generate_synthetic_weather(lat, lon)
                
        except Exception as e:
            print(f"Error fetching weather data: {str(e)}")
            return self._generate_synthetic_weather(lat, lon)
    
    def get_weather_forecast(self, lat, lon, days=5):
        """
        Get weather forecast for fire risk prediction.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of forecast days
            
        Returns:
            dict: Weather forecast data
        """
        if not self.api_key:
            return self._generate_synthetic_forecast(lat, lon, days)
        
        try:
            url = f"{self.openweather_base}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast_data(data, days)
            else:
                print(f"Forecast API error: {response.status_code}")
                return self._generate_synthetic_forecast(lat, lon, days)
                
        except Exception as e:
            print(f"Error fetching forecast data: {str(e)}")
            return self._generate_synthetic_forecast(lat, lon, days)
    
    def calculate_fire_weather_index(self, weather_data):
        """
        Calculate fire weather index based on current conditions.
        
        Args:
            weather_data: Current weather conditions
            
        Returns:
            dict: Fire weather index and risk assessment
        """
        try:
            temp = weather_data.get('temperature', 20)
            humidity = weather_data.get('humidity', 50)
            wind_speed = weather_data.get('wind_speed', 5)
            precipitation = weather_data.get('precipitation_24h', 0)
            
            # Modified Haines Index calculation
            haines_index = self._calculate_haines_index(temp, humidity)
            
            # Drought factor (simplified)
            drought_factor = max(0, min(10, 10 - precipitation / 10))
            
            # Wind factor
            wind_factor = min(10, wind_speed / 5)
            
            # Temperature factor
            temp_factor = max(0, (temp - 20) / 3) if temp > 20 else 0
            
            # Humidity factor (inverse relationship)
            humidity_factor = max(0, (80 - humidity) / 10)
            
            # Combined fire weather index
            fwi = (temp_factor * 0.3 + 
                   humidity_factor * 0.25 + 
                   wind_factor * 0.2 + 
                   drought_factor * 0.15 + 
                   haines_index * 0.1)
            
            # Normalize to 0-100 scale
            fwi_normalized = min(100, max(0, fwi * 10))
            
            # Determine danger level
            danger_level = self._get_fire_danger_level(temp, humidity, wind_speed)
            
            return {
                'fire_weather_index': round(fwi_normalized, 1),
                'danger_level': danger_level,
                'haines_index': round(haines_index, 1),
                'factors': {
                    'temperature_factor': round(temp_factor, 2),
                    'humidity_factor': round(humidity_factor, 2),
                    'wind_factor': round(wind_factor, 2),
                    'drought_factor': round(drought_factor, 2)
                },
                'recommendations': self._get_fire_recommendations(danger_level, fwi_normalized)
            }
            
        except Exception as e:
            print(f"Error calculating fire weather index: {str(e)}")
            return {
                'fire_weather_index': 30.0,
                'danger_level': 'moderate',
                'haines_index': 4.0,
                'factors': {},
                'recommendations': ['Monitor conditions closely']
            }
    
    def _calculate_haines_index(self, temp, humidity):
        """Calculate simplified Haines Index for fire weather assessment."""
        # Stability component (simplified)
        stability = 1 if temp > 25 else 0
        
        # Moisture component
        moisture = 2 if humidity < 30 else (1 if humidity < 50 else 0)
        
        return stability + moisture + 2  # Base value of 2
    
    def _get_fire_danger_level(self, temp, humidity, wind_speed):
        """Determine fire danger level based on weather conditions."""
        if (temp <= self.fire_danger_thresholds['low']['temp_max'] and 
            humidity >= self.fire_danger_thresholds['low']['humidity_min'] and
            wind_speed <= self.fire_danger_thresholds['low']['wind_max']):
            return 'Low'
        elif (temp <= self.fire_danger_thresholds['moderate']['temp_max'] and
              humidity >= self.fire_danger_thresholds['moderate']['humidity_min'] and
              wind_speed <= self.fire_danger_thresholds['moderate']['wind_max']):
            return 'Moderate'
        elif (temp <= self.fire_danger_thresholds['high']['temp_max'] and
              humidity >= self.fire_danger_thresholds['high']['humidity_min'] and
              wind_speed <= self.fire_danger_thresholds['high']['wind_max']):
            return 'High'
        else:
            return 'Extreme'
    
    def _get_fire_recommendations(self, danger_level, fwi):
        """Get fire safety recommendations based on danger level."""
        recommendations = {
            'Low': [
                "Normal fire safety precautions",
                "Regular equipment inspections recommended",
                "Monitor for changes in conditions"
            ],
            'Moderate': [
                "Increase vegetation monitoring frequency",
                "Prepare emergency response teams",
                "Consider controlled burns if conditions permit",
                "Enhanced power line inspections recommended"
            ],
            'High': [
                "Heightened alert for maintenance crews",
                "Suspend non-essential power line work",
                "Deploy additional monitoring equipment",
                "Prepare for potential power outages",
                "Coordinate with local fire services"
            ],
            'Extreme': [
                "CRITICAL: Suspend all power line maintenance",
                "Activate emergency response protocols",
                "Consider proactive power shut-offs in high-risk areas",
                "Deploy all available monitoring resources",
                "Full coordination with emergency services"
            ]
        }
        
        return recommendations.get(danger_level, ["Monitor conditions closely"])
    
    def _parse_current_weather(self, data):
        """Parse OpenWeatherMap current weather response."""
        try:
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind'].get('speed', 0) * 3.6,  # Convert m/s to km/h
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 10000) / 1000,  # Convert m to km
                'weather_condition': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'precipitation_1h': data.get('rain', {}).get('1h', 0),
                'precipitation_24h': data.get('rain', {}).get('1h', 0) * 24,  # Rough estimate
                'cloud_cover': data['clouds']['all'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'location': f"{data['name']}, {data['sys']['country']}"
            }
        except Exception as e:
            print(f"Error parsing weather data: {str(e)}")
            return self._generate_synthetic_weather(0, 0)
    
    def _parse_forecast_data(self, data, days):
        """Parse OpenWeatherMap forecast response."""
        try:
            forecast_list = []
            
            for item in data['list'][:days * 8]:  # 8 forecasts per day (3-hour intervals)
                forecast_list.append({
                    'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'wind_speed': item['wind'].get('speed', 0) * 3.6,
                    'precipitation': item.get('rain', {}).get('3h', 0),
                    'weather_condition': item['weather'][0]['main'],
                    'fire_weather_index': self.calculate_fire_weather_index({
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'wind_speed': item['wind'].get('speed', 0) * 3.6,
                        'precipitation_24h': item.get('rain', {}).get('3h', 0) * 8
                    })['fire_weather_index']
                })
            
            return {
                'location': f"{data['city']['name']}, {data['city']['country']}",
                'forecast': forecast_list,
                'daily_summary': self._create_daily_summary(forecast_list, days)
            }
            
        except Exception as e:
            print(f"Error parsing forecast data: {str(e)}")
            return self._generate_synthetic_forecast(0, 0, days)
    
    def _create_daily_summary(self, forecast_list, days):
        """Create daily weather summary from hourly forecasts."""
        daily_data = {}
        
        for item in forecast_list:
            date = item['datetime'].split('T')[0]
            
            if date not in daily_data:
                daily_data[date] = {
                    'temperatures': [],
                    'humidity': [],
                    'wind_speeds': [],
                    'precipitation': [],
                    'fire_indices': []
                }
            
            daily_data[date]['temperatures'].append(item['temperature'])
            daily_data[date]['humidity'].append(item['humidity'])
            daily_data[date]['wind_speeds'].append(item['wind_speed'])
            daily_data[date]['precipitation'].append(item['precipitation'])
            daily_data[date]['fire_indices'].append(item['fire_weather_index'])
        
        # Calculate daily averages and extremes
        summary = []
        for date, data in list(daily_data.items())[:days]:
            summary.append({
                'date': date,
                'temp_min': min(data['temperatures']),
                'temp_max': max(data['temperatures']),
                'temp_avg': sum(data['temperatures']) / len(data['temperatures']),
                'humidity_avg': sum(data['humidity']) / len(data['humidity']),
                'wind_max': max(data['wind_speeds']),
                'precipitation_total': sum(data['precipitation']),
                'fire_weather_index_max': max(data['fire_indices']),
                'fire_weather_index_avg': sum(data['fire_indices']) / len(data['fire_indices'])
            })
        
        return summary
    
    def _generate_synthetic_weather(self, lat, lon):
        """Generate realistic synthetic weather data when API is not available."""
        # Generate weather based on location and season
        month = datetime.now().month
        
        # Seasonal temperature adjustment
        if 6 <= month <= 8:  # Summer
            base_temp = np.random.normal(28, 6)
            base_humidity = np.random.normal(45, 15)
        elif 12 <= month <= 2:  # Winter
            base_temp = np.random.normal(15, 8)
            base_humidity = np.random.normal(65, 20)
        else:  # Spring/Fall
            base_temp = np.random.normal(22, 7)
            base_humidity = np.random.normal(55, 18)
        
        # Ensure realistic ranges
        temperature = np.clip(base_temp, -5, 45)
        humidity = np.clip(base_humidity, 20, 95)
        
        return {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'pressure': round(np.random.normal(1013, 20), 1),
            'wind_speed': round(np.random.gamma(2, 3), 1),
            'wind_direction': round(np.random.uniform(0, 360), 0),
            'visibility': round(np.random.uniform(8, 15), 1),
            'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain'], p=[0.6, 0.3, 0.1]),
            'weather_description': 'Clear sky',
            'precipitation_1h': round(np.random.exponential(0.5) if np.random.random() < 0.1 else 0, 1),
            'precipitation_24h': round(np.random.exponential(2) if np.random.random() < 0.3 else 0, 1),
            'cloud_cover': round(np.random.uniform(0, 70), 0),
            'timestamp': datetime.now().isoformat(),
            'location': f"Synthetic Location ({lat:.2f}, {lon:.2f})"
        }
    
    def _generate_synthetic_forecast(self, lat, lon, days):
        """Generate synthetic weather forecast."""
        forecast_list = []
        
        for i in range(days * 8):  # 8 forecasts per day
            base_time = datetime.now() + timedelta(hours=i * 3)
            weather_data = self._generate_synthetic_weather(lat, lon)
            
            forecast_item = {
                'datetime': base_time.isoformat(),
                'temperature': weather_data['temperature'] + np.random.normal(0, 2),
                'humidity': np.clip(weather_data['humidity'] + np.random.normal(0, 5), 20, 95),
                'wind_speed': max(0, weather_data['wind_speed'] + np.random.normal(0, 2)),
                'precipitation': weather_data['precipitation_1h'],
                'weather_condition': weather_data['weather_condition'],
                'fire_weather_index': 0  # Will be calculated below
            }
            
            # Calculate fire weather index for forecast
            fwi_result = self.calculate_fire_weather_index({
                'temperature': forecast_item['temperature'],
                'humidity': forecast_item['humidity'],
                'wind_speed': forecast_item['wind_speed'],
                'precipitation_24h': forecast_item['precipitation'] * 8
            })
            
            forecast_item['fire_weather_index'] = fwi_result['fire_weather_index']
            forecast_list.append(forecast_item)
        
        return {
            'location': f"Synthetic Location ({lat:.2f}, {lon:.2f})",
            'forecast': forecast_list,
            'daily_summary': self._create_daily_summary(forecast_list, days)
        }
    
    def get_historical_weather_trends(self, lat, lon, days_back=30):
        """
        Get historical weather trends for risk pattern analysis.
        This would typically use historical weather API, but we'll generate
        representative historical data for demonstration.
        """
        historical_data = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            weather = self._generate_synthetic_weather(lat, lon)
            
            # Add some seasonal and historical variation
            temp_trend = -0.1 * i + np.random.normal(0, 3)  # Slight cooling trend
            weather['temperature'] += temp_trend
            
            historical_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature_max': weather['temperature'] + 5,
                'temperature_min': weather['temperature'] - 5,
                'humidity_avg': weather['humidity'],
                'wind_max': weather['wind_speed'] + 5,
                'precipitation': weather['precipitation_24h'],
                'fire_weather_index': self.calculate_fire_weather_index(weather)['fire_weather_index']
            })
        
        return {
            'location': f"Historical data ({lat:.2f}, {lon:.2f})",
            'period': f"{days_back} days",
            'data': list(reversed(historical_data))  # Chronological order
        }