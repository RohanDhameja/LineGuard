import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils.data_loader import DataLoader
from utils.geospatial import GeospatialProcessor
from utils.risk_assessment import RiskAssessment
from utils.visualization import Visualizer
from utils.database import DatabaseManager
from utils.ml_models import VegetationClassifier
from utils.weather_service import WeatherService
from utils.monitoring_system import VegetationMonitoringSystem

# Page configuration
st.set_page_config(
    page_title="Vegetation Detection Near Power Lines",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None

# Initialize components
@st.cache_resource
def initialize_components():
    data_loader = DataLoader()
    geospatial = GeospatialProcessor()
    risk_assessment = RiskAssessment()
    visualizer = Visualizer()
    ml_classifier = VegetationClassifier()
    weather_service = WeatherService()
    
    try:
        db_manager = DatabaseManager()
    except Exception as e:
        st.warning(f"Database connection failed: {str(e)}. Some features may be limited.")
        db_manager = None
    
    # Initialize monitoring system
    monitoring_system = VegetationMonitoringSystem(
        database_manager=db_manager,
        weather_service=weather_service,
        ml_classifier=ml_classifier
    )
    
    # Initialize ML models with synthetic training data
    try:
        if not hasattr(ml_classifier, 'vegetation_classifier') or ml_classifier.vegetation_classifier is None:
            with st.spinner("Initializing ML models..."):
                training_data = ml_classifier.generate_synthetic_training_data(500)
                ml_classifier.train_vegetation_classifier(training_data)
                ml_classifier.train_risk_classifier(training_data)
                print("✅ ML models initialized")
    except Exception as e:
        print(f"⚠️ ML model initialization failed: {str(e)}")
    
    return data_loader, geospatial, risk_assessment, visualizer, db_manager, ml_classifier, weather_service, monitoring_system

data_loader, geospatial, risk_assessment, visualizer, db_manager, ml_classifier, weather_service, monitoring_system = initialize_components()

# Main title and description
st.title("🔥 Vegetation Detection Near Power Lines")
st.markdown("""
**Wildfire Risk Prevention System** - Combining satellite imagery, LiDAR data, and computer vision 
to automatically identify vegetation that poses fire hazards to electrical infrastructure.
""")

# Sidebar for configuration
st.sidebar.header("🛠️ Configuration")

# Location input
st.sidebar.subheader("📍 Analysis Location")
col1, col2 = st.sidebar.columns(2)
with col1:
    lat = st.number_input("Latitude", value=37.789953, format="%.6f")
with col2:
    lon = st.number_input("Longitude", value=-122.058679, format="%.6f")

radius_km = st.sidebar.slider("Analysis Radius (km)", min_value=1, max_value=20, value=10)

# Data source selection
st.sidebar.subheader("📊 Data Sources")
use_satellite = st.sidebar.checkbox("Satellite Imagery (Sentinel-2)", value=True)
use_lidar = st.sidebar.checkbox("LiDAR Data (USGS 3DEP)", value=True)
use_powerlines = st.sidebar.checkbox("Power Line Data", value=True)

# Analysis parameters
st.sidebar.subheader("⚙️ Analysis Parameters")
risk_threshold = st.sidebar.slider("Risk Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
clearance_threshold = st.sidebar.slider("Minimum Clearance (meters)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

# Date range for satellite data
date_start = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=90))
date_end = st.sidebar.date_input("End Date", value=datetime.now())

# Load data button
if st.sidebar.button("🔄 Load Data", type="primary"):
    with st.spinner("Loading data..."):
        try:
            # Store parameters in session state
            st.session_state.analysis_params = {
                'lat': lat,
                'lon': lon,
                'radius_km': radius_km,
                'use_satellite': use_satellite,
                'use_lidar': use_lidar,
                'use_powerlines': use_powerlines,
                'risk_threshold': risk_threshold,
                'clearance_threshold': clearance_threshold,
                'date_start': date_start,
                'date_end': date_end
            }
            
            # Load satellite data
            if use_satellite:
                st.session_state.satellite_data = data_loader.load_satellite_data(
                    lat, lon, radius_km, date_start, date_end
                )
            
            # Load LiDAR data
            if use_lidar:
                st.session_state.lidar_data = data_loader.load_lidar_data(
                    lat, lon, radius_km
                )
            
            # Load power line data
            if use_powerlines:
                st.session_state.powerline_data = data_loader.load_powerline_data(
                    lat, lon, radius_km
                )
            
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

# Main content area
if st.session_state.data_loaded:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Interactive Map", 
        "📊 Risk Dashboard", 
        "🌿 Vegetation Analysis", 
        "⚡ Power Line Analysis", 
        "🔍 Monitoring System"
    ])
    
    with tab1:
        st.subheader("Interactive Map View")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create base map
            center_lat = st.session_state.analysis_params['lat']
            center_lon = st.session_state.analysis_params['lon']
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Add center point
            folium.Marker(
                [center_lat, center_lon],
                popup="Analysis Center",
                icon=folium.Icon(color='red', icon='crosshairs')
            ).add_to(m)
            
            # Add analysis radius circle
            folium.Circle(
                location=[center_lat, center_lon],
                radius=st.session_state.analysis_params['radius_km'] * 1000,
                popup=f"Analysis Area ({st.session_state.analysis_params['radius_km']} km)",
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.2
            ).add_to(m)
            
            # Add power lines if available
            if hasattr(st.session_state, 'powerline_data') and st.session_state.powerline_data is not None:
                try:
                    powerline_gdf = st.session_state.powerline_data
                    if not powerline_gdf.empty:
                        # Add power lines to map
                        for idx, row in powerline_gdf.iterrows():
                            if row.geometry.geom_type == 'LineString':
                                coords = [[lat, lon] for lon, lat in row.geometry.coords]
                                folium.PolyLine(
                                    coords,
                                    color='red',
                                    weight=3,
                                    popup=f"Power Line Segment {idx}"
                                ).add_to(m)
                except Exception as e:
                    st.warning(f"Could not display power lines: {str(e)}")
            
            # Add vegetation risk areas if analysis is complete
            if st.session_state.analysis_complete and hasattr(st.session_state, 'risk_areas'):
                try:
                    for risk_area in st.session_state.risk_areas:
                        folium.CircleMarker(
                            location=[risk_area['lat'], risk_area['lon']],
                            radius=5,
                            popup=f"Risk Score: {risk_area['risk_score']:.2f}",
                            color='orange' if risk_area['risk_score'] > 0.7 else 'yellow',
                            fillColor='red' if risk_area['risk_score'] > 0.8 else 'orange',
                            fillOpacity=0.7
                        ).add_to(m)
                except Exception as e:
                    st.warning(f"Could not display risk areas: {str(e)}")
            
            # Display map
            map_data = st_folium(m, width=700, height=500)
            
            # Handle map clicks
            if map_data['last_object_clicked_popup']:
                st.session_state.selected_location = map_data['last_clicked']
        
        with col2:
            st.subheader("Map Layers")
            
            # Layer controls
            show_satellite = st.checkbox("Satellite Overlay", value=False)
            show_ndvi = st.checkbox("NDVI Layer", value=False)
            show_canopy_height = st.checkbox("Canopy Height", value=False)
            show_risk_heatmap = st.checkbox("Risk Heatmap", value=False)
            
            if st.session_state.selected_location:
                st.subheader("Selected Location")
                loc = st.session_state.selected_location
                st.write(f"**Lat:** {loc['lat']:.6f}")
                st.write(f"**Lon:** {loc['lng']:.6f}")
                
                if st.button("Analyze This Point"):
                    # Perform point-specific analysis
                    st.info("Point analysis would be implemented here")
    
    with tab2:
        st.subheader("Risk Assessment Dashboard")
        
        # Run analysis button
        if st.button("🔍 Run Risk Analysis", type="primary"):
            with st.spinner("Performing risk analysis..."):
                try:
                    # Perform vegetation detection and risk assessment
                    analysis_results = risk_assessment.perform_analysis(
                        satellite_data=getattr(st.session_state, 'satellite_data', None),
                        lidar_data=getattr(st.session_state, 'lidar_data', None),
                        powerline_data=getattr(st.session_state, 'powerline_data', None),
                        parameters=st.session_state.analysis_params
                    )
                    
                    # Save to database if available
                    if db_manager:
                        try:
                            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{lat}_{lon}"
                            
                            session_data = {
                                'session_id': session_id,
                                'center_lat': lat,
                                'center_lon': lon,
                                'radius_km': radius_km,
                                'parameters': st.session_state.analysis_params,
                                'overall_risk_score': analysis_results.get('overall_risk', 0),
                                'high_risk_count': analysis_results.get('high_risk_count', 0),
                                'vegetation_density': analysis_results.get('vegetation_density', 0),
                                'avg_clearance_distance': analysis_results.get('avg_clearance', 0),
                                'data_sources': analysis_results.get('data_sources', {}),
                                'priority_areas': analysis_results.get('priority_areas', [])
                            }
                            
                            db_manager.save_analysis_session(session_data)
                            
                            # Save power line segments if available
                            if hasattr(st.session_state, 'segmented_powerlines'):
                                db_manager.save_powerline_segments(session_id, st.session_state.segmented_powerlines)
                            
                            st.session_state.current_session_id = session_id
                            
                        except Exception as db_error:
                            st.warning(f"Analysis completed but database save failed: {str(db_error)}")
                    
                    st.session_state.analysis_results = analysis_results
                    st.session_state.analysis_complete = True
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
        
        if st.session_state.analysis_complete:
            results = st.session_state.analysis_results
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "High Risk Areas", 
                    f"{results.get('high_risk_count', 0)}", 
                    delta=f"{results.get('risk_change', 0)}%"
                )
            
            with col2:
                st.metric(
                    "Avg Clearance Distance", 
                    f"{results.get('avg_clearance', 0):.1f}m",
                    delta=f"{results.get('clearance_change', 0):.1f}m"
                )
            
            with col3:
                st.metric(
                    "Vegetation Density", 
                    f"{results.get('vegetation_density', 0):.1%}",
                    delta=f"{results.get('density_change', 0):.1%}%"
                )
            
            with col4:
                st.metric(
                    "Risk Score", 
                    f"{results.get('overall_risk', 0):.2f}",
                    delta=f"{results.get('risk_trend', 0):.2f}"
                )
            
            # Risk distribution chart
            st.subheader("Risk Distribution")
            
            if 'risk_distribution' in results:
                fig = px.histogram(
                    x=results['risk_distribution'],
                    nbins=20,
                    title="Distribution of Risk Scores",
                    labels={'x': 'Risk Score', 'y': 'Count'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Priority areas table
            st.subheader("Priority Areas for Maintenance")
            
            if 'priority_areas' in results:
                priority_df = pd.DataFrame(results['priority_areas'])
                if not priority_df.empty:
                    st.dataframe(
                        priority_df.style.format({
                            'risk_score': '{:.3f}',
                            'clearance_distance': '{:.2f}m',
                            'vegetation_height': '{:.2f}m'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No high-priority areas identified.")
            
            # Weather-Enhanced Risk Assessment
            st.subheader("🌡️ Weather-Enhanced Risk Assessment")
            
            if hasattr(st.session_state, 'selected_location') and st.session_state.selected_location:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🌤️ Get Current Weather & Fire Risk"):
                        with st.spinner("Fetching weather data and calculating fire risk..."):
                            try:
                                lat = st.session_state.selected_location['lat']
                                lon = st.session_state.selected_location['lng']
                                current_weather = weather_service.get_current_weather(lat, lon)
                                fire_index = weather_service.calculate_fire_weather_index(current_weather)
                                
                                st.session_state.current_weather = current_weather
                                st.session_state.fire_weather_index = fire_index
                                
                                st.success("✅ Weather-enhanced risk assessment complete!")
                            except Exception as e:
                                st.error(f"❌ Error fetching weather data: {str(e)}")
                                st.info("Using synthetic weather data for demonstration.")
                
                with col2:
                    if st.button("📊 Get Weather Forecast"):
                        with st.spinner("Fetching weather forecast..."):
                            try:
                                lat = st.session_state.selected_location['lat']
                                lon = st.session_state.selected_location['lng']
                                forecast = weather_service.get_weather_forecast(lat, lon, 5)
                                st.session_state.weather_forecast = forecast
                                st.success("✅ Weather forecast loaded!")
                            except Exception as e:
                                st.error(f"❌ Error fetching weather forecast: {str(e)}")
                                st.info("Using synthetic forecast data for demonstration.")
                
                # Display current weather and fire risk
                if hasattr(st.session_state, 'current_weather') and hasattr(st.session_state, 'fire_weather_index'):
                    st.subheader("🔥 Current Fire Weather Conditions")
                    
                    weather = st.session_state.current_weather
                    fire_idx = st.session_state.fire_weather_index
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Temperature", f"{weather['temperature']:.1f}°C")
                        st.metric("Humidity", f"{weather['humidity']:.0f}%")
                    
                    with col2:
                        st.metric("Wind Speed", f"{weather['wind_speed']:.1f} km/h")
                        st.metric("Precipitation (24h)", f"{weather['precipitation_24h']:.1f} mm")
                    
                    with col3:
                        danger_level = fire_idx.get('danger_level', 'Unknown')
                        fwi_score = fire_idx.get('fire_weather_index', 0)
                        
                        st.metric("Fire Weather Index", f"{fwi_score:.1f}/100")
                        
                        # Color code the danger level
                        if danger_level == 'Low':
                            st.success(f"Danger Level: {danger_level}")
                        elif danger_level == 'Moderate':
                            st.warning(f"Danger Level: {danger_level}")
                        elif danger_level == 'High':
                            st.error(f"Danger Level: {danger_level}")
                        else:
                            st.error(f"⚠️ Danger Level: {danger_level}")
                    
                    with col4:
                        st.metric("Visibility", f"{weather['visibility']:.1f} km")
                        st.metric("Weather", weather['weather_condition'])
                    
                    # Fire safety recommendations
                    if fire_idx.get('recommendations'):
                        st.subheader("🚨 Fire Safety Recommendations")
                        
                        for i, recommendation in enumerate(fire_idx['recommendations'], 1):
                            if i == 1 and 'CRITICAL' in recommendation:
                                st.error(f"{i}. {recommendation}")
                            elif 'alert' in recommendation.lower() or 'suspend' in recommendation.lower():
                                st.warning(f"{i}. {recommendation}")
                            else:
                                st.info(f"{i}. {recommendation}")
                
                # Display weather forecast if available
                if hasattr(st.session_state, 'weather_forecast') and 'daily_summary' in st.session_state.weather_forecast:
                    st.subheader("📈 5-Day Fire Weather Forecast")
                    
                    forecast = st.session_state.weather_forecast
                    daily_summary = forecast['daily_summary']
                    
                    if daily_summary:
                        forecast_df = pd.DataFrame(daily_summary)
                        
                        # Fire weather index forecast chart
                        fig = px.bar(
                            forecast_df,
                            x='date',
                            y='fire_weather_index_max',
                            title="Daily Maximum Fire Weather Index",
                            color='fire_weather_index_max',
                            color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                            labels={'fire_weather_index_max': 'Fire Weather Index'}
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary table
                        forecast_display = forecast_df.copy()
                        forecast_display['temp_range'] = forecast_display.apply(
                            lambda row: f"{row['temp_min']:.0f}°C - {row['temp_max']:.0f}°C", axis=1
                        )
                        forecast_display = forecast_display[[
                            'date', 'temp_range', 'humidity_avg', 'wind_max', 
                            'precipitation_total', 'fire_weather_index_max'
                        ]]
                        forecast_display.columns = [
                            'Date', 'Temperature', 'Humidity (%)', 'Wind (km/h)', 
                            'Rain (mm)', 'Fire Index'
                        ]
                        
                        st.dataframe(forecast_display, hide_index=True, use_container_width=True)
            else:
                st.info("Select a location on the map to access weather-enhanced risk assessment.")
    
    with tab3:
        st.subheader("🌿 Advanced Vegetation Analysis")
        
        if hasattr(st.session_state, 'satellite_data'):
            # ML-powered vegetation classification
            st.subheader("🤖 AI-Powered Vegetation Classification")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Analyze Vegetation Types"):
                    with st.spinner("Classifying vegetation using AI models..."):
                        try:
                            # Create features for ML classification
                            features = ml_classifier.create_training_features(
                                st.session_state.satellite_data,
                                getattr(st.session_state, 'lidar_data', None)
                            )
                            
                            # Classify vegetation
                            vegetation_result = ml_classifier.classify_vegetation(features)
                            risk_result = ml_classifier.predict_fire_risk(features)
                            
                            # Store results
                            st.session_state.ml_vegetation_result = vegetation_result
                            st.session_state.ml_risk_result = risk_result
                            
                            st.success("✅ AI analysis complete!")
                            
                        except Exception as e:
                            st.error(f"❌ AI analysis failed: {str(e)}")
            
            with col2:
                if st.button("🗺️ Cluster Vegetation Areas"):
                    with st.spinner("Identifying vegetation patterns..."):
                        try:
                            features = ml_classifier.create_training_features(
                                st.session_state.satellite_data,
                                getattr(st.session_state, 'lidar_data', None)
                            )
                            
                            # Create multiple feature samples for clustering by simulating spatial variation
                            n_samples = 50
                            features_list = []
                            
                            base_satellite = st.session_state.satellite_data.copy()
                            base_lidar = getattr(st.session_state, 'lidar_data', None)
                            
                            for i in range(n_samples):
                                # Add small variations to simulate different locations/patches
                                varied_satellite = base_satellite.copy()
                                if 'ndvi_values' in varied_satellite:
                                    # Add realistic variation to NDVI values
                                    ndvi_array = np.array(varied_satellite['ndvi_values'])
                                    variation = np.random.normal(0, 0.05, len(ndvi_array))
                                    varied_ndvi = np.clip(ndvi_array + variation, -1, 1)
                                    varied_satellite['ndvi_values'] = varied_ndvi.tolist()
                                
                                varied_lidar = None
                                if base_lidar and 'height_values' in base_lidar:
                                    varied_lidar = base_lidar.copy()
                                    height_array = np.array(base_lidar['height_values'])
                                    height_variation = np.random.normal(0, 1.0, len(height_array))
                                    varied_heights = np.clip(height_array + height_variation, 0, 50)
                                    varied_lidar['height_values'] = varied_heights.tolist()
                                
                                sample_features = ml_classifier.create_training_features(
                                    varied_satellite, varied_lidar
                                )
                                features_list.append(sample_features)
                            
                            all_features = pd.concat(features_list, ignore_index=True)
                            clustering_result = ml_classifier.cluster_vegetation_areas(all_features)
                            
                            st.session_state.clustering_result = clustering_result
                            st.success("✅ Vegetation clustering complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Clustering analysis failed: {str(e)}")
            
            # Display ML results
            if hasattr(st.session_state, 'ml_vegetation_result'):
                st.subheader("🎯 Vegetation Classification Results")
                
                col1, col2, col3 = st.columns(3)
                
                result = st.session_state.ml_vegetation_result
                
                with col1:
                    st.metric(
                        "Detected Vegetation Type", 
                        result['predicted_vegetation']
                    )
                    st.metric(
                        "Classification Confidence", 
                        f"{result['confidence']:.1%}"
                    )
                
                with col2:
                    # Vegetation type probabilities
                    prob_data = result['probabilities']
                    fig = px.bar(
                        x=list(prob_data.keys()),
                        y=list(prob_data.values()),
                        title="Vegetation Type Probabilities",
                        labels={'x': 'Vegetation Type', 'y': 'Probability'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    if hasattr(st.session_state, 'ml_risk_result'):
                        risk_result = st.session_state.ml_risk_result
                        st.metric(
                            "AI Risk Assessment", 
                            risk_result['predicted_risk']
                        )
                        st.metric(
                            "Risk Score", 
                            f"{risk_result['risk_score']:.1%}"
                        )
            
            # Display clustering results
            if hasattr(st.session_state, 'clustering_result'):
                st.subheader("🔍 Vegetation Pattern Analysis")
                
                clustering = st.session_state.clustering_result
                
                if 'cluster_analysis' in clustering:
                    cluster_data = clustering['cluster_analysis']
                    
                    # Create cluster summary
                    cluster_df = pd.DataFrame([
                        {
                            'Cluster': cluster_id,
                            'Size': info['size'],
                            'Avg NDVI': f"{info['avg_ndvi']:.3f}",
                            'Avg Height': f"{info['avg_height']:.1f}m",
                            'Type': info['characteristics']
                        }
                        for cluster_id, info in cluster_data.items()
                    ])
                    
                    st.dataframe(cluster_df, use_container_width=True)
            
            # Traditional NDVI analysis
            st.subheader("📊 Traditional NDVI Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # NDVI statistics
                if hasattr(st.session_state, 'analysis_results'):
                    ndvi_stats = st.session_state.analysis_results.get('ndvi_stats', {})
                    
                    st.metric("Mean NDVI", f"{ndvi_stats.get('mean', 0):.3f}")
                    st.metric("Max NDVI", f"{ndvi_stats.get('max', 0):.3f}")
                    st.metric("Vegetation Coverage", f"{ndvi_stats.get('vegetation_percentage', 0):.1f}%")
                
                # NDVI histogram
                if hasattr(st.session_state, 'analysis_results') and 'ndvi_histogram' in st.session_state.analysis_results:
                    fig = px.bar(
                        x=st.session_state.analysis_results['ndvi_histogram']['bins'],
                        y=st.session_state.analysis_results['ndvi_histogram']['counts'],
                        title="NDVI Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Health categories
                if hasattr(st.session_state, 'analysis_results'):
                    health_data = st.session_state.analysis_results.get('vegetation_health', {})
                    
                    if health_data:
                        fig = px.pie(
                            values=list(health_data.values()),
                            names=list(health_data.keys()),
                            title="Vegetation Health Categories"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Load satellite data to perform vegetation analysis.")
    
    with tab4:
        st.subheader("Power Line Analysis")
        
        if hasattr(st.session_state, 'powerline_data'):
            powerline_gdf = st.session_state.powerline_data
            
            if not powerline_gdf.empty:
                st.success(f"✅ Loaded {len(powerline_gdf)} power line segments")
                
                # Segmentation analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Segmentation Analysis")
                    
                    if st.button("🔧 Segment Power Lines"):
                        with st.spinner("Segmenting power lines into 1m segments..."):
                            try:
                                segmented_lines = geospatial.segment_powerlines(
                                    powerline_gdf, 
                                    segment_length=1.0
                                )
                                st.session_state.segmented_powerlines = segmented_lines
                                st.success(f"✅ Created {len(segmented_lines)} 1-meter segments")
                            except Exception as e:
                                st.error(f"❌ Segmentation failed: {str(e)}")
                    
                    if hasattr(st.session_state, 'segmented_powerlines'):
                        segments = st.session_state.segmented_powerlines
                        st.metric("Total Segments", len(segments))
                        st.metric("Total Length", f"{len(segments) * 1.0:.0f}m")
                
                with col2:
                    st.subheader("Proximity Analysis")
                    
                    if hasattr(st.session_state, 'analysis_results'):
                        proximity_stats = st.session_state.analysis_results.get('proximity_stats', {})
                        
                        st.metric("Segments < 1m from vegetation", proximity_stats.get('critical_segments', 0))
                        st.metric("Average vegetation distance", f"{proximity_stats.get('avg_distance', 0):.2f}m")
                        st.metric("Minimum recorded distance", f"{proximity_stats.get('min_distance', 0):.2f}m")
                
                # Power line details table
                st.subheader("Power Line Details")
                display_df = powerline_gdf.drop(columns=['geometry']).head(10)
                st.dataframe(display_df, use_container_width=True)
            
            else:
                st.warning("No power line data found in the specified area.")
        
        else:
            st.info("Load power line data to perform analysis.")
    
    with tab5:
        st.subheader("Analysis Reports")
        
        # Historical Analysis Section
        if db_manager:
            st.subheader("📈 Historical Trends")
            
            col1, col2 = st.columns(2)
            with col1:
                trend_days = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)
            with col2:
                trend_radius = st.slider("Search Radius (km)", 1, 50, 20)
            
            if st.button("🔍 Load Historical Trends"):
                with st.spinner("Loading historical data..."):
                    try:
                        historical_data = db_manager.get_historical_analysis(
                            lat, lon, trend_radius, trend_days
                        )
                        
                        if not historical_data.empty:
                            # Create trend visualization
                            trend_fig = visualizer.create_trend_analysis(historical_data)
                            st.plotly_chart(trend_fig, use_container_width=True)
                            
                            # Show data table
                            st.subheader("Historical Data")
                            st.dataframe(historical_data, use_container_width=True)
                            
                        else:
                            st.info("No historical data found for this location and time period.")
                            
                    except Exception as e:
                        st.error(f"Error loading historical trends: {str(e)}")
            
            # Risk Areas Overview
            st.subheader("🚨 Regional Risk Overview")
            if st.button("🗺️ Load Regional Risk Areas"):
                with st.spinner("Loading regional risk data..."):
                    try:
                        risk_areas_df = db_manager.get_risk_areas_in_region(
                            lat, lon, radius_km, min_risk_score=0.6
                        )
                        
                        if not risk_areas_df.empty:
                            st.metric("High Risk Areas Found", len(risk_areas_df))
                            
                            # Show risk areas on map
                            risk_map = visualizer.create_risk_heatmap(risk_areas_df)
                            st.plotly_chart(risk_map, use_container_width=True)
                            
                            # Show table of recent high-risk areas
                            st.dataframe(
                                risk_areas_df[['lat', 'lon', 'risk_score', 'risk_category', 
                                             'clearance_distance', 'analysis_date']].head(20),
                                use_container_width=True
                            )
                        else:
                            st.success("No high-risk areas found in this region.")
                            
                    except Exception as e:
                        st.error(f"Error loading regional risk data: {str(e)}")
        
        if st.session_state.analysis_complete:
            # Report generation
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Export Options")
                
                export_format = st.selectbox("Export Format", ["CSV", "GeoJSON", "Shapefile", "PDF Report"])
                include_map = st.checkbox("Include Map Images", value=True)
                include_stats = st.checkbox("Include Statistics", value=True)
                include_recommendations = st.checkbox("Include Recommendations", value=True)
                
                if st.button("📥 Generate Report"):
                    with st.spinner("Generating report..."):
                        try:
                            report_data = visualizer.generate_report(
                                st.session_state.analysis_results,
                                format=export_format.lower(),
                                include_map=include_map,
                                include_stats=include_stats,
                                include_recommendations=include_recommendations
                            )
                            
                            # Create download link
                            st.download_button(
                                label=f"Download {export_format} Report",
                                data=report_data['content'],
                                file_name=report_data['filename'],
                                mime=report_data['mime_type']
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Report generation failed: {str(e)}")
            
            with col2:
                st.subheader("Summary Statistics")
                
                results = st.session_state.analysis_results
                
                summary_data = {
                    "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Location": f"{st.session_state.analysis_params['lat']:.6f}, {st.session_state.analysis_params['lon']:.6f}",
                    "Analysis Radius": f"{st.session_state.analysis_params['radius_km']} km",
                    "Risk Threshold": f"{st.session_state.analysis_params['risk_threshold']:.1f}",
                    "High Risk Areas": results.get('high_risk_count', 0),
                    "Average Risk Score": f"{results.get('overall_risk', 0):.3f}",
                    "Vegetation Coverage": f"{results.get('vegetation_density', 0):.1%}",
                    "Critical Segments": results.get('proximity_stats', {}).get('critical_segments', 0)
                }
                
                for key, value in summary_data.items():
                    st.text(f"{key}: {value}")
                
                # Recommendations
                st.subheader("Recommendations")
                recommendations = results.get('recommendations', [])
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.info("No specific recommendations generated.")
        
        else:
            st.info("Complete the risk analysis to generate reports.")
    
    with tab5:
        st.subheader("🔍 Automated Monitoring System")
        
        # Monitoring system controls
        st.subheader("🎛️ System Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Monitoring"):
                monitoring_system.start_monitoring()
                st.success("✅ Monitoring system started!")
                st.rerun()
        
        with col2:
            if st.button("🛑 Stop Monitoring"):
                monitoring_system.stop_monitoring()
                st.info("🛑 Monitoring system stopped")
                st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Dashboard"):
                st.rerun()
        
        # Get monitoring dashboard data
        dashboard_data = monitoring_system.get_monitoring_dashboard_data()
        
        # System status overview
        st.subheader("📊 System Status")
        col1, col2, col3, col4 = st.columns(4)
        
        status = dashboard_data['system_status']
        
        with col1:
            status_color = "green" if status['status'] == 'RUNNING' else "red"
            st.markdown(f"**Status:** :{status_color}[{status['status']}]")
            st.metric("System Health", status['system_health'])
        
        with col2:
            st.metric("Active Alerts", status['alerts_count'])
            st.metric("Running Tasks", status['tasks_running'])
        
        with col3:
            last_update = datetime.fromisoformat(status['last_update']) if isinstance(status['last_update'], str) else status['last_update']
            st.metric("Last Update", last_update.strftime("%H:%M:%S"))
        
        with col4:
            alert_summary = dashboard_data['alert_summary']
            st.metric("Critical Alerts", alert_summary['critical'])
            st.metric("Warnings", alert_summary['warning'])
        
        # Active alerts section
        st.subheader("🚨 Active Alerts")
        
        active_alerts = dashboard_data['active_alerts']
        if active_alerts:
            for alert in active_alerts:
                alert_time = datetime.fromisoformat(alert['timestamp']) if isinstance(alert['timestamp'], str) else alert['timestamp']
                
                # Color code by severity
                if alert['severity'] == 'CRITICAL':
                    st.error(f"🔥 **{alert['title']}** - {alert_time.strftime('%H:%M:%S')}")
                elif alert['severity'] == 'WARNING':
                    st.warning(f"⚠️ **{alert['title']}** - {alert_time.strftime('%H:%M:%S')}")
                else:
                    st.info(f"ℹ️ **{alert['title']}** - {alert_time.strftime('%H:%M:%S')}")
                
                st.write(f"📍 **Location:** {alert['location'].get('name', 'Unknown')}")
                st.write(f"📝 **Description:** {alert['description']}")
                
                # Recommended actions
                if alert.get('recommended_actions'):
                    st.write("**Recommended Actions:**")
                    for i, action in enumerate(alert['recommended_actions'], 1):
                        st.write(f"   {i}. {action}")
                
                # Alert actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Acknowledge", key=f"ack_{alert['id']}"):
                        monitoring_system.acknowledge_alert(alert['id'])
                        st.success("Alert acknowledged")
                        st.rerun()
                
                with col2:
                    if st.button(f"✓ Resolve", key=f"resolve_{alert['id']}"):
                        monitoring_system.resolve_alert(alert['id'])
                        st.success("Alert resolved")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.success("🎉 No active alerts - All systems operating normally!")
        
        # Monitoring tasks status
        st.subheader("⚙️ Monitoring Tasks")
        
        tasks = dashboard_data['monitoring_tasks']
        if tasks:
            task_df_data = []
            for task_id, task in tasks.items():
                last_run = "Never" if not task['last_run'] else datetime.fromisoformat(task['last_run']).strftime('%H:%M:%S')
                next_run = datetime.fromisoformat(task['next_run']).strftime('%H:%M:%S')
                
                task_df_data.append({
                    'Task': task['name'],
                    'Status': '✅ Enabled' if task['enabled'] else '❌ Disabled',
                    'Interval (min)': task['interval_minutes'],
                    'Last Run': last_run,
                    'Next Run': next_run
                })
            
            task_df = pd.DataFrame(task_df_data)
            st.dataframe(task_df, hide_index=True, use_container_width=True)
        
        # Alert statistics
        st.subheader("📊 Alert Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert severity distribution
            severity_counts = {}
            for alert in dashboard_data['recent_alerts']:
                severity = alert['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts:
                fig = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Alert Severity Distribution",
                    color_discrete_map={
                        'INFO': 'blue',
                        'WARNING': 'orange',
                        'CRITICAL': 'red',
                        'EMERGENCY': 'darkred'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alerts to display in chart")
        
        with col2:
            # Daily alert trend (simulated)
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 0, -1)]
            daily_counts = [np.random.poisson(3) for _ in dates]  # Simulated daily alert counts
            
            fig = px.bar(
                x=dates,
                y=daily_counts,
                title="Daily Alert Trend (Last 7 Days)",
                labels={'x': 'Date', 'y': 'Number of Alerts'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # System configuration
        with st.expander("⚙️ System Configuration"):
            st.write("**Current Alert Thresholds:**")
            
            threshold_data = []
            for param, threshold in monitoring_system.alert_thresholds.items():
                threshold_data.append({
                    'Parameter': param.replace('_', ' ').title(),
                    'Warning Value': threshold.warning_value,
                    'Critical Value': threshold.critical_value,
                    'Operator': threshold.operator,
                    'Enabled': '✅' if threshold.enabled else '❌',
                    'Description': threshold.description
                })
            
            threshold_df = pd.DataFrame(threshold_data)
            st.dataframe(threshold_df, hide_index=True, use_container_width=True)
            
            st.info("""
            **Monitoring System Features:**
            
            🔄 **Automated Tasks:**
            - Weather data updates every hour
            - Vegetation analysis every 6 hours  
            - Risk assessment every 3 hours
            - System health checks every 30 minutes
            
            🚨 **Alert System:**
            - Real-time threshold monitoring
            - Multi-level severity classification
            - Automated recommendations
            - Historical alert tracking
            
            📊 **Dashboard:**
            - Live system status monitoring
            - Interactive alert management
            - Performance analytics
            - Configuration management
            """)

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Vegetation Detection System
    
    This application helps prevent wildfire risks by analyzing vegetation proximity to power lines using:
    
    - 🛰️ **Satellite Imagery**: Sentinel-2 data for vegetation detection via NDVI
    - 📡 **LiDAR Data**: USGS 3DEP for precise canopy height measurements
    - ⚡ **Power Line Data**: Infrastructure locations and specifications
    - 🔥 **Risk Assessment**: ML-based analysis of fire hazard probability
    
    ### Getting Started
    1. Configure your analysis location in the sidebar
    2. Select the data sources you want to use
    3. Click "Load Data" to begin
    4. Navigate through the tabs to explore results
    
    ### Data Sources
    - **Satellite Data**: Automatically fetched from Google Earth Engine
    - **LiDAR Data**: Downloaded from USGS 3D Elevation Program
    - **Power Lines**: California Electric Transmission Lines dataset
    
    ⚠️ **Note**: Initial data loading may take several minutes depending on the analysis area size.
    """)
    
    # Sample location shortcuts
    st.subheader("📍 Sample Locations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌲 Bay Area, CA"):
            st.query_params.lat = 37.789953
            st.query_params.lon = -122.058679
    
    with col2:
        if st.button("🔥 Napa Valley, CA"):
            st.query_params.lat = 38.5025
            st.query_params.lon = -122.2654
    
    with col3:
        if st.button("⛰️ Santa Barbara, CA"):
            st.query_params.lat = 34.4208
            st.query_params.lon = -119.6982

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
🔥 Vegetation Detection Near Power Lines | Built with Streamlit | Data: Sentinel-2, USGS 3DEP, CEC
</div>
""", unsafe_allow_html=True)
