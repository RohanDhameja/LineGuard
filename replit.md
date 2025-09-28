# Overview

This is a comprehensive Vegetation Detection Near Power Lines application designed to prevent wildfire risks. The system combines satellite imagery, LiDAR data, and machine learning to automatically identify vegetation hazards around power line infrastructure. It provides real-time monitoring, risk assessment, and early warning capabilities through an interactive Streamlit web interface with mobile-responsive design.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with mobile-responsive design
- **Visualization**: Folium for interactive maps, Plotly for charts and analytics
- **User Interface**: Clean, intuitive dashboard with sidebar navigation and real-time updates
- **Mobile Support**: CSS media queries and responsive layout optimization

## Backend Architecture
- **Core Processing**: Modular utility system with specialized processors for different data types
- **Data Pipeline**: Sequential processing from raw data ingestion to risk assessment output
- **Machine Learning**: Ensemble models (Random Forest, Gradient Boosting) for vegetation classification and risk prediction
- **Geospatial Processing**: Advanced geometric operations for distance calculations and coordinate transformations

## Data Storage Solutions
- **Primary Database**: PostgreSQL for structured data storage including analysis sessions, alerts, and historical monitoring data
- **Temporary Storage**: File system for cached satellite imagery and processing artifacts
- **Configuration Storage**: JSON-based parameter storage and model metadata

## Authentication and Authorization
- **Service Account Integration**: Google Earth Engine authentication for satellite data access
- **Environment Variables**: Secure credential management for API keys and database connections
- **No User Authentication**: Currently operates as a single-tenant application

## Key Architectural Components

### Data Processing Pipeline
1. **Data Ingestion**: Multi-source data loading (satellite, LiDAR, power line infrastructure)
2. **Geospatial Analysis**: Distance calculations, coordinate transformations, and geometric operations
3. **Machine Learning**: Vegetation classification and risk scoring
4. **Visualization**: Interactive maps and analytical charts

### Risk Assessment Engine
- **Multi-factor Analysis**: Combines vegetation density, proximity to power lines, environmental conditions
- **Threshold-based Alerting**: Configurable risk levels (low, moderate, high, critical)
- **Predictive Modeling**: Growth prediction and trend analysis

### Monitoring System
- **Real-time Alerts**: Automated threshold monitoring with severity classification
- **Scheduled Tasks**: Periodic data updates and analysis execution
- **Event Logging**: Comprehensive activity tracking and audit trails

# External Dependencies

## Third-party Services
- **Google Earth Engine**: Satellite imagery and geospatial data access
- **Weather APIs**: Real-time environmental data (OpenWeatherMap integration)
- **Sentinel-2 (ESA)**: High-resolution multispectral satellite imagery

## Core Python Libraries
- **Streamlit**: Web application framework and UI components
- **Folium**: Interactive mapping and geospatial visualization
- **Plotly**: Advanced charting and data visualization
- **GeoPandas**: Geospatial data manipulation and analysis
- **Scikit-learn**: Machine learning models and preprocessing
- **Shapely**: Geometric operations and spatial analysis
- **PyProj**: Coordinate system transformations

## Database and Storage
- **PostgreSQL**: Primary database for persistent data storage
- **psycopg2**: PostgreSQL database adapter for Python
- **Pandas**: Data manipulation and analysis framework

## Development and Deployment
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific computing and spatial algorithms
- **Joblib**: Model serialization and parallel processing
- **Requests**: HTTP client for external API integration

## Optional Integrations
- **Firebase**: Alternative authentication and real-time database option
- **Email Services**: Alert notification delivery system
- **Additional Weather Services**: Multiple weather data source support