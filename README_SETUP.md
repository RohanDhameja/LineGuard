# LineGuardAI Setup Guide

## Quick Start (5 minutes)

### 1. Install Python Dependencies

The project uses Python 3.11+. Install dependencies using one of these methods:

#### Using pip:
```bash
cd /Users/rhuria/Downloads/LineGuardAI
pip install streamlit folium streamlit-folium geopandas plotly numpy pandas scikit-learn scipy shapely pyproj requests earthengine-api geemap rasterio fiona psycopg2-binary
```

#### Using uv (faster):
```bash
cd /Users/rhuria/Downloads/LineGuardAI
pip install uv
uv pip install -e .
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

**Note**: The database features are optional. The app will run with limited functionality if DATABASE_URL is not set.

---

## Full Setup (With All Features)

### 1. Install Dependencies (same as above)

### 2. Set Up PostgreSQL Database (Optional)

If you want historical tracking and monitoring features:

#### Install PostgreSQL:
- **Mac**: `brew install postgresql`
- **Ubuntu**: `sudo apt-get install postgresql`
- **Windows**: Download from [postgresql.org](https://www.postgresql.org/download/)

#### Create Database:
```bash
# Start PostgreSQL
brew services start postgresql  # Mac
sudo service postgresql start   # Linux

# Create database
createdb lineguardai

# Get connection URL (default):
# postgresql://localhost/lineguardai
```

### 3. Set Environment Variables

Create a `.env` file or export variables:

```bash
# Required for database features
export DATABASE_URL="postgresql://localhost/lineguardai"

# Optional: For Google Earth Engine (advanced satellite data access)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Optional: For weather data
export OPENWEATHER_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```bash
echo 'DATABASE_URL=postgresql://localhost/lineguardai' > .env
```

### 4. Run with Environment Variables

```bash
# If using .env file
export $(cat .env | xargs)
streamlit run app.py

# Or directly
DATABASE_URL=postgresql://localhost/lineguardai streamlit run app.py
```

---

## First Time Usage

1. **Launch the app** - `streamlit run app.py`
2. **Select a location** - Use the quick presets (Bay Area, Napa Valley, etc.) or enter custom coordinates
3. **Configure data sources** - Check the boxes for satellite data, LiDAR, and power lines
4. **Click "Load Data"** - This may take 1-2 minutes for the first load
5. **Run Risk Analysis** - Navigate to the "Risk Dashboard" tab and click "Run Risk Analysis"
6. **Explore results** - Use the different tabs to view maps, vegetation analysis, and monitoring

---

## Troubleshooting

### App won't start
- Make sure you're in the correct directory: `cd /Users/rhuria/Downloads/LineGuardAI`
- Check Python version: `python --version` (should be 3.11+)
- Try reinstalling Streamlit: `pip install --upgrade streamlit`

### "DATABASE_URL not set" warning
- This is normal if you haven't set up PostgreSQL
- The app will still work but without historical tracking features
- To enable database: Follow "Set Up PostgreSQL Database" steps above

### Data loading fails
- Initial data loads can take several minutes
- Some data sources may be unavailable depending on location
- The app uses synthetic demo data as fallback

### Import errors
- Run: `pip install -r <(grep 'dependencies' pyproject.toml -A 20 | grep '    "' | sed 's/.*"\(.*\)>=.*/\1/')`
- Or install each package individually from the error message

---

## Performance Tips

- **First run is slowest**: ML models need to initialize (30-60 seconds)
- **Smaller analysis radius**: Start with 5-10 km radius for faster results
- **Browser compatibility**: Works best in Chrome, Firefox, or Safari
- **Mobile access**: The UI is mobile-responsive, access from phone/tablet browser at your local IP

---

## Advanced Configuration

### Google Earth Engine Setup (Optional)
For real satellite data access:
1. Sign up at [https://earthengine.google.com/](https://earthengine.google.com/)
2. Install: `pip install earthengine-api`
3. Authenticate: `earthengine authenticate`

### Weather API Setup (Optional)
For real-time weather data:
1. Get free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Set environment variable: `export OPENWEATHER_API_KEY="your_key"`

---

## Quick Test

Run this to verify everything works:

```bash
cd /Users/rhuria/Downloads/LineGuardAI
python -c "import streamlit; import folium; import geopandas; print('✅ All dependencies installed!')"
streamlit run app.py
```

You should see:
- ✅ All dependencies installed!
- Browser opens with the app running
- Welcome screen with configuration options

