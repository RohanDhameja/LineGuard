from flask import Flask, render_template, jsonify, request
import requests
import random
from datetime import date, timedelta
import os

app = Flask(__name__)

# === Load ML Risk Model ===
risk_model = None
try:
    from utils.risk_model import PowerLineRiskModel
    model_path = 'models/fire_app_risk_model'
    if os.path.exists('models'):
        risk_model = PowerLineRiskModel(model_type='both')
        if risk_model.load_models(filepath_prefix=model_path):
            print("✅ ML Risk Model loaded successfully")
        else:
            print("⚠️  ML model files not found - train the model first")
            risk_model = None
    else:
        print("⚠️  Models directory not found - train the model first using: python train_risk_model.py")
except Exception as e:
    print(f"⚠️  ML Risk Model not available: {e}")
    risk_model = None

# === Fetch transmission line geometries from ArcGIS API ===
API_URL = "https://services3.arcgis.com/bWPjFyq029ChCGur/arcgis/rest/services/Transmission_Line/FeatureServer/2/query?outFields=*&where=1%3D1&f=geojson"

def fetch_lines():
    try:
        res = requests.get(API_URL)
        data = res.json()
        lines = []
        for i, feature in enumerate(data["features"]):
            coords = feature["geometry"]["coordinates"]
            
            # Flatten if nested
            while isinstance(coords[0], list) and isinstance(coords[0][0], list):
                coords = coords[0]
            
            line_points = []
            for lon, lat in coords:
                line_points.append({"lat": float(lat), "lon": float(lon)})

            # Extract attributes (use keys present in API)
            attrs = feature.get("properties", {})
            lines.append({
                "id": i,
                "points": line_points,
                "kv": attrs.get("kV", "N/A"),        # Example key from API
                "status": attrs.get("Status", "N/A"),
                "length_mile": attrs.get("Length_Mile", "N/A")
            })
        return lines
    except Exception as e:
        print(f"Error fetching transmission lines: {e}")
        return []


# === Transmission lines ===
LINES = fetch_lines()

LINE_HEIGHT_M = 8.0
THRESHOLD_DISTANCE = 6.0  # meters

# === Define 4 vegetation zones manually ===
ZONES = [
    {"id": 0, "min_lat": 35.7674275193022, "max_lat": 35.7674275193022, "min_lon": -120.442012631, "max_lon": -120.442012631},
    {"id": 1, "min_lat": 35.7868696904087, "max_lat": 35.7868696904087, "min_lon": -120.418170521401, "max_lon": -120.418170521401},
    {"id": 2, "min_lat": 35.8489547981213, "max_lat": 35.8489547981213, "min_lon": -120.348411490484, "max_lon": -120.348411490484},
    {"id": 3, "min_lat": 34.2239140945453, "max_lat": 34.2239140945453, "min_lon": -116.90996414966, "max_lon": -116.90996414966},
    {"id": 4, "min_lat": 34.2106297635416, "max_lat": 34.2106297635416, "min_lon": -116.906609995797, "max_lon": -116.906609995797},
    {"id": 5, "min_lat": 34.1976199560805, "max_lat": 34.1976199560805, "min_lon": -116.909203216427, "max_lon": -116.909203216427},
    {"id": 6, "min_lat": 36.6457420953003, "max_lat": 36.6457420953003, "min_lon": -120.972005184682, "max_lon": -120.972005184682},
    {"id": 7, "min_lat": 36.6432756517884, "max_lat": 36.6432756517884, "min_lon": -120.947377254923, "max_lon": -120.947377254923},
]

# === Artificial alert test points ===
# === Artificial alerts for today and tomorrow ===




# === Simulation parameters ===
# TEMPORARY: Weekly intervals instead of daily
WEEKS = 8  # 8 weeks = ~2 months
START_DATE = date.today()  # Current date as start
DATE_LIST = [(START_DATE + timedelta(weeks=i)).isoformat() for i in range(WEEKS)]
DAYS = WEEKS  # Keep DAYS variable for compatibility

# === Load synthetic datasets or fall back to hardcoded ===
def load_synthetic_data():
    """Load synthetic LIDAR and growth simulation data"""
    try:
        import json
        with open('data/growth_simulation_data.json', 'r') as f:
            growth_data = json.load(f)
        with open('data/lidar_canopy_data.json', 'r') as f:
            lidar_data = json.load(f)
        print("✅ Loaded synthetic LIDAR and growth simulation data")
        return growth_data, lidar_data
    except FileNotFoundError:
        print("⚠️  Synthetic data files not found. Run: python3 generate_datasets.py")
        print("⚠️  Falling back to hardcoded simulation...")
        return None, None

growth_simulation_data, lidar_data = load_synthetic_data()

# === Vegetation growth simulation ===
veg_time_series = {}

if growth_simulation_data:
    # Use realistic synthetic data
    for zone_id_str, growth_data in growth_simulation_data.items():
        zone_id = int(zone_id_str)
        heights = []
        for week_data in growth_data['weekly_heights']:
            heights.append(round(week_data['cumulative_height'], 3))
        veg_time_series[zone_id] = heights
else:
    # Fall back to hardcoded simulation
    random.seed(42)
    for zone in ZONES:
        zone_id = zone["id"]
        
        # Special variation for zones 6 & 7
        if zone_id == 6:
            # Zone 6: Starts higher and grows faster (high risk zone)
            initial = random.uniform(0.5, 0.7)
            weekly_rate = random.uniform(0.2, 0.35)
        elif zone_id == 7:
            # Zone 7: Variable growth with fluctuations (unpredictable zone)
            initial = random.uniform(0.3, 0.5)
            weekly_rate = random.uniform(0.15, 0.3)
        else:
            # Normal zones
            initial = random.uniform(0.2, 0.6)
            weekly_rate = random.uniform(0.1, 0.25)
        
        heights = []
        for w in range(WEEKS):
            # Add extra variation for zones 6 & 7
            if zone_id == 7:
                # Zone 7 has more random fluctuations
                variation = random.uniform(-0.08, 0.08)
            else:
                variation = random.uniform(-0.02, 0.02)
            
            h = initial + weekly_rate * w + variation
            h = max(0.0, round(h, 3))
            heights.append(h)
        veg_time_series[zone_id] = heights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata')
def metadata():
    return jsonify({
        "lines": LINES,
        "zones": ZONES,
        "line_height_m": LINE_HEIGHT_M,
        "threshold_distance_m": THRESHOLD_DISTANCE,
        "dates": DATE_LIST,
        "using_synthetic_data": growth_simulation_data is not None
    })

@app.route('/api/lidar_data')
def get_lidar_data():
    """Get LIDAR canopy height data for all zones"""
    if lidar_data:
        return jsonify({
            "status": "success",
            "data": lidar_data,
            "message": "LIDAR canopy height data from synthetic dataset"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "LIDAR data not available. Generate datasets: python3 generate_datasets.py"
        }), 404

@app.route('/api/vegetation_spread')
def get_vegetation_spread():
    """Get vegetation spread data over time"""
    try:
        import json
        with open('data/vegetation_spread_data.json', 'r') as f:
            spread_data = json.load(f)
        return jsonify({
            "status": "success",
            "data": spread_data,
            "message": "Vegetation spread data from synthetic dataset"
        })
    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": "Vegetation spread data not available. Generate datasets: python3 generate_datasets.py"
        }), 404

@app.route('/api/growth_simulation')
def get_growth_simulation():
    """Get complete growth simulation data"""
    if growth_simulation_data:
        return jsonify({
            "status": "success",
            "data": growth_simulation_data,
            "message": "Growth simulation data from synthetic dataset"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Growth simulation data not available. Generate datasets: python3 generate_datasets.py"
        }), 404

@app.route('/api/state')
def state():
    req_date = request.args.get("date")
    if req_date not in DATE_LIST:
        req_date = DATE_LIST[0]
    day_i = DATE_LIST.index(req_date)

    zones_out = []
    alerts = []

    # === Artificial alerts for all weeks (persistent zones) ===
    ARTIFICIAL_ALERTS = [
    {
        "date": DATE_LIST[0],  # week 0
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 5.2, "clearance_m": 2.8},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 4.9, "clearance_m": 3.1}
        ]
    },
    {
        "date": DATE_LIST[1],  # week 1
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 5.4, "clearance_m": 2.6},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 5.1, "clearance_m": 2.9}
        ]
    },
    {
        "date": DATE_LIST[2],  # week 2
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 5.6, "clearance_m": 2.4},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 5.3, "clearance_m": 2.7}
        ]
    },
    {
        "date": DATE_LIST[3],  # week 3
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 5.8, "clearance_m": 2.2},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 5.5, "clearance_m": 2.5}
        ]
    },
    {
        "date": DATE_LIST[4],  # week 4
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 6.0, "clearance_m": 2.0},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 5.7, "clearance_m": 2.3}
        ]
    },
    {
        "date": DATE_LIST[5],  # week 5
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 6.2, "clearance_m": 1.8},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 5.9, "clearance_m": 2.1}
        ]
    },
    {
        "date": DATE_LIST[6],  # week 6
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 6.4, "clearance_m": 1.6},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 6.1, "clearance_m": 1.9}
        ]
    },
    {
        "date": DATE_LIST[7],  # week 7
        "zones": [
            {"lat": 39.8189566248284, "lon": -121.451189414863, "veg_height_m": 6.6, "clearance_m": 1.4},
            {"lat": 39.7648885984727, "lon": -121.486122509439, "veg_height_m": 6.3, "clearance_m": 1.7}
        ]
    }]

    for zone in ZONES:
        zid = zone["id"]
        veg_h = veg_time_series[zid][day_i]
        clearance = LINE_HEIGHT_M - veg_h
        alert = clearance <= THRESHOLD_DISTANCE

        zones_out.append({
            "id": zid,
            "veg_height_m": veg_h,
            "clearance_m": round(clearance, 3),
            "alert": alert,
            "bbox": zone,
        })

        if alert:
            center_lat = (zone["min_lat"] + zone["max_lat"]) / 2
            center_lon = (zone["min_lon"] + zone["max_lon"]) / 2
            alerts.append({
                "zone_id": zid,
                "lat": center_lat,
                "lon": center_lon,
                "veg_height_m": veg_h,
                "clearance_m": round(clearance, 3),
            })
        # === Inject artificial alerts if applicable ===
    for artificial in ARTIFICIAL_ALERTS:
        if artificial["date"] == req_date:
            for a in artificial["zones"]:
                alerts.append({
                    "zone_id": f"Artificial-{a['lat']:.4f}",
                    "lat": a["lat"],
                    "lon": a["lon"],
                    "veg_height_m": a["veg_height_m"],
                    "clearance_m": a["clearance_m"]
                })
                zones_out.append({
                    "id": f"Artificial-{a['lat']:.4f}",
                    "veg_height_m": a["veg_height_m"],
                    "clearance_m": a["clearance_m"],
                    "alert": True,
                    "bbox": {
                        "min_lat": a["lat"] - 0.0001,
                        "max_lat": a["lat"] + 0.0001,
                        "min_lon": a["lon"] - 0.0001,
                        "max_lon": a["lon"] + 0.0001
                    }
                })


    return jsonify({"date": req_date, "zones": zones_out, "alerts": alerts})

@app.route('/api/predictions')
def predictions():
    """Predict which zones will become critical in future weeks"""
    predictions_list = []
    
    for zone in ZONES:
        zid = zone["id"]
        zone_predictions = []
        
        # Check each future week
        for week_idx in range(len(DATE_LIST)):
            veg_h = veg_time_series[zid][week_idx]
            clearance = LINE_HEIGHT_M - veg_h
            
            # If this week is safe but will be critical, predict it
            if clearance > THRESHOLD_DISTANCE:
                # Check if it will become critical in next few weeks
                for future_week in range(week_idx + 1, min(week_idx + 4, len(DATE_LIST))):
                    future_veg = veg_time_series[zid][future_week]
                    future_clearance = LINE_HEIGHT_M - future_veg
                    
                    if future_clearance <= THRESHOLD_DISTANCE:
                        weeks_until_critical = future_week - week_idx
                        center_lat = (zone["min_lat"] + zone["max_lat"]) / 2
                        center_lon = (zone["min_lon"] + zone["max_lon"]) / 2
                        
                        zone_predictions.append({
                            "zone_id": zid,
                            "current_week": week_idx,
                            "critical_week": future_week,
                            "weeks_until_critical": weeks_until_critical,
                            "current_clearance": round(clearance, 2),
                            "future_clearance": round(future_clearance, 2),
                            "lat": center_lat,
                            "lon": center_lon,
                            "date": DATE_LIST[future_week]
                        })
                        break  # Only predict the first critical week
                break  # Move to next zone after finding first safe week
        
        if zone_predictions:
            predictions_list.extend(zone_predictions)
    
    return jsonify({"predictions": predictions_list})

@app.route('/api/notify', methods=['POST'])
def notify_authority():
    """Simulate notifying authorities about alerts"""
    data = request.get_json()
    alert_count = data.get('alert_count', 0)
    zones = data.get('zones', [])
    
    # In production, this would send emails/SMS/API calls to authorities
    notification = {
        "status": "success",
        "message": f"Alert notification sent to authorities for {alert_count} zone(s)",
        "timestamp": date.today().isoformat(),
        "zones": zones,
        "authority": "California Fire Department",
        "priority": "HIGH" if alert_count > 2 else "MEDIUM"
    }
    
    print(f"🚨 AUTHORITY NOTIFICATION: {notification}")
    
    return jsonify(notification)

@app.route('/api/risk_prediction', methods=['POST'])
def risk_prediction():
    """ML-based risk prediction for a specific zone"""
    if not risk_model:
        return jsonify({
            "status": "error",
            "message": "ML model not loaded. Train the model first using: python train_risk_model.py"
        }), 503
    
    data = request.get_json()
    
    # Extract zone data
    vegetation_height = data.get('vegetation_height', 0.5)
    clearance = data.get('clearance', 8.0)
    ndvi = data.get('ndvi', 0.3)
    temperature = data.get('temperature', 22.0)
    humidity = data.get('humidity', 50.0)
    wind_speed = data.get('wind_speed', 3.0)
    line_voltage = data.get('line_voltage', 115.0)
    line_age = data.get('line_age', 20)
    month = data.get('month', date.today().month)
    
    # Prepare data for prediction
    prediction_data = {
        'vegetation_height': vegetation_height,
        'clearance': clearance,
        'ndvi': ndvi,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'line_voltage': line_voltage,
        'line_age': line_age,
        'month': month
    }
    
    try:
        # Make prediction
        prediction = risk_model.predict(prediction_data)
        
        # Format response
        response = {
            "status": "success",
            "zone_data": prediction_data,
            "prediction": prediction
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/api/batch_risk_prediction')
def batch_risk_prediction():
    """Get ML risk predictions for all zones at a specific date"""
    if not risk_model:
        return jsonify({
            "status": "error",
            "message": "ML model not available"
        }), 503
    
    req_date = request.args.get("date")
    if req_date not in DATE_LIST:
        req_date = DATE_LIST[0]
    day_i = DATE_LIST.index(req_date)
    
    predictions = []
    
    # Get predictions for each zone
    for zone in ZONES:
        zid = zone["id"]
        veg_h = veg_time_series[zid][day_i]
        clearance = LINE_HEIGHT_M - veg_h
        
        # Prepare data for ML prediction
        prediction_data = {
            'vegetation_height': veg_h,
            'clearance': clearance,
            'ndvi': 0.3 + (veg_h / 5.0) * 0.4,  # Estimate NDVI from vegetation height
            'temperature': 25.0,  # Default values
            'humidity': 45.0,
            'wind_speed': 5.0,
            'line_voltage': 115.0,
            'line_age': 25,
            'month': date.today().month
        }
        
        try:
            prediction = risk_model.predict(prediction_data)
            
            center_lat = (zone["min_lat"] + zone["max_lat"]) / 2
            center_lon = (zone["min_lon"] + zone["max_lon"]) / 2
            
            predictions.append({
                "zone_id": zid,
                "lat": center_lat,
                "lon": center_lon,
                "vegetation_height": veg_h,
                "clearance": round(clearance, 3),
                "ml_risk_level": prediction.get('classification', {}).get('risk_level', 'Unknown'),
                "ml_risk_score": prediction.get('regression', {}).get('risk_score', 0),
                "ml_confidence": prediction.get('classification', {}).get('confidence', 0),
                "probabilities": prediction.get('classification', {}).get('probabilities', {})
            })
        except Exception as e:
            print(f"Error predicting for zone {zid}: {e}")
    
    return jsonify({
        "status": "success",
        "date": req_date,
        "predictions": predictions
    })

# === Email Notification Endpoints ===

# Initialize email notifier (will check environment variables)
email_notifier = None
try:
    from utils.email_notifier import EmailNotifier
    email_notifier = EmailNotifier()
    if email_notifier.is_configured:
        print("✅ Email notification system configured")
    else:
        print("⚠️  Email not configured. Set SMTP environment variables to enable.")
except Exception as e:
    print(f"⚠️  Email notifier not available: {e}")

@app.route('/api/send_alert_email', methods=['POST'])
def send_alert_email():
    """Send email notification for high-risk zones"""
    if not email_notifier or not email_notifier.is_configured:
        return jsonify({
            "status": "error",
            "message": "Email notification system not configured. Please set SMTP environment variables."
        }), 503
    
    data = request.get_json()
    recipient_email = data.get('recipient_email')
    alert_zones = data.get('alert_zones', [])
    date_str = data.get('date')
    
    if not recipient_email:
        return jsonify({
            "status": "error",
            "message": "Recipient email is required"
        }), 400
    
    if not alert_zones:
        return jsonify({
            "status": "error",
            "message": "No alert zones provided"
        }), 400
    
    try:
        result = email_notifier.send_alert_email(
            recipient_email,
            alert_zones,
            date_str
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to send email: {str(e)}"
        }), 500

@app.route('/api/send_weekly_summary', methods=['POST'])
def send_weekly_summary():
    """Send weekly summary email"""
    if not email_notifier or not email_notifier.is_configured:
        return jsonify({
            "status": "error",
            "message": "Email notification system not configured"
        }), 503
    
    data = request.get_json()
    recipient_email = data.get('recipient_email')
    summary_data = data.get('summary_data', {})
    
    if not recipient_email:
        return jsonify({
            "status": "error",
            "message": "Recipient email is required"
        }), 400
    
    try:
        result = email_notifier.send_weekly_summary(
            recipient_email,
            summary_data
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to send summary: {str(e)}"
        }), 500

@app.route('/api/email_status')
def email_status():
    """Check if email notification system is configured"""
    if not email_notifier:
        return jsonify({
            "configured": False,
            "message": "Email notifier module not loaded"
        })
    
    return jsonify({
        "configured": email_notifier.is_configured,
        "smtp_server": email_notifier.smtp_server if email_notifier.is_configured else None,
        "sender_email": email_notifier.sender_email if email_notifier.is_configured else None,
        "message": "Email system ready" if email_notifier.is_configured else "Configure SMTP environment variables"
    })

if __name__ == "__main__":
    app.run(debug=True)
