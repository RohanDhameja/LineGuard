let map = L.map('map').setView([37.5, -121.8], 6);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);

let zoneLayers = {}, alertMarkers = [], lineLayers = [];
let isPlaying = false, currentDate = null;
let intervalId = null;
let dateList = [];

function colorForHeight(h, clearance) {
    if (clearance <= 6.0) return '#dc143c'; // alert - dark red
    if (h < 0.7) return '#00b327'; // low - dark green
    if (h < 1.5) return '#ff9500'; // moderate - dark orange
    return '#ff9500';
}

async function fetchMetadata() {
    const res = await fetch('/api/metadata');
    const data = await res.json();
    dateList = data.dates;
    return data;
}

function drawLines(lines) {
    lines.forEach(l => {
        const latlngs = l.points.map(p => [p.lat, p.lon]);
        const poly = L.polyline(latlngs, { color: 'black', weight: 3 }).addTo(map);

        // Bind a tooltip showing KV, length, and status
        const tooltipContent = `
            <strong>Transmission Line ${l.id}</strong><br>
            kV: ${l.kv}<br>
            Length: ${l.length_mile} miles<br>
            Status: ${l.status}
        `;
        poly.bindTooltip(tooltipContent, { sticky: true });

        lineLayers.push(poly);
        
        // Add yellow dot for line 1098
        if (l.id === 1098 && l.points.length > 0) {
            // Get center point of the line
            const midIndex = Math.floor(l.points.length / 2);
            const centerPoint = l.points[midIndex];
            
            const yellowCircle = L.circleMarker([centerPoint.lat, centerPoint.lon], {
                color: '#ff9500',
                fillColor: '#ff9500',
                fillOpacity: 0.85,
                radius: 8,
                weight: 2
            }).addTo(map);
            
            const line1098Popup = `
                <strong>Transmission Line ${l.id}</strong><br>
                Status: 🟡 Monitored<br>
                ⚡ Voltage: ${l.kv} kV<br>
                📏 Length: ${l.length_mile} miles<br>
                📍 ${centerPoint.lat.toFixed(4)}, ${centerPoint.lon.toFixed(4)}
            `;
            yellowCircle.bindPopup(line1098Popup);
            lineLayers.push(yellowCircle);
        }
    });
}


async function updateMapByDate(dateStr) {
    const res = await fetch(`/api/state?date=${dateStr}`);
    const data = await res.json();
    
    // Display week number and date
    const index = dateList.indexOf(dateStr);
    document.getElementById('dayLabel').innerText = `${index} (${dateStr})`;

    // Update slider
    document.getElementById('dayRange').value = index;

    // Clear previous layers
    Object.values(zoneLayers).forEach(layer => map.removeLayer(layer));
    zoneLayers = {};
    alertMarkers.forEach(m => map.removeLayer(m));
    alertMarkers = [];

    const alertItems = document.getElementById('alertItems');
    alertItems.innerHTML = '';

    // Metrics calculation
    let totalVegetation = 0;
    let zoneCount = 0;

    data.zones.forEach(z => {
        // Calculate center of zone
        const centerLat = (z.bbox.min_lat + z.bbox.max_lat) / 2;
        const centerLon = (z.bbox.min_lon + z.bbox.max_lon) / 2;

        const color = colorForHeight(z.veg_height_m, z.clearance_m);
        const opacity = Math.min(0.85, 0.5 + z.veg_height_m / 5);

        // Determine risk level
        let riskLevel = '';
        if (z.clearance_m <= 6.0) {
            riskLevel = '🔴 High Risk';
        } else if (z.veg_height_m >= 1.5) {
            riskLevel = '🟡 Medium Risk';
        } else {
            riskLevel = '🟢 Low Risk';
        }

        // Define radius (in meters)
        const radius = 1000; // 15 km for example, adjust as needed

        const circle = L.circle([centerLat, centerLon], {
            color: color,
            fillColor: color,
            fillOpacity: opacity,
            radius: radius,
            weight: 2
        }).addTo(map);

        const tooltipContent = `
            <strong>Zone ${z.id}</strong><br>
            Risk Level: ${riskLevel}<br>
            🌱 Vegetation: ${z.veg_height_m}m<br>
            📏 Clearance: ${z.clearance_m}m<br>
            📍 ${centerLat.toFixed(4)}, ${centerLon.toFixed(4)}
        `;

        circle.bindTooltip(tooltipContent, { sticky: true });

        zoneLayers[z.id] = circle;

        // Accumulate for metrics
        totalVegetation += z.veg_height_m;
        zoneCount++;
    });

    if (data.alerts.length > 0) {
        data.alerts.forEach(a => {
            const marker = L.marker([a.lat, a.lon]).addTo(map);
            const alertPopup = `
                <strong>Zone ${a.zone_id}</strong><br>
                Risk Level: 🔴 High Risk<br>
                🌱 Vegetation: ${a.veg_height_m}m<br>
                📏 Clearance: ${a.clearance_m}m<br>
                📍 ${a.lat.toFixed(4)}, ${a.lon.toFixed(4)}
            `;
            marker.bindPopup(alertPopup);
            alertMarkers.push(marker);

            const item = document.createElement('div');
            item.className = 'alert-item';
            item.innerHTML = `
                <strong>Zone ${a.zone_id}</strong><br>
                Risk: 🔴 High Risk<br>
                🌱 Vegetation: ${a.veg_height_m}m<br>
                📏 Clearance: ${a.clearance_m}m<br>
                📍 ${a.lat.toFixed(4)}, ${a.lon.toFixed(4)}
            `;
            alertItems.appendChild(item);

            if ("Notification" in window && Notification.permission === "granted") {
                new Notification(`🔴 ALERT! Zone ${a.zone_id}`, {
                    body: `Risk: High Risk\nVegetation: ${a.veg_height_m}m\nClearance: ${a.clearance_m}m`
                });
            }
        });
    } else {
        alertItems.innerHTML = '<div>No active alerts</div>';
    }

    // Update metrics panel
    const avgVeg = zoneCount > 0 ? (totalVegetation / zoneCount).toFixed(2) : 0;
    document.getElementById('activeAlerts').innerText = data.alerts.length;
    document.getElementById('monitoredZones').innerText = zoneCount;
    document.getElementById('avgVegetation').innerText = `${avgVeg}m`;
    
    // Update ML risk predictions
    await updateMLPredictions(dateStr);
}

async function updateMLPredictions(dateStr) {
    try {
        const res = await fetch(`/api/batch_risk_prediction?date=${dateStr}`);
        const data = await res.json();
        
        if (data.status === 'success' && data.predictions.length > 0) {
            // Calculate average ML risk metrics across all zones
            let totalRiskScore = 0;
            let totalConfidence = 0;
            let riskLevelCounts = { 'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0 };
            
            data.predictions.forEach(p => {
                totalRiskScore += p.ml_risk_score;
                totalConfidence += p.ml_confidence;
                const level = p.ml_risk_level;
                if (riskLevelCounts.hasOwnProperty(level)) {
                    riskLevelCounts[level]++;
                }
            });
            
            const avgRiskScore = (totalRiskScore / data.predictions.length).toFixed(3);
            const avgConfidence = (totalConfidence / data.predictions.length * 100).toFixed(1);
            
            // Determine dominant risk level
            let dominantLevel = 'Low';
            let maxCount = 0;
            for (const [level, count] of Object.entries(riskLevelCounts)) {
                if (count > maxCount) {
                    maxCount = count;
                    dominantLevel = level;
                }
            }
            
            // Update UI with risk level emoji
            let riskEmoji = '🟢';
            if (dominantLevel === 'Moderate') riskEmoji = '🟡';
            else if (dominantLevel === 'High') riskEmoji = '🟠';
            else if (dominantLevel === 'Critical') riskEmoji = '🔴';
            
            document.getElementById('mlRiskLevel').innerText = `${riskEmoji} ${dominantLevel}`;
            document.getElementById('mlRiskScore').innerText = `${(avgRiskScore * 100).toFixed(1)}%`;
            document.getElementById('mlConfidence').innerText = `${avgConfidence}%`;
            
            // Add ML risk info to zone tooltips
            data.predictions.forEach(p => {
                if (zoneLayers[p.zone_id]) {
                    const circle = zoneLayers[p.zone_id];
                    const existingTooltip = circle.getTooltip();
                    if (existingTooltip) {
                        let content = existingTooltip.getContent();
                        content += `<br><strong>ML Risk:</strong> ${p.ml_risk_level} (${(p.ml_risk_score * 100).toFixed(1)}%)`;
                        circle.setTooltipContent(content);
                    }
                }
            });
        } else {
            // ML model not available
            document.getElementById('mlRiskLevel').innerText = 'N/A';
            document.getElementById('mlRiskScore').innerText = 'N/A';
            document.getElementById('mlConfidence').innerText = 'N/A';
        }
    } catch (error) {
        console.log('ML predictions not available:', error);
        document.getElementById('mlRiskLevel').innerText = 'N/A';
        document.getElementById('mlRiskScore').innerText = 'N/A';
        document.getElementById('mlConfidence').innerText = 'N/A';
    }
}

function playSimulation() {
    if (isPlaying) return;
    isPlaying = true;
    intervalId = setInterval(() => {
        let index = dateList.indexOf(currentDate);
        if (index >= dateList.length - 1) { pauseSimulation(); return; }
        currentDate = dateList[index + 1];
        updateMapByDate(currentDate);
    }, 1500); // Slowed down for weekly intervals
}

function pauseSimulation() {
    isPlaying = false;
    if (intervalId) clearInterval(intervalId);
}

(async function init() {
    if ("Notification" in window) Notification.requestPermission();
    const metadata = await fetchMetadata();
    drawLines(metadata.lines);
    currentDate = dateList[0];
    await updateMapByDate(currentDate);

    // Slider
    const slider = document.getElementById('dayRange');
    slider.max = dateList.length - 1;
    slider.addEventListener('input', e => {
        currentDate = dateList[parseInt(e.target.value)];
        updateMapByDate(currentDate);
    });

    // Buttons
    document.getElementById('playBtn').addEventListener('click', () => playSimulation());
    document.getElementById('pauseBtn').addEventListener('click', pauseSimulation);
    document.getElementById('resetBtn').addEventListener('click', () => {
        currentDate = dateList[0];
        updateMapByDate(currentDate);
        pauseSimulation();
    });

    // Jump to date input (YYYY-MM-DD)
    document.getElementById('jumpDay').addEventListener('change', e => {
        const val = e.target.value;
        if (dateList.includes(val)) {
            currentDate = val;
            updateMapByDate(currentDate);
        }
    });
})();

// California approximate bounds
const CA_BOUNDS = {
    minLat: 32.5,
    maxLat: 42.0,
    minLon: -124.5,
    maxLon: -114.0
};

// City coordinates (lat, lon)
const cityCenters = {
    adelanto: [34.58277, -117.40922],
agouraHills: [34.13639, -118.77453],
alameda: [37.76521, -122.24164],
alhambra: [34.09528, -118.12701],
alisoViejo: [33.56768, -117.72500],
alpine: [32.83532, -116.76675],
altadena: [34.19653, -118.13120],
alturas: [41.48753, -120.54595],
amador: [38.44520, -120.78360],
americanCanyon: [38.18560, -122.26955],
anaheim: [33.8366, -117.9143],
anaheimHills: [33.84119, -117.75865],
anderson: [40.10578, -122.23100],
angelsCamp: [38.06777, -120.53862],
angelusOaks: [34.26750, -116.71500],
antioch: [38.00492, -121.80579],
appleValley: [34.50083, -117.18588],
aptos: [36.97757, -121.90268],
arcadia: [34.13973, -118.03534],
arcata: [40.86697, -124.08251],
arnold: [38.42513, -120.26393],
arroyoGrande: [35.11847, -120.59475],
artesia: [33.84457, -118.08146],
atascadero: [35.48947, -120.67029],
auburn: [38.89660, -121.07688],
avalon: [33.34281, -118.32779],
avilaBeach: [35.14100, -120.62667],
azusa: [34.13066, -117.90687],
bakersfield: [35.3733, -119.0187],
boulderCreek: [37.1269, -122.1211],
bigbar: [39.8189566248284, -121.451189414863],
chulaVista: [32.6401, -117.0842],
fresno: [36.7378, -119.7871],
irvine: [33.6846, -117.8265],
longBeach: [33.7701, -118.1937],
losAngeles: [34.0522, -118.2437],
oakland: [37.8044, -122.2712],
panocheValley: [36.6432756517884, -120.947377254923],
riverside: [33.9806, -117.3755],
sacramento: [38.5816, -121.4944],
sanDiego: [32.7157, -117.1611],
sanFrancisco: [37.7749, -122.4194],
sanJose: [37.3382, -121.8863],
santaAna: [33.7455, -117.8677],
stockton: [37.9577, -121.2908]
};

document.getElementById('goBtn').addEventListener('click', () => {
    let lat = parseFloat(document.getElementById('latInput').value);
    let lon = parseFloat(document.getElementById('lonInput').value);
    let cityVal = document.getElementById('citySelect').value;

    let targetLat, targetLon;

    if (cityVal !== "") {
        // Focus on selected city
        const coords = cityCenters[cityVal];
        targetLat = coords[0];
        targetLon = coords[1];
    } else if (!isNaN(lat) && !isNaN(lon)) {
        // Focus on input coordinates
        if (lat < CA_BOUNDS.minLat || lat > CA_BOUNDS.maxLat || lon < CA_BOUNDS.minLon || lon > CA_BOUNDS.maxLon) {
            alert("Data for electric transmission lines is only available in California!");
            return;
        }
        targetLat = lat;
        targetLon = lon;
    } else {
        // Default view (first city)
        const coords = cityCenters["sacramento"];
        targetLat = coords[0];
        targetLon = coords[1];
    }

    // Smooth zoom/fly to the location
    map.flyTo([targetLat, targetLon], 10, { animate: true, duration: 2.5 });
});


const toggleBtn = document.getElementById('togglePanelsBtn');
const mergedPanel = document.getElementById('mergedPanel');
let panelVisible = true;

toggleBtn.addEventListener('click', () => {
    panelVisible = !panelVisible;
    if (panelVisible) {
        mergedPanel.style.left = "10px";       // show merged panel
    } else {
        mergedPanel.style.left = "-400px";     // hide merged panel (slide left)
    }
});

// Notify authorities button
document.getElementById('notifyAuthorityBtn').addEventListener('click', async () => {
    const alertCount = parseInt(document.getElementById('activeAlerts').innerText);
    
    if (alertCount === 0) {
        alert('No active alerts to report at this time.');
        return;
    }
    
    // Get current alert zones
    const alertZones = [];
    alertMarkers.forEach((marker, idx) => {
        const latlng = marker.getLatLng();
        alertZones.push({
            lat: latlng.lat,
            lon: latlng.lng,
            index: idx
        });
    });
    
    try {
        const response = await fetch('/api/notify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                alert_count: alertCount,
                zones: alertZones,
                date: currentDate
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert(`✅ ${data.message}\n\nAuthority: ${data.authority}\nPriority: ${data.priority}\nTimestamp: ${data.timestamp}`);
            
            // Desktop notification
            if ("Notification" in window && Notification.permission === "granted") {
                new Notification('🚨 Authority Notified', {
                    body: `${alertCount} zone(s) reported to ${data.authority}`,
                    icon: '🔥'
                });
            }
        }
    } catch (error) {
        alert('❌ Failed to notify authorities. Please try again.');
        console.error('Notification error:', error);
    }
});




