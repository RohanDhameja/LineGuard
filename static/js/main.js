// Initialize map centered on Central California - optimized for zone grid
let map = L.map('map').setView([37.25, -120.9], 7);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { 
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

let zoneLayers = {}, alertMarkers = [], lineLayers = [];
let isPlaying = false, currentDate = null;
let intervalId = null;
let dateList = [];
let isUpdatingSlider = false; // Flag to prevent slider feedback loop

function colorForHeight(h, clearance) {
    if (clearance <= 6.0) return 'red'; // High risk / alert
    if (h < 1.0) return 'green'; // Low vegetation
    if (h < 2.5) return 'yellow'; // Moderate vegetation
    return 'red'; // High vegetation
}

function getRiskLevel(h, clearance) {
    if (clearance <= 6.0) return 'high';
    if (h < 1.0) return 'low';
    if (h < 2.5) return 'moderate';
    return 'high';
}

async function updateMLRiskAssessment(zones) {
    // Find the highest risk zone
    let highestRiskZone = null;
    let highestRisk = -1;
    
    zones.forEach(zone => {
        const riskLevel = getRiskLevel(zone.veg_height_m, zone.clearance_m);
        const riskValue = riskLevel === 'high' ? 3 : (riskLevel === 'moderate' ? 2 : 1);
        if (riskValue > highestRisk) {
            highestRisk = riskValue;
            highestRiskZone = zone;
        }
    });
    
    if (!highestRiskZone) {
        // No zones, show default
        document.getElementById('mlRiskLevel').innerText = '-';
        document.getElementById('mlRiskScore').innerText = '-';
        document.getElementById('mlConfidence').innerText = '-';
        document.getElementById('growthRateDisplay').innerText = '-';
        document.getElementById('breachPrediction').innerText = '-';
        return;
    }
    
    // Update Growth Rate and Breach Prediction in Live Metrics
    if (highestRiskZone.growth_rate_cm_day) {
        document.getElementById('growthRateDisplay').innerText = `${highestRiskZone.growth_rate_cm_day} cm/day`;
    }
    
    if (highestRiskZone.days_until_breach !== undefined) {
        if (highestRiskZone.days_until_breach === 0 || highestRiskZone.breach_date === 'BREACHED') {
            document.getElementById('breachPrediction').innerHTML = '<span style="color: #ef4444; font-weight: bold;">🔴 BREACHED</span>';
        } else {
            document.getElementById('breachPrediction').innerText = `${highestRiskZone.days_until_breach} days`;
        }
    }
    
    // Call ML prediction API
    try {
        const params = new URLSearchParams({
            veg_height: highestRiskZone.veg_height_m,
            clearance: highestRiskZone.clearance_m,
            temperature: 25, // Default weather values
            humidity: 50,
            wind_speed: 5,
            days_since_rain: 7,
            latitude: (highestRiskZone.bbox.min_lat + highestRiskZone.bbox.max_lat) / 2,
            longitude: (highestRiskZone.bbox.min_lon + highestRiskZone.bbox.max_lon) / 2
        });
        
        const response = await fetch(`/api/ml_predict?${params}`);
        const data = await response.json();
        
        if (data.success && data.prediction) {
            const pred = data.prediction;
            
            // Update ML Risk Level with emoji and color
            const riskEmoji = {
                'Low': '🟢',
                'Moderate': '🟡',
                'High': '🟠',
                'Critical': '🔴'
            };
            document.getElementById('mlRiskLevel').innerText = `${riskEmoji[pred.risk_level] || ''} ${pred.risk_level}`;
            
            // Update Risk Score
            document.getElementById('mlRiskScore').innerText = `${pred.risk_percentage.toFixed(1)}%`;
            
            // Update Confidence
            document.getElementById('mlConfidence').innerText = `${(pred.confidence * 100).toFixed(0)}%`;
        } else {
            throw new Error('Prediction failed');
        }
    } catch (error) {
        console.error('ML Prediction error:', error);
        document.getElementById('mlRiskLevel').innerText = 'Error';
        document.getElementById('mlRiskScore').innerText = '-';
        document.getElementById('mlConfidence').innerText = '-';
    }
}

function getIconForRisk(riskLevel) {
    const icons = {
        'low': '🟢',
        'moderate': '🟡', 
        'high': '🔴'
    };
    return icons[riskLevel] || '⚪';
}

async function fetchMetadata() {
    const res = await fetch('/api/metadata');
    const data = await res.json();
    dateList = data.dates;
    
    // Set initial date label immediately
    if (dateList.length > 0) {
        const dayLabel = document.getElementById('dayLabel');
        if (dayLabel) {
            dayLabel.innerText = dateList[0];
        }
    }
    
    return data;
}

function drawLines(lines) {
    lines.forEach(l => {
        const latlngs = l.points.map(p => [p.lat, p.lon]);
        const poly = L.polyline(latlngs, { 
            color: '#2d3748', 
            weight: 3,
            opacity: 0.7,
            smoothFactor: 1
        }).addTo(map);

        // Bind a tooltip showing KV, length, and status
        const tooltipContent = `
            <div style="font-family: 'Poppins', sans-serif;">
                <strong style="color: #764ba2;"><i class="fas fa-bolt"></i> Transmission Line ${l.id}</strong><br>
                <strong>kV:</strong> ${l.kv}<br>
                <strong>Length:</strong> ${l.length_mile} miles<br>
                <strong>Status:</strong> ${l.status}
            </div>
        `;
        poly.bindTooltip(tooltipContent, { sticky: true });

        lineLayers.push(poly);
    });
}


// Helper function to calculate distance between two coordinates (Haversine formula)
function getDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c; // Distance in km
}

async function updateMapByDate(dateStr) {
    // Preserve button states during update (don't reset them)
    const wasPlaying = isPlaying;
    
    try {
        console.log('📡 Fetching state for date:', dateStr);
        const res = await fetch(`/api/state?date=${dateStr}`);
        if (!res.ok) {
            throw new Error(`API error: ${res.status}`);
        }
        const data = await res.json();
        console.log('✅ State fetched, zones:', data.zones.length);
        
        // Filter zones by current location if a city/location is selected
        if (currentLocation) {
            console.log(`📍 Filtering zones by location: ${currentLocation.lat}, ${currentLocation.lon}, radius ${currentLocation.radius}km`);
            const filteredZones = [];
            const filteredAlerts = [];
            
            data.zones.forEach(z => {
                const zoneCenterLat = (z.bbox.min_lat + z.bbox.max_lat) / 2;
                const zoneCenterLon = (z.bbox.min_lon + z.bbox.max_lon) / 2;
                const distance = getDistance(currentLocation.lat, currentLocation.lon, zoneCenterLat, zoneCenterLon);
                if (distance <= currentLocation.radius) {
                    filteredZones.push(z);
                    console.log(`  ✅ Zone ${z.id} within range: ${distance.toFixed(2)}km`);
                } else {
                    console.log(`  ❌ Zone ${z.id} out of range: ${distance.toFixed(2)}km > ${currentLocation.radius}km`);
                }
            });
            
            console.log(`📍 Filter result: ${data.zones.length} total zones -> ${filteredZones.length} zones within ${currentLocation.radius}km`);
            
            // Only filter if we have zones to show, otherwise show all zones (fallback)
            if (filteredZones.length > 0) {
                data.zones = filteredZones;
                
                // Also filter alerts
                data.alerts.forEach(a => {
                    const distance = getDistance(currentLocation.lat, currentLocation.lon, a.lat, a.lon);
                    if (distance <= currentLocation.radius) {
                        filteredAlerts.push(a);
                    }
                });
                data.alerts = filteredAlerts;
            } else {
                console.warn(`⚠️ No zones found within ${currentLocation.radius}km. Showing all zones as fallback.`);
                // Keep all zones if none match (fallback behavior)
            }
        }
        
        const dayLabel = document.getElementById('dayLabel');
        if (dayLabel) {
            dayLabel.innerText = dateStr;
        }

    // Update slider (without triggering input event)
    isUpdatingSlider = true;
    const index = dateList.indexOf(dateStr);
    if (index >= 0 && index < dateList.length) {
        document.getElementById('dayRange').value = index;
    }
    // Use setTimeout to reset flag after DOM update
    setTimeout(() => { isUpdatingSlider = false; }, 100);

    // Clear previous layers - ensure all markers are removed
    console.log(`🗑️ Clearing markers: ${Object.keys(zoneLayers).length} zone markers, ${alertMarkers.length} alert markers, ${transmissionTowerMarkers.length} tower markers`);
    
    // Clear transmission tower markers (shouldn't be here in Time Predictor, but clear anyway)
    transmissionTowerMarkers.forEach(marker => {
        try {
            map.removeLayer(marker);
        } catch (e) {
            console.warn('⚠️ Error removing tower marker:', e);
        }
    });
    transmissionTowerMarkers = [];
    
    Object.values(zoneLayers).forEach(layer => {
        try {
            map.removeLayer(layer);
        } catch (e) {
            console.warn('⚠️ Error removing zone layer:', e);
        }
    });
    zoneLayers = {};
    alertMarkers.forEach(m => {
        try {
            map.removeLayer(m);
        } catch (e) {
            console.warn('⚠️ Error removing alert marker:', e);
        }
    });
    alertMarkers = [];

    const alertItems = document.getElementById('alertItems');
    const alertItemsPredictor = document.getElementById('alertItemsPredictor');
    alertItems.innerHTML = '';
    alertItemsPredictor.innerHTML = '';
    
    // Calculate statistics
    let totalHeight = 0;
    let alertCount = 0;
    let riskCounts = { low: 0, moderate: 0, high: 0 };
    const zonesProcessed = new Set(); // Track which zones we've processed to prevent duplicates

    data.zones.forEach(z => {
        // Prevent duplicate processing
        if (zonesProcessed.has(z.id)) {
            console.warn(`⚠️ Duplicate zone ${z.id} detected! Skipping.`);
            return;
        }
        zonesProcessed.add(z.id);
        
        // Calculate center of zone
        const centerLat = (z.bbox.min_lat + z.bbox.max_lat) / 2;
        const centerLon = (z.bbox.min_lon + z.bbox.max_lon) / 2;

        const riskLevel = getRiskLevel(z.veg_height_m, z.clearance_m);
        
        // SKIP adding regular zone marker if this zone has an alert OR is high risk
        // Alert zones and high-risk zones will get a special alert marker instead (added later)
        // High-risk zones (red) should also get alert pins for visibility
        if (z.alert || riskLevel === 'high') {
            // Just update statistics, don't add marker (will be added as alert marker)
            totalHeight += z.veg_height_m;
            if (z.alert) {
                alertCount++;
                console.log(`  ✅ Zone ${z.id} has alert: veg=${z.veg_height_m}m, clearance=${z.clearance_m}m`);
            } else {
                console.log(`  ⚠️ Zone ${z.id} is high risk (no alert): veg=${z.veg_height_m}m, clearance=${z.clearance_m}m`);
            }
            riskCounts[riskLevel]++;
            return; // Skip adding regular zone marker
        }
        
        // Create custom icon based on risk level (only for non-alert zones)
        let iconHtml, iconClass;
        
        // Better-looking markers with correct positioning
        let markerColor, markerLabel;
        if (riskLevel === 'low') {
            markerColor = '#10b981';
            markerLabel = '✓';
        } else if (riskLevel === 'moderate') {
            markerColor = '#f59e0b';
            markerLabel = '!';
        } else {
            markerColor = '#ef4444';
            markerLabel = '⚠';
        }

        const zoneIcon = L.divIcon({
            className: 'clean-marker',
            html: `<div style="
                background-color: ${markerColor}; 
                width: 30px; 
                height: 30px; 
                border-radius: 50%; 
                border: 3px solid white; 
                box-shadow: 0 3px 8px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 16px;
            ">${markerLabel}</div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        });

        const marker = L.marker([centerLat, centerLon], { icon: zoneIcon }).addTo(map);
        
        // Enhanced tooltip with consistent risk labels
        const riskLabels = {
            'low': '<span style="color: #10b981; font-weight: bold;">🟢 Low Risk</span>',
            'moderate': '<span style="color: #f59e0b; font-weight: bold;">🟡 Moderate Risk</span>',
            'high': '<span style="color: #ef4444; font-weight: bold;">🔴 High Risk</span>'
        };
        
        marker.bindTooltip(`
            <div style="font-family: 'Poppins', sans-serif; padding: 8px; max-width: 300px;">
                <strong style="font-size: 1.1em; color: #1e293b;">Zone ${z.id}</strong><br>
                <div style="margin-top: 4px; padding: 4px 0; border-top: 1px solid #e2e8f0;">
                    Risk Level: ${riskLabels[riskLevel]}<br>
                    <i class="fas fa-tree"></i> Vegetation: <strong>${z.veg_height_m}m</strong><br>
                    <i class="fas fa-ruler-vertical"></i> Clearance: <strong>${z.clearance_m}m</strong><br>
                    ${z.growth_rate_cm_day ? `<i class="fas fa-chart-line"></i> Growth: <strong>${z.growth_rate_cm_day} cm/day</strong><br>` : ''}
                    ${z.days_until_breach ? `<i class="fas fa-calendar-alt"></i> Breach: <strong>${z.days_until_breach} days</strong> (${z.breach_date})<br>` : ''}
                    ${z.ground_elevation_m ? `<i class="fas fa-mountain"></i> Elevation: ${z.ground_elevation_m}m<br>` : ''}
                    ${z.utility_operator ? `<small style="color: #64748b;"><i class="fas fa-bolt"></i> Operator: <strong>${z.utility_operator}</strong></small><br>` : ''}
                    <small style="color: #64748b;"><i class="fas fa-map-pin"></i> ${centerLat.toFixed(4)}, ${centerLon.toFixed(4)}</small><br>
                    <small style="color: #94a3b8; font-style: italic;">${z.data_source || 'Simulated Data'}</small>
                </div>
            </div>
        `, { sticky: true });

        zoneLayers[z.id] = marker;
        console.log(`  📍 Added zone marker for Zone ${z.id} (${riskLevel} risk)`);
        
        // Update statistics
        totalHeight += z.veg_height_m;
        riskCounts[riskLevel]++;
    });
    
    // Update statistics display
    const avgHeight = data.zones.length > 0 ? (totalHeight / data.zones.length).toFixed(2) : 0;
    // Count zones with alert=true (not data.alerts.length which may be incomplete)
    console.log(`📊 Statistics: ${data.zones.length} zones, ${alertCount} alerts, ${Object.keys(zoneLayers).length} zone markers added`);
    document.getElementById('alertCount').innerText = alertCount;
    document.getElementById('zoneCount').innerText = data.zones.length;
    document.getElementById('avgHeight').innerText = avgHeight + 'm';
    
    // Update risk level counts in legend
    document.getElementById('lowCount').innerText = riskCounts.low;
    document.getElementById('moderateCount').innerText = riskCounts.moderate;
    document.getElementById('highCount').innerText = riskCounts.high;
    
    // Update AI Risk Assessment (ML predictions for highest risk zone)
    updateMLRiskAssessment(data.zones);
    
    // Timeline badge removed - using Bootstrap tabs now

    // Build alert list from zones with alert=true OR high risk level
    // High-risk zones (red) should also get alert pins for visibility, even if clearance > 6.0m
    const zonesWithAlerts = data.zones.filter(z => {
        const riskLevel = getRiskLevel(z.veg_height_m, z.clearance_m);
        return z.alert || riskLevel === 'high';
    });
    console.log('📊 Alert count - calculated during loop:', alertCount, 'zonesWithAlerts (alerts + high risk):', zonesWithAlerts.length, 'data.alerts array:', data.alerts.length, 'date:', dateStr);
    
    // Update the display with the count of zones that have alert pins (alerts + high risk)
    // This ensures the count matches the number of alert markers on the map
    const finalAlertCount = zonesWithAlerts.length;
    console.log(`✅ Setting alert count to ${finalAlertCount} (matches ${zonesWithAlerts.length} alert markers on map)`);
    document.getElementById('alertCount').innerText = finalAlertCount;
    
    if (zonesWithAlerts.length > 0) {
        console.log(`📍 Adding ${zonesWithAlerts.length} alert markers to map`);
        const alertZonesProcessed = new Set(); // Prevent duplicate alert markers
        zonesWithAlerts.forEach(z => {
            // Prevent duplicate alert markers
            if (alertZonesProcessed.has(z.id)) {
                console.warn(`⚠️ Duplicate alert marker for zone ${z.id}! Skipping.`);
                return;
            }
            alertZonesProcessed.add(z.id);
            
            // Calculate center of zone
            const centerLat = (z.bbox.min_lat + z.bbox.max_lat) / 2;
            const centerLon = (z.bbox.min_lon + z.bbox.max_lon) / 2;
            
            // Create custom icon for alert markers (consistent with zone markers)
            const alertIcon = L.divIcon({
                className: 'custom-alert-marker',
                html: '<i class="fas fa-location-dot" style="color: #ef4444; font-size: 32px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));"></i>',
                iconSize: [32, 32],
                iconAnchor: [16, 32]
            });
            
            const marker = L.marker([centerLat, centerLon], { icon: alertIcon }).addTo(map);
            marker.bindTooltip(`
                <div style="font-family: 'Poppins', sans-serif; padding: 8px; max-width: 300px;">
                    <strong style="font-size: 1.1em; color: #ef4444;">🚨 Zone ${z.id}</strong><br>
                    <div style="margin-top: 4px; padding: 4px 0; border-top: 1px solid #fee2e2;">
                        Risk Level: <span style="color: #ef4444; font-weight: bold;">🔴 High Risk</span><br>
                        <i class="fas fa-tree"></i> Vegetation: <strong>${z.veg_height_m}m</strong><br>
                        <i class="fas fa-ruler-vertical"></i> Clearance: <strong>${z.clearance_m}m</strong><br>
                        ${z.growth_rate_cm_day ? `<i class="fas fa-chart-line"></i> Growth: <strong>${z.growth_rate_cm_day} cm/day</strong><br>` : ''}
                        ${z.days_until_breach ? `<i class="fas fa-calendar-alt"></i> Breach: <strong>${z.days_until_breach} days</strong><br>` : ''}
                        <small style="color: #64748b;"><i class="fas fa-map-pin"></i> ${centerLat.toFixed(4)}, ${centerLon.toFixed(4)}</small>
                    </div>
                </div>
            `, { sticky: true });
            alertMarkers.push(marker);
            console.log(`  📍 Added alert marker for Zone ${z.id}`);

            const item = document.createElement('div');
            item.className = 'alert-item';
            item.innerHTML = `
                <strong><i class="fas fa-map-marker-alt"></i> Zone ${z.id}</strong><br>
                <i class="fas fa-tree"></i> Vegetation: ${z.veg_height_m}m<br>
                <i class="fas fa-ruler-vertical"></i> Clearance: ${z.clearance_m}m
            `;
            alertItems.appendChild(item);
            
            // Also add to predictor tab's alert list
            const itemPredictor = item.cloneNode(true);
            alertItemsPredictor.appendChild(itemPredictor);

            if ("Notification" in window && Notification.permission === "granted") {
                new Notification(`🔥 ALERT! Zone ${z.id}`, {
                    body: `Vegetation Height: ${z.veg_height_m}m | Clearance: ${z.clearance_m}m`,
                    icon: 'https://em-content.zobj.net/thumbs/120/apple/325/fire_1f525.png'
                });
            }
        });
    } else {
        const noAlertsHtml = '<div style="text-align: center; color: var(--text-dark); padding: 10px;"><i class="fas fa-check-circle" style="color: #56ab2f; font-size: 1.5rem;"></i><br>No active alerts</div>';
        alertItems.innerHTML = noAlertsHtml;
        alertItemsPredictor.innerHTML = noAlertsHtml;
    }
    
    // Final marker count verification
    const totalMarkers = Object.keys(zoneLayers).length + alertMarkers.length;
    console.log(`✅ Map update complete: ${Object.keys(zoneLayers).length} zone markers + ${alertMarkers.length} alert markers = ${totalMarkers} total (should match ${data.zones.length} zones)`);
    if (totalMarkers !== data.zones.length) {
        console.warn(`⚠️ Marker count mismatch! Expected ${data.zones.length} markers, got ${totalMarkers}`);
    }
    
    // Restore button states after update (preserve play/pause state)
    if (wasPlaying !== undefined) {
        updateButtonStates(wasPlaying);
    }
    } catch (error) {
        console.error('❌ Error in updateMapByDate:', error);
        alert('Error loading map data: ' + error.message);
        // Restore button states even on error
        if (wasPlaying !== undefined) {
            updateButtonStates(wasPlaying);
        }
    }
}

function playSimulation() {
    if (isPlaying) {
        console.log('⚠️ Already playing, ignoring play request');
        return;
    }
    
    // Ensure we have a valid currentDate
    if (!currentDate || dateList.indexOf(currentDate) === -1) {
        const slider = document.getElementById('dayRange');
        if (slider) {
            const index = parseInt(slider.value) || 0;
            currentDate = dateList[index] || dateList[0];
        } else {
            currentDate = dateList[0];
        }
    }
    
    console.log('▶️ Starting simulation from date:', currentDate);
    isPlaying = true;
    
    // Clear any existing interval
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
    
    // Store interval ID in a way we can access it
    const currentIntervalId = setInterval(() => {
        // CRITICAL: Check if paused FIRST thing
        if (!isPlaying) {
            console.log('⏸️ Simulation paused in interval, clearing:', currentIntervalId);
            clearInterval(currentIntervalId);
            intervalId = null;
            updateButtonStates(false);
            return;
        }
        
        // Verify interval ID matches
        if (intervalId !== currentIntervalId) {
            console.log('⚠️ Interval ID mismatch, stopping');
            clearInterval(currentIntervalId);
            return;
        }
        
        // Get current index
        let index = dateList.indexOf(currentDate);
        
        // Safety check - if date not found, use current slider position
        if (index === -1) {
            const slider = document.getElementById('dayRange');
            if (slider) {
                index = parseInt(slider.value) || 0;
            } else {
                index = 0;
            }
        }
        
        // Check if we've reached the end
        if (index >= dateList.length - 1) {
            console.log('🏁 Reached end of simulation');
            isPlaying = false;
            clearInterval(currentIntervalId);
            intervalId = null;
            updateButtonStates(false);
            return;
        }
        
        // Move to next date
        const nextIndex = index + 1;
        currentDate = dateList[nextIndex];
        console.log('📅 Moving to date:', currentDate, `(${nextIndex + 1}/${dateList.length})`);
        
        // Update map with new date
        updateMapByDate(currentDate);
    }, 800);
    
    // Store the interval ID
    intervalId = currentIntervalId;
    
    // Update button states
    updateButtonStates(true); // true = playing
}

// Helper function to update button states
function updateButtonStates(playing) {
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    
    if (playing) {
        // Show pause, hide play
        if (playBtn) {
            playBtn.style.display = 'none';
            playBtn.style.visibility = 'hidden';
        }
        if (pauseBtn) {
            pauseBtn.style.display = 'inline-block';
            pauseBtn.style.visibility = 'visible';
        }
    } else {
        // Show play, hide pause
        if (playBtn) {
            playBtn.style.display = 'inline-block';
            playBtn.style.visibility = 'visible';
        }
        if (pauseBtn) {
            pauseBtn.style.display = 'none';
            pauseBtn.style.visibility = 'hidden';
        }
    }
}

function pauseSimulation() {
    console.log('⏸️ Pausing simulation, isPlaying was:', isPlaying, 'intervalId:', intervalId);
    
    // CRITICAL: Set flag FIRST before clearing interval
    isPlaying = false;
    
    // Force clear interval - use multiple methods to ensure it's cleared
    if (intervalId !== null && intervalId !== undefined) {
        try {
            clearInterval(intervalId);
            console.log('✅ Interval cleared:', intervalId);
        } catch (e) {
            console.error('❌ Error clearing interval:', e);
        }
        intervalId = null;
    }
    
    // Double-check: clear any remaining intervals
    // This is a safety measure - shouldn't be needed but helps debug
    let cleared = 0;
    for (let i = 1; i < 10000; i++) {
        try {
            clearInterval(i);
            cleared++;
        } catch (e) {
            // Ignore errors
        }
    }
    if (cleared > 0) {
        console.log('⚠️ Cleared', cleared, 'additional intervals');
    }
    
    // Update button states
    updateButtonStates(false); // false = paused
    console.log('✅ Pause complete - buttons updated, isPlaying:', isPlaying);
}

// Wait for DOM to be fully loaded before setting up event listeners
(async function init() {
    try {
        console.log('🚀 Starting initialization...');
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
            return;
        }
        
        console.log('📡 Fetching metadata...');
        if ("Notification" in window) Notification.requestPermission();
        
        const metadata = await fetchMetadata();
        console.log('✅ Metadata fetched, dates:', dateList.length);
        
        if (!dateList || dateList.length === 0) {
            console.error('❌ No dates found in metadata!');
            return;
        }
        
        console.log('🗺️ Drawing transmission lines...');
        drawLines(metadata.lines);
        
        console.log('📅 Setting initial date...');
        currentDate = dateList[0];
        console.log('Initial date:', currentDate);
        
        console.log('🗺️ Loading zones for initial date...');
        await updateMapByDate(currentDate);
        console.log('✅ Zones loaded!');

        // Setup slider (retry if not ready)
        function setupSlider() {
            const slider = document.getElementById('dayRange');
            if (slider && dateList.length > 0) {
                slider.max = dateList.length - 1;
                
                // Remove old handler if it exists
                if (slider._handler) {
                    slider.removeEventListener('input', slider._handler);
                    slider.removeEventListener('change', slider._handler);
                }
                
                // Create new handler with better logging
                slider._handler = function(e) {
                    if (isUpdatingSlider) {
                        console.log('⏸️ Slider update skipped (isUpdatingSlider flag)');
                        return;
                    }
                    
                    const index = parseInt(e.target.value);
                    console.log('📅 Slider moved to index:', index, 'of', dateList.length, 'isPlaying:', isPlaying);
                    
                    // Pause simulation if playing
                    if (isPlaying) {
                        console.log('⏸️ Pausing simulation due to slider movement');
                        isPlaying = false;
                        if (intervalId) {
                            clearInterval(intervalId);
                            intervalId = null;
                        }
                        updateButtonStates(false);
                    }
                    
                    if (index >= 0 && index < dateList.length) {
                        currentDate = dateList[index];
                        console.log('📅 Updating map to date:', currentDate);
                        // Don't preserve playing state - we're manually moving slider
                        updateMapByDate(currentDate);
                    } else {
                        console.warn('⚠️ Invalid slider index:', index);
                    }
                };
                // Add both input and change listeners for better compatibility
                slider.addEventListener('input', slider._handler);
                slider.addEventListener('change', slider._handler);
                console.log('✅ Slider initialized with max:', slider.max);
            } else {
                console.log('⏳ Slider not ready, retrying...');
                setTimeout(setupSlider, 100);
            }
        }
        setupSlider();
        
        // Also setup slider when Time Predictor tab is shown (Bootstrap tab event)
        const tab2Btn = document.getElementById('tab2-btn');
        if (tab2Btn) {
            tab2Btn.addEventListener('shown.bs.tab', function() {
                console.log('📑 Time Predictor tab shown, re-initializing slider');
                setTimeout(setupSlider, 50);
            });
        }
        
        console.log('✅ App initialized successfully!');
    } catch (error) {
        console.error('❌ Initialization error:', error);
        alert('Error loading app: ' + error.message);
    }
})();

// Use event delegation for buttons (works even if buttons aren't in DOM yet)
// This MUST be outside the async function so it runs immediately
document.addEventListener('click', function(e) {
    // Check if click is on play button or its icon
    const playBtn = e.target.closest('#playBtn');
    const pauseBtn = e.target.closest('#pauseBtn');
    const resetBtn = e.target.closest('#resetBtn');
    
    // Play button
    if (playBtn) {
        e.preventDefault();
        e.stopPropagation();
        console.log('▶️ Play button clicked, isPlaying:', isPlaying);
        if (!isPlaying) {
            playSimulation();
        }
        return false;
    }
    
    // Pause button  
    if (pauseBtn) {
        e.preventDefault();
        e.stopPropagation();
        console.log('⏸️ Pause button clicked, isPlaying:', isPlaying, 'intervalId:', intervalId);
        // Force pause regardless of state
        isPlaying = false;
        pauseSimulation();
        return false;
    }
    
    // Reset button
    if (resetBtn) {
        e.preventDefault();
        e.stopPropagation();
        console.log('⏮️ Reset button clicked');
        pauseSimulation();
        if (dateList.length > 0) {
            currentDate = dateList[0];
            updateMapByDate(currentDate);
        }
        return false;
    }
});

// California approximate bounds
const CA_BOUNDS = {
    minLat: 32.5,
    maxLat: 42.0,
    minLon: -124.5,
    maxLon: -114.0
};

// Store transmission tower markers
let transmissionTowerMarkers = [];

// City search using OpenStreetMap Nominatim API (free, no API key needed)
let searchTimeout = null;
let selectedCity = null;
let currentLocation = null; // Store current location for Predictor tab (lat, lon, radius)

// Search for cities using Nominatim API
async function searchCities(query) {
    if (!query || query.length < 2) {
        document.getElementById('citySuggestions').style.display = 'none';
        return;
    }

    const suggestionsDiv = document.getElementById('citySuggestions');
    
    // Show loading indicator
    suggestionsDiv.innerHTML = '<div class="list-group-item"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
    suggestionsDiv.style.display = 'block';

    try {
        // Use Flask proxy endpoint to avoid CORS issues
        const url = `/api/search_cities?q=${encodeURIComponent(query)}`;
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Use cities from API response
        const cities = result.cities || [];
        displayCitySuggestions(cities);
    } catch (error) {
        console.error('Error searching cities:', error);
        suggestionsDiv.innerHTML = 
            '<div class="list-group-item text-danger"><i class="fas fa-exclamation-triangle"></i> Error searching cities. Please try again or use coordinates.</div>';
        suggestionsDiv.style.display = 'block';
    }
}

// Display city suggestions
function displayCitySuggestions(cities) {
    const suggestionsDiv = document.getElementById('citySuggestions');
    
    if (!cities || cities.length === 0) {
        suggestionsDiv.innerHTML = '<div class="list-group-item">No cities found</div>';
        suggestionsDiv.style.display = 'block';
        return;
    }
    
    suggestionsDiv.innerHTML = cities.map(city => {
        const displayName = city.display_name.split(',')[0]; // City name
        const state = city.address?.state || 'California';
        return `
            <a href="#" class="list-group-item list-group-item-action city-suggestion" 
               data-lat="${city.lat}" 
               data-lon="${city.lon}"
               data-name="${displayName}">
                <strong>${displayName}</strong>
                <small class="text-muted d-block">${state}</small>
            </a>
        `;
    }).join('');
    
    suggestionsDiv.style.display = 'block';
    
    // Add click handlers
    document.querySelectorAll('.city-suggestion').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const lat = parseFloat(item.dataset.lat);
            const lon = parseFloat(item.dataset.lon);
            const name = item.dataset.name;
            
            selectedCity = { lat, lon, name };
            document.getElementById('citySearch').value = name;
            document.getElementById('selectedCityLat').value = lat;
            document.getElementById('selectedCityLon').value = lon;
            suggestionsDiv.style.display = 'none';
        });
    });
}

// Initialize city search
document.addEventListener('DOMContentLoaded', () => {
    const citySearch = document.getElementById('citySearch');
    const suggestionsDiv = document.getElementById('citySuggestions');
    
    if (citySearch) {
        // Search as user types (with debounce)
        citySearch.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            const query = e.target.value.trim();
            
            if (query.length >= 2) {
                searchTimeout = setTimeout(() => searchCities(query), 300); // Wait 300ms after typing stops
            } else {
                suggestionsDiv.style.display = 'none';
                selectedCity = null;
            }
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!citySearch.contains(e.target) && !suggestionsDiv.contains(e.target)) {
                suggestionsDiv.style.display = 'none';
            }
        });
        
        // Clear selection when input is cleared
        citySearch.addEventListener('focus', () => {
            if (citySearch.value.length >= 2) {
                searchCities(citySearch.value);
            }
        });
    }
});

document.getElementById('goBtn').addEventListener('click', async () => {
    let lat = parseFloat(document.getElementById('latInput').value);
    let lon = parseFloat(document.getElementById('lonInput').value);
    const selectedLat = parseFloat(document.getElementById('selectedCityLat').value);
    const selectedLon = parseFloat(document.getElementById('selectedCityLon').value);

    let targetLat, targetLon;

    // Check if city is selected from search
    if (!isNaN(selectedLat) && !isNaN(selectedLon)) {
        targetLat = selectedLat;
        targetLon = selectedLon;
    } else if (!isNaN(lat) && !isNaN(lon)) {
        // Focus on input coordinates
        if (lat < CA_BOUNDS.minLat || lat > CA_BOUNDS.maxLat || lon < CA_BOUNDS.minLon || lon > CA_BOUNDS.maxLon) {
            alert("Data for electric transmission lines is only available in California!");
            return;
        }
        targetLat = lat;
        targetLon = lon;
    } else {
        alert("Please select a city or enter coordinates!");
        return;
    }

    // RESET METRICS FIRST to prevent showing stale data
    console.log('🔄 Resetting metrics before new scan...');
    document.getElementById('alertCount').innerText = '0';
    document.getElementById('zoneCount').innerText = '0';
    document.getElementById('avgHeight').innerText = '0.0m';
    document.getElementById('lowCount').innerText = '0';
    document.getElementById('moderateCount').innerText = '0';
    document.getElementById('highCount').innerText = '0';
    document.getElementById('mlRiskLevel').innerText = '-';
    document.getElementById('mlRiskScore').innerText = '-';
    document.getElementById('mlConfidence').innerText = '-';
    document.getElementById('growthRateDisplay').innerText = '-';
    document.getElementById('breachPrediction').innerText = '-';

    // Clear old transmission tower markers BEFORE fetching new data
    console.log(`🗑️ Clearing ${transmissionTowerMarkers.length} old tower markers...`);
    transmissionTowerMarkers.forEach(marker => {
        try {
            map.removeLayer(marker);
        } catch (e) {
            console.warn('⚠️ Error removing tower marker:', e);
        }
    });
    transmissionTowerMarkers = [];
    
    // Clear old zones when searching a city (to avoid confusion)
    Object.values(zoneLayers).forEach(layer => {
        try {
            map.removeLayer(layer);
        } catch (e) {
            console.warn('⚠️ Error removing zone layer:', e);
        }
    });
    zoneLayers = {};
    alertMarkers.forEach(m => {
        try {
            map.removeLayer(m);
        } catch (e) {
            console.warn('⚠️ Error removing alert marker:', e);
        }
    });
    alertMarkers = [];

    // Store current location for Predictor tab
    // Use a larger radius for zones (25km) since zones cover larger areas than individual towers
    currentLocation = {
        lat: targetLat,
        lon: targetLon,
        radius: 25 // km - larger radius for zones (towers use 15km)
    };
    console.log(`📍 Storing location for Predictor: ${targetLat}, ${targetLon}, radius ${currentLocation.radius}km`);
    
    // Smooth zoom/fly to the location
    map.flyTo([targetLat, targetLon], 11, { animate: true, duration: 2.0 });
    
    // Show loading indicator
    const alertItemsElement = document.getElementById('alertItems');
    alertItemsElement.innerHTML = '<div style="text-align: center; padding: 20px;"><i class="fas fa-spinner fa-spin" style="font-size: 24px; color: #6366f1;"></i><br><small>Scanning transmission towers...</small></div>';
    
    // Fetch transmission tower data with risk metrics
    try {
        console.log(`🔍 Fetching towers for location: ${targetLat}, ${targetLon}`);
        const response = await fetch(`/api/transmission_lines?lat=${targetLat}&lon=${targetLon}&radius=15`);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Tower data received:', data.towers_found, 'towers');
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Markers and zones already cleared above, now draw new towers
        console.log(`✅ Drawing ${data.towers_found} new tower markers...`);
        
        // Draw transmission towers on map with risk metrics
        if (data.towers && data.towers.length > 0) {
            data.towers.forEach(tower => {
                // Determine marker color based on risk level
                let markerColor, markerLabel;
                if (tower.risk_level === 'low') {
                    markerColor = '#10b981';
                    markerLabel = '⚡';
                } else if (tower.risk_level === 'moderate') {
                    markerColor = '#f59e0b';
                    markerLabel = '⚡';
                } else {
                    markerColor = '#ef4444';
                    markerLabel = '⚡';
                }

                const towerIcon = L.divIcon({
                    className: 'clean-marker',
                    html: `<div style="
                        background-color: ${markerColor}; 
                        width: 28px; 
                        height: 28px; 
                        border-radius: 50%; 
                        border: 3px solid white; 
                        box-shadow: 0 3px 8px rgba(0,0,0,0.3);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        font-size: 14px;
                    ">${markerLabel}</div>`,
                    iconSize: [28, 28],
                    iconAnchor: [14, 14]
                });

                const marker = L.marker([tower.latitude, tower.longitude], { icon: towerIcon }).addTo(map);
                
                // Enhanced tooltip with tower metrics
                const riskLabels = {
                    'low': '<span style="color: #10b981; font-weight: bold;">🟢 Low Risk</span>',
                    'moderate': '<span style="color: #f59e0b; font-weight: bold;">🟡 Moderate Risk</span>',
                    'high': '<span style="color: #ef4444; font-weight: bold;">🔴 High Risk</span>'
                };
                
                const alertBadge = tower.alert ? '<span style="background: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7em;">🚨 ALERT</span>' : '';
                
                marker.bindTooltip(`
                    <div style="font-family: 'Poppins', sans-serif; padding: 8px; max-width: 300px;">
                        <strong style="font-size: 1.1em; color: #1e293b;">⚡ Tower ${tower.tower_id}</strong> ${alertBadge}<br>
                        <div style="margin-top: 4px; padding: 4px 0; border-top: 1px solid #e2e8f0;">
                            Risk Level: ${riskLabels[tower.risk_level]}<br>
                            <i class="fas fa-tree"></i> Vegetation: <strong>${tower.veg_height_m}m</strong><br>
                            <i class="fas fa-ruler-vertical"></i> Clearance: <strong>${tower.clearance_m}m</strong><br>
                            <i class="fas fa-chart-line"></i> Growth: <strong>${tower.growth_rate_cm_day} cm/day</strong><br>
                            <i class="fas fa-calendar-alt"></i> Breach: <strong>${tower.days_until_breach} days</strong><br>
                            <small style="color: #64748b;"><i class="fas fa-bolt"></i> ${tower.owner} | ${tower.voltage}</small><br>
                            <small style="color: #64748b;"><i class="fas fa-tools"></i> ${tower.structure_type}</small><br>
                            <small style="color: #94a3b8; font-style: italic;">Last inspected: ${tower.last_inspection}</small>
                        </div>
                    </div>
                `, { sticky: true });
                
                transmissionTowerMarkers.push(marker);
            });
            
            // Update info panel with statistics
            const stats = data.statistics;
            const sourceNote = data.data_source === 'simulated' ? '<br><em style="font-size: 0.55rem; color: #94a3b8;">Demo Mode - Production connects to utility SCADA systems</em>' : '';
            alertItemsElement.innerHTML = `
                <div style="padding: 10px; background: rgba(99, 102, 241, 0.1); border-radius: 8px; border-left: 3px solid #6366f1;">
                    <strong style="color: #4f46e5;"><i class="fas fa-broadcast-tower"></i> ${data.towers_found} Transmission Towers Monitored</strong><br>
                    <div style="margin-top: 8px; padding: 6px; background: rgba(255,255,255,0.5); border-radius: 4px;">
                        <small style="color: #ef4444; font-weight: bold;">🔴 ${stats.critical_alerts} Critical Alerts</small><br>
                        <small style="color: #64748b;">
                            🟢 Low: ${stats.risk_distribution.low} | 
                            🟡 Moderate: ${stats.risk_distribution.moderate} | 
                            🔴 High: ${stats.risk_distribution.high}
                        </small><br>
                        <small style="color: #64748b;">Avg Vegetation: ${stats.avg_vegetation_height}m</small>
                    </div>
                    <small style="color: #64748b; margin-top: 4px; display: block;">Location: ${targetLat.toFixed(4)}, ${targetLon.toFixed(4)}<br>
                    Radius: ${data.radius_km}km${sourceNote}</small>
                </div>
            `;
            
            // Update RIGHT PANEL Live Metrics to show tower data
            document.getElementById('alertCount').innerText = stats.critical_alerts;
            document.getElementById('zoneCount').innerText = stats.total_towers;
            document.getElementById('avgHeight').innerText = stats.avg_vegetation_height + 'm';
            
            // Update Risk Levels in right panel
            document.getElementById('lowCount').innerText = stats.risk_distribution.low;
            document.getElementById('moderateCount').innerText = stats.risk_distribution.moderate;
            document.getElementById('highCount').innerText = stats.risk_distribution.high;
            
            // Update ML Risk Assessment for highest risk tower
            const highRiskTowers = data.towers.filter(t => t.risk_level === 'high' || t.alert);
            if (highRiskTowers.length > 0) {
                const worstTower = highRiskTowers.reduce((prev, curr) => 
                    curr.clearance_m < prev.clearance_m ? curr : prev
                );
                
                document.getElementById('growthRateDisplay').innerText = `${worstTower.growth_rate_cm_day} cm/day`;
                
                if (worstTower.days_until_breach <= 0) {
                    document.getElementById('breachPrediction').innerHTML = '<span style="color: #ef4444; font-weight: bold;">🔴 BREACHED</span>';
                } else if (worstTower.days_until_breach < 999) {
                    document.getElementById('breachPrediction').innerText = `${worstTower.days_until_breach} days`;
                } else {
                    document.getElementById('breachPrediction').innerText = 'Safe';
                }
                
                // Call ML prediction for this tower
                try {
                    console.log('🤖 Fetching ML prediction for tower:', worstTower.tower_id);
                    const mlRes = await fetch(`/api/ml_predict?veg_height=${worstTower.veg_height_m}&clearance=${worstTower.clearance_m}&temperature=25&humidity=50&wind_speed=5&days_since_rain=7&latitude=${worstTower.latitude}&longitude=${worstTower.longitude}`);
                    
                    if (!mlRes.ok) {
                        throw new Error(`ML API error: ${mlRes.status}`);
                    }
                    
                    const mlData = await mlRes.json();
                    console.log('✅ ML prediction received:', mlData);
                    
                    if (mlData.success && mlData.prediction) {
                        const pred = mlData.prediction;
                        const riskEmoji = {
                            'Low': '🟢',
                            'Moderate': '🟡',
                            'High': '🟠',
                            'Critical': '🔴'
                        };
                        document.getElementById('mlRiskLevel').innerText = `${riskEmoji[pred.risk_level] || ''} ${pred.risk_level}`;
                        document.getElementById('mlRiskScore').innerText = `${pred.risk_percentage.toFixed(1)}%`;
                        document.getElementById('mlConfidence').innerText = `${(pred.confidence * 100).toFixed(0)}%`;
                    } else {
                        console.warn('⚠️ ML prediction failed:', mlData);
                        document.getElementById('mlRiskLevel').innerText = '-';
                        document.getElementById('mlRiskScore').innerText = '-';
                        document.getElementById('mlConfidence').innerText = '-';
                    }
                } catch (mlError) {
                    console.error('❌ Error fetching ML prediction for tower:', mlError);
                    document.getElementById('mlRiskLevel').innerText = '-';
                    document.getElementById('mlRiskScore').innerText = '-';
                    document.getElementById('mlConfidence').innerText = '-';
                }
            }
        } else {
            console.warn('⚠️ No towers returned from API');
            alertItemsElement.innerHTML = '<div style="text-align: center; color: #64748b; padding: 10px;"><i class="fas fa-info-circle"></i><br>No transmission towers found in this area</div>';
            // Reset metrics to zero
            document.getElementById('alertCount').innerText = '0';
            document.getElementById('zoneCount').innerText = '0';
            document.getElementById('avgHeight').innerText = '0.0m';
        }
    } catch (error) {
        console.error('❌ Error fetching transmission towers:', error);
        alertItemsElement.innerHTML = `<div style="text-align: center; color: #ef4444; padding: 10px;"><i class="fas fa-exclamation-triangle"></i><br>Error loading transmission infrastructure<br><small>${error.message}</small></div>`;
        // Reset metrics on error
        document.getElementById('alertCount').innerText = '0';
        document.getElementById('zoneCount').innerText = '0';
        document.getElementById('avgHeight').innerText = '0.0m';
    }
});


// ========== TOGGLE PANELS ==========
// Bootstrap tabs handle tab switching automatically
const toggleBtn = document.getElementById('togglePanelsBtn');
const mergedPanel = document.getElementById('mergedPanel');
const metricsPanel = document.getElementById('metricsPanel');
let panelsVisible = true;

toggleBtn.addEventListener('click', () => {
    panelsVisible = !panelsVisible;
    if (panelsVisible) {
        mergedPanel.classList.remove('hidden');
        metricsPanel.classList.remove('hidden');
    } else {
        mergedPanel.classList.add('hidden');
        metricsPanel.classList.add('hidden');
    }
});

// ========== NOTIFY AUTHORITY BUTTON ==========
document.getElementById('notifyAuthorityBtn').addEventListener('click', async () => {
    const alertCount = parseInt(document.getElementById('alertCount').innerText);
    if (alertCount === 0) {
        alert('No active alerts to report.');
        return;
    }
    
    const confirmed = confirm(`Send notification for ${alertCount} critical alert(s) to authorities?`);
    if (confirmed) {
        try {
            const response = await fetch('/api/notify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    date: currentDate,
                    alert_count: alertCount
                })
            });
            
            if (response.ok) {
                alert('✅ Authorities have been notified!');
            } else {
                alert('⚠️  Failed to send notification. Please try again.');
            }
        } catch (error) {
            console.error('Error sending notification:', error);
            alert('⚠️  Error sending notification. Please check your connection.');
        }
    }
});

// ========== INITIALIZE: Load zones on page load ==========
// This is handled by the init() function above, so we don't need duplicate initialization

// ========== TAB SWITCHING: Load zones when switching to Time Predictor ==========
const tab2Btn = document.getElementById('tab2-btn');
if (tab2Btn) {
    tab2Btn.addEventListener('click', async () => {
        // Clear any transmission tower markers
        transmissionTowerMarkers.forEach(marker => map.removeLayer(marker));
        transmissionTowerMarkers = [];
        
        // Load zones (will be filtered by currentLocation if a city was selected)
        if (dateList.length > 0) {
            currentDate = dateList[0];
            if (currentLocation) {
                console.log(`📍 Loading Predictor for location: ${currentLocation.lat}, ${currentLocation.lon}, radius ${currentLocation.radius}km`);
                // Zoom to the selected location
                map.flyTo([currentLocation.lat, currentLocation.lon], 11, { animate: true, duration: 1.0 });
            } else {
                console.log('📍 Loading Predictor with default zones (no city selected)');
            }
            await updateMapByDate(currentDate);
        }
        
        // Reset the info text in Tab 1
        const alertItems = document.getElementById('alertItems');
        if (alertItems) {
            alertItems.innerHTML = 'Select a city or enter coordinates to scan transmission towers with vegetation risk metrics';
        }
        
        // Controls are already set up via event delegation, no need to re-setup
        
        console.log('✅ Predictor tab loaded');
    });
}
