"""
Synthetic Data Generator for Fire App
Generates realistic LIDAR canopy height and vegetation spread data
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import json


class SyntheticDataGenerator:
    """Generate realistic synthetic data for vegetation monitoring"""
    
    def __init__(self, random_seed=42):
        """Initialize the generator with a random seed for reproducibility"""
        np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def generate_lidar_canopy_data(self, zones, resolution=10):
        """
        Generate synthetic LIDAR canopy height data for each zone.
        
        Args:
            zones: List of zone dictionaries with lat/lon bounds
            resolution: Number of sample points per zone (10x10 grid = 100 points)
            
        Returns:
            dict: LIDAR data for each zone with height measurements
        """
        lidar_data = {}
        
        for zone in zones:
            zone_id = zone['id']
            
            # Create a grid of points within the zone
            min_lat, max_lat = zone['min_lat'], zone['max_lat']
            min_lon, max_lon = zone['min_lon'], zone['max_lon']
            
            # Generate grid points
            lat_points = np.linspace(min_lat, max_lat, resolution)
            lon_points = np.linspace(min_lon, max_lon, resolution)
            
            # Create meshgrid
            lat_grid, lon_grid = np.meshgrid(lat_points, lon_points)
            
            # Generate realistic canopy heights based on zone characteristics
            base_height = self._get_base_height_for_zone(zone_id)
            
            # Add spatial variation (some areas denser than others)
            height_variation = np.random.normal(0, 2.0, (resolution, resolution))
            
            # Add realistic patterns (e.g., clusters of tall trees)
            for _ in range(3):  # Add 3 clusters of tall vegetation
                cluster_lat_idx = np.random.randint(0, resolution)
                cluster_lon_idx = np.random.randint(0, resolution)
                cluster_radius = resolution // 4
                
                for i in range(resolution):
                    for j in range(resolution):
                        dist = np.sqrt((i - cluster_lat_idx)**2 + (j - cluster_lon_idx)**2)
                        if dist < cluster_radius:
                            height_variation[i, j] += 3.0 * np.exp(-dist / cluster_radius)
            
            # Calculate final heights
            canopy_heights = np.maximum(0, base_height + height_variation)
            
            # Store data
            lidar_data[zone_id] = {
                'zone_id': zone_id,
                'resolution': resolution,
                'lat_grid': lat_grid.tolist(),
                'lon_grid': lon_grid.tolist(),
                'canopy_heights': canopy_heights.tolist(),
                'statistics': {
                    'mean_height': float(np.mean(canopy_heights)),
                    'max_height': float(np.max(canopy_heights)),
                    'min_height': float(np.min(canopy_heights)),
                    'std_height': float(np.std(canopy_heights)),
                    'median_height': float(np.median(canopy_heights)),
                    'percentile_75': float(np.percentile(canopy_heights, 75)),
                    'percentile_95': float(np.percentile(canopy_heights, 95))
                },
                'vegetation_type': self._get_vegetation_type(zone_id),
                'density': self._calculate_density(canopy_heights),
                'coverage_percentage': float(np.sum(canopy_heights > 0.5) / (resolution * resolution) * 100)
            }
        
        return lidar_data
    
    def generate_vegetation_spread_data(self, zones, weeks=8):
        """
        Generate vegetation spread patterns over time.
        Shows how vegetation coverage expands spatially.
        
        Args:
            zones: List of zone dictionaries
            weeks: Number of weeks to simulate
            
        Returns:
            dict: Vegetation spread data showing coverage expansion
        """
        spread_data = {}
        
        for zone in zones:
            zone_id = zone['id']
            
            # Initial coverage percentage
            initial_coverage = np.random.uniform(40, 80)
            
            # Growth rate (coverage expansion rate)
            spread_rate = self._get_spread_rate_for_zone(zone_id)
            
            # Weekly coverage data
            weekly_coverage = []
            weekly_spread_direction = []
            
            for week in range(weeks):
                # Coverage expands over time but plateaus at max
                max_coverage = 95  # Can't cover 100% due to natural gaps
                current_coverage = min(
                    max_coverage,
                    initial_coverage + spread_rate * week + np.random.uniform(-2, 2)
                )
                
                # Dominant spread direction (wind patterns, slope, etc.)
                direction = self._get_spread_direction(zone_id, week)
                
                # Spread velocity (meters per week)
                spread_velocity = spread_rate * 0.5 + np.random.uniform(-0.1, 0.1)
                
                weekly_coverage.append({
                    'week': week,
                    'coverage_percentage': float(current_coverage),
                    'spread_direction': direction,
                    'spread_velocity_m_per_week': float(spread_velocity),
                    'new_area_covered_m2': float(spread_velocity * 10)  # Rough estimate
                })
            
            spread_data[zone_id] = {
                'zone_id': zone_id,
                'initial_coverage': float(initial_coverage),
                'spread_rate': float(spread_rate),
                'weekly_data': weekly_coverage,
                'spread_pattern': self._get_spread_pattern(zone_id),
                'environmental_factors': {
                    'soil_moisture': np.random.uniform(0.2, 0.7),
                    'sun_exposure': np.random.uniform(0.5, 1.0),
                    'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
                    'slope_direction': np.random.choice(['N', 'S', 'E', 'W', 'flat'])
                }
            }
        
        return spread_data
    
    def generate_realistic_growth_simulation(self, lidar_data, spread_data, weeks=8):
        """
        Generate realistic vegetation growth simulation using LIDAR and spread data.
        
        Args:
            lidar_data: LIDAR canopy height data
            spread_data: Vegetation spread data
            weeks: Number of weeks to simulate
            
        Returns:
            dict: Growth simulation for each zone over time
        """
        growth_simulation = {}
        
        for zone_id in lidar_data.keys():
            # Get initial conditions
            initial_height = lidar_data[zone_id]['statistics']['mean_height']
            initial_coverage = spread_data[zone_id]['initial_coverage']
            
            # Growth parameters based on zone characteristics
            growth_params = self._get_growth_parameters(zone_id)
            
            weekly_heights = []
            
            for week in range(weeks):
                # Calculate height growth
                # Growth is affected by: season, moisture, coverage, existing height
                
                # Base growth rate
                base_rate = growth_params['base_weekly_growth']
                
                # Seasonal factor (varies by month)
                month = (date.today().month + week // 4) % 12 + 1
                seasonal_factor = self._get_seasonal_factor(month)
                
                # Coverage factor (more coverage = more competition = slower growth)
                coverage = spread_data[zone_id]['weekly_data'][week]['coverage_percentage']
                coverage_factor = 1.0 - (coverage / 100) * 0.3  # Max 30% reduction
                
                # Height factor (diminishing returns as vegetation gets taller)
                current_height = initial_height + sum([h['height_growth'] for h in weekly_heights])
                height_factor = 1.0 / (1.0 + current_height / 10.0)
                
                # Random variation
                random_factor = np.random.uniform(0.85, 1.15)
                
                # Calculate weekly growth
                weekly_growth = (
                    base_rate * 
                    seasonal_factor * 
                    coverage_factor * 
                    height_factor * 
                    random_factor
                )
                
                # Ensure non-negative
                weekly_growth = max(0, weekly_growth)
                
                # Store data
                weekly_heights.append({
                    'week': week,
                    'height_growth': float(weekly_growth),
                    'cumulative_height': float(initial_height + sum([h['height_growth'] for h in weekly_heights]) + weekly_growth),
                    'seasonal_factor': float(seasonal_factor),
                    'coverage_factor': float(coverage_factor),
                    'height_factor': float(height_factor)
                })
            
            growth_simulation[zone_id] = {
                'zone_id': zone_id,
                'initial_height': float(initial_height),
                'growth_parameters': growth_params,
                'weekly_heights': weekly_heights,
                'vegetation_type': lidar_data[zone_id]['vegetation_type']
            }
        
        return growth_simulation
    
    # Helper methods
    
    def _get_base_height_for_zone(self, zone_id):
        """Get realistic base canopy height for a zone"""
        # Different zones have different vegetation types and initial risk levels
        # LINE_HEIGHT = 8.0m, THRESHOLD = 6.0m
        # Red alert: clearance <= 6.0m (height >= 2.0m)
        # Yellow (moderate): clearance 6.0-7.5m (height 0.5-2.0m)
        # Green (safe): clearance > 7.5m (height < 0.5m)
        
        zone_types = {
            0: 1.0,   # Yellow/Moderate - Mixed forest (clearance ~7.0m)
            1: 0.7,   # Yellow/Moderate - Shrubland (clearance ~7.3m)
            2: 2.5,   # Red/High - Dense forest (clearance ~5.5m)
            3: 0.9,   # Yellow/Moderate - Grassland with trees (clearance ~7.1m)
            4: 2.2,   # Red/High - Sparse forest (clearance ~5.8m)
            5: 2.8,   # Red/High - Oak woodland (clearance ~5.2m)
            6: 3.5,   # Red/High - Eucalyptus (fast-growing, tall) (clearance ~4.5m)
            7: 3.0,   # Red/High - Mixed tall shrubs (clearance ~5.0m)
        }
        return zone_types.get(zone_id, 1.5)
    
    def _get_vegetation_type(self, zone_id):
        """Get vegetation type for a zone"""
        types = {
            0: 'Mixed Oak-Pine Forest',
            1: 'Chaparral Shrubland',
            2: 'Dense Pine Forest',
            3: 'Grassland with Oak',
            4: 'Sparse Mixed Forest',
            5: 'Oak Woodland',
            6: 'Eucalyptus Grove',
            7: 'Tall Manzanita Shrubs'
        }
        return types.get(zone_id, 'Mixed Forest')
    
    def _calculate_density(self, heights):
        """Calculate vegetation density from heights"""
        # Density = percentage of area with vegetation > 0.5m
        return float(np.sum(np.array(heights) > 0.5) / len(heights) * 100)
    
    def _get_spread_rate_for_zone(self, zone_id):
        """Get vegetation spread rate (coverage expansion per week)"""
        # Zones 6 & 7 grow/spread faster
        if zone_id in [6, 7]:
            return np.random.uniform(1.5, 3.0)
        return np.random.uniform(0.5, 1.5)
    
    def _get_spread_direction(self, zone_id, week):
        """Get dominant spread direction (affected by wind, slope)"""
        # Simplified: mostly expands in prevailing wind direction
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        # Some randomness but generally consistent for a zone
        base_direction = (zone_id * 2) % 8
        variation = int(np.random.normal(0, 1))
        direction_idx = (base_direction + variation) % 8
        return directions[direction_idx]
    
    def _get_spread_pattern(self, zone_id):
        """Get overall spread pattern type"""
        patterns = ['uniform', 'clustered', 'linear', 'radial']
        return patterns[zone_id % len(patterns)]
    
    def _get_growth_parameters(self, zone_id):
        """Get growth parameters for a zone"""
        # Zones 6 & 7 have faster growth
        if zone_id == 6:
            return {
                'base_weekly_growth': 0.35,
                'max_height': 25.0,
                'growth_rate_type': 'fast'
            }
        elif zone_id == 7:
            return {
                'base_weekly_growth': 0.25,
                'max_height': 20.0,
                'growth_rate_type': 'variable'
            }
        else:
            return {
                'base_weekly_growth': 0.15,
                'max_height': 15.0,
                'growth_rate_type': 'normal'
            }
    
    def _get_seasonal_factor(self, month):
        """Get seasonal growth factor (spring=high, winter=low)"""
        # Peak growth in spring (March-May)
        if month in [3, 4, 5]:
            return 1.5
        # Good growth in early summer (June-July)
        elif month in [6, 7]:
            return 1.2
        # Slower growth in late summer (Aug-Sept)
        elif month in [8, 9]:
            return 0.8
        # Slow growth in fall (Oct-Nov)
        elif month in [10, 11]:
            return 0.5
        # Minimal growth in winter (Dec-Feb)
        else:
            return 0.3
    
    def save_datasets(self, lidar_data, spread_data, growth_simulation, output_dir='data'):
        """Save all synthetic datasets to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/lidar_canopy_data.json', 'w') as f:
            json.dump(lidar_data, f, indent=2)
        
        with open(f'{output_dir}/vegetation_spread_data.json', 'w') as f:
            json.dump(spread_data, f, indent=2)
        
        with open(f'{output_dir}/growth_simulation_data.json', 'w') as f:
            json.dump(growth_simulation, f, indent=2)
        
        print(f"✅ Datasets saved to '{output_dir}/' directory")
        print(f"   - lidar_canopy_data.json")
        print(f"   - vegetation_spread_data.json")
        print(f"   - growth_simulation_data.json")


def generate_all_datasets(zones, weeks=8, random_seed=42):
    """
    Convenience function to generate all synthetic datasets at once.
    
    Args:
        zones: List of zone dictionaries
        weeks: Number of weeks to simulate
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (lidar_data, spread_data, growth_simulation)
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)
    
    print("🌲 Generating LIDAR canopy height data...")
    lidar_data = generator.generate_lidar_canopy_data(zones, resolution=10)
    
    print("🌿 Generating vegetation spread data...")
    spread_data = generator.generate_vegetation_spread_data(zones, weeks=weeks)
    
    print("📈 Generating realistic growth simulation...")
    growth_simulation = generator.generate_realistic_growth_simulation(
        lidar_data, spread_data, weeks=weeks
    )
    
    print("✅ All datasets generated successfully!")
    
    return lidar_data, spread_data, growth_simulation

