import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import pyproj
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class GeospatialProcessor:
    """
    Handles geospatial processing operations including distance calculations,
    coordinate transformations, and geometric operations.
    """
    
    def __init__(self):
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.web_mercator = pyproj.CRS("EPSG:3857")
        
    def segment_powerlines(self, powerline_gdf, segment_length=1.0):
        """
        Segment power lines into smaller segments of specified length.
        
        Args:
            powerline_gdf: GeoDataFrame containing power line geometries
            segment_length: Length of each segment in meters
            
        Returns:
            GeoDataFrame: Segmented power lines with additional attributes
        """
        if powerline_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Project to a metric CRS for accurate distance calculations
        powerline_projected = powerline_gdf.to_crs(self.web_mercator)
        
        segmented_lines = []
        segment_id = 0
        
        for idx, row in powerline_projected.iterrows():
            line = row.geometry
            
            if line.geom_type != 'LineString':
                continue
                
            line_length = line.length
            n_segments = max(1, int(np.ceil(line_length / segment_length)))
            
            for i in range(n_segments):
                start_distance = i * segment_length
                end_distance = min((i + 1) * segment_length, line_length)
                
                if start_distance >= line_length:
                    break
                
                # Create segment
                start_point = line.interpolate(start_distance)
                end_point = line.interpolate(end_distance)
                segment_line = LineString([start_point.coords[0], end_point.coords[0]])
                
                # Calculate center point for distance calculations
                center_point = segment_line.centroid
                
                segment_data = {
                    'geometry': segment_line,
                    'segment_id': f"{row.get('line_id', idx)}_{segment_id}",
                    'parent_line_id': row.get('line_id', idx),
                    'segment_number': i,
                    'start_distance': start_distance,
                    'end_distance': end_distance,
                    'segment_length': end_distance - start_distance,
                    'center_lat': None,  # Will be filled after reprojection
                    'center_lon': None,  # Will be filled after reprojection
                    'voltage': row.get('voltage'),
                    'operator': row.get('operator'),
                    'structure_type': row.get('structure_type')
                }
                
                segmented_lines.append(segment_data)
                segment_id += 1
        
        if not segmented_lines:
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame and reproject back to WGS84
        segments_gdf = gpd.GeoDataFrame(segmented_lines, crs=self.web_mercator)
        segments_gdf = segments_gdf.to_crs(self.wgs84)
        
        # Add lat/lon coordinates for center points
        for idx, row in segments_gdf.iterrows():
            center = row.geometry.centroid
            segments_gdf.at[idx, 'center_lat'] = center.y
            segments_gdf.at[idx, 'center_lon'] = center.x
        
        return segments_gdf
    
    def calculate_vegetation_distances(self, powerline_segments, vegetation_points):
        """
        Calculate minimum distances between power line segments and vegetation points.
        
        Args:
            powerline_segments: GeoDataFrame of power line segments
            vegetation_points: List of vegetation point coordinates [(lat, lon), ...]
            
        Returns:
            pandas.DataFrame: Distance matrix and minimum distances
        """
        if powerline_segments.empty or not vegetation_points:
            return pd.DataFrame()
        
        # Project to metric CRS for accurate distance calculations
        segments_proj = powerline_segments.to_crs(self.web_mercator)
        
        # Create vegetation points GeoDataFrame
        veg_gdf = gpd.GeoDataFrame(
            [{'geometry': Point(lon, lat)} for lat, lon in vegetation_points],
            crs=self.wgs84
        ).to_crs(self.web_mercator)
        
        distances = []
        
        for seg_idx, segment in segments_proj.iterrows():
            segment_distances = []
            
            for veg_idx, veg_point in veg_gdf.iterrows():
                # Calculate distance from vegetation point to power line segment
                distance = segment.geometry.distance(veg_point.geometry)
                segment_distances.append(distance)
            
            min_distance = min(segment_distances) if segment_distances else float('inf')
            closest_veg_idx = np.argmin(segment_distances) if segment_distances else -1
            
            distances.append({
                'segment_id': segment['segment_id'],
                'segment_idx': seg_idx,
                'min_distance_m': min_distance,
                'closest_vegetation_idx': closest_veg_idx,
                'center_lat': powerline_segments.iloc[seg_idx]['center_lat'],
                'center_lon': powerline_segments.iloc[seg_idx]['center_lon'],
                'all_distances': segment_distances
            })
        
        return pd.DataFrame(distances)
    
    def identify_critical_segments(self, distance_df, threshold_m=1.0):
        """
        Identify power line segments with vegetation closer than the threshold distance.
        
        Args:
            distance_df: DataFrame from calculate_vegetation_distances
            threshold_m: Minimum safe distance in meters
            
        Returns:
            pandas.DataFrame: Critical segments requiring attention
        """
        if distance_df.empty:
            return pd.DataFrame()
        
        critical_segments = distance_df[distance_df['min_distance_m'] < threshold_m].copy()
        
        if not critical_segments.empty:
            # Add risk categories
            critical_segments['risk_category'] = pd.cut(
                critical_segments['min_distance_m'],
                bins=[0, 0.3, 0.6, threshold_m],
                labels=['Critical', 'High', 'Moderate']
            )
            
            # Sort by distance (most critical first)
            critical_segments = critical_segments.sort_values('min_distance_m')
        
        return critical_segments
    
    def create_vegetation_clusters(self, vegetation_points, eps_m=50, min_samples=5):
        """
        Cluster vegetation points to identify dense vegetation areas.
        
        Args:
            vegetation_points: List of vegetation coordinates [(lat, lon), ...]
            eps_m: Maximum distance between samples in a cluster (meters)
            min_samples: Minimum samples in a cluster
            
        Returns:
            dict: Cluster information and cluster centers
        """
        if len(vegetation_points) < min_samples:
            return {'clusters': [], 'n_clusters': 0}
        
        # Convert to metric coordinates for clustering
        points_gdf = gpd.GeoDataFrame(
            [{'geometry': Point(lon, lat)} for lat, lon in vegetation_points],
            crs=self.wgs84
        ).to_crs(self.web_mercator)
        
        # Extract x, y coordinates
        coords = np.array([(point.geometry.x, point.geometry.y) for _, point in points_gdf.iterrows()])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps_m, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
        
        # Process clusters
        unique_labels = set(labels)
        clusters = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_mask = labels == label
            cluster_coords = coords[cluster_mask]
            
            # Calculate cluster center and bounds
            center_x, center_y = np.mean(cluster_coords, axis=0)
            center_point = Point(center_x, center_y)
            
            # Transform center back to WGS84
            center_gdf = gpd.GeoDataFrame([{'geometry': center_point}], crs=self.web_mercator)
            center_wgs84 = center_gdf.to_crs(self.wgs84)
            center_lat, center_lon = center_wgs84.geometry.iloc[0].y, center_wgs84.geometry.iloc[0].x
            
            clusters.append({
                'cluster_id': int(label),
                'center_lat': center_lat,
                'center_lon': center_lon,
                'point_count': np.sum(cluster_mask),
                'radius_m': np.max(cdist([coords[cluster_mask].mean(axis=0)], cluster_coords)[0])
            })
        
        return {
            'clusters': clusters,
            'n_clusters': len(clusters),
            'noise_points': np.sum(labels == -1),
            'cluster_labels': labels.tolist()
        }
    
    def calculate_clearance_zones(self, powerline_segments, clearance_distance=10.0):
        """
        Create clearance zones around power line segments.
        
        Args:
            powerline_segments: GeoDataFrame of power line segments
            clearance_distance: Buffer distance in meters
            
        Returns:
            GeoDataFrame: Clearance zone polygons
        """
        if powerline_segments.empty:
            return gpd.GeoDataFrame()
        
        # Project to metric CRS
        segments_proj = powerline_segments.to_crs(self.web_mercator)
        
        # Create buffer zones
        clearance_zones = []
        
        for idx, segment in segments_proj.iterrows():
            buffer_geom = segment.geometry.buffer(clearance_distance)
            
            clearance_zones.append({
                'geometry': buffer_geom,
                'segment_id': segment['segment_id'],
                'clearance_distance': clearance_distance,
                'voltage': segment.get('voltage'),
                'operator': segment.get('operator')
            })
        
        zones_gdf = gpd.GeoDataFrame(clearance_zones, crs=self.web_mercator)
        return zones_gdf.to_crs(self.wgs84)
    
    def intersect_vegetation_clearance(self, vegetation_points, clearance_zones):
        """
        Find vegetation points that intersect with clearance zones.
        
        Args:
            vegetation_points: List of vegetation coordinates
            clearance_zones: GeoDataFrame of clearance zone polygons
            
        Returns:
            pandas.DataFrame: Intersection results
        """
        if not vegetation_points or clearance_zones.empty:
            return pd.DataFrame()
        
        # Create vegetation points GeoDataFrame
        veg_gdf = gpd.GeoDataFrame(
            [{'geometry': Point(lon, lat), 'veg_id': i} 
             for i, (lat, lon) in enumerate(vegetation_points)],
            crs=self.wgs84
        )
        
        # Perform spatial join
        intersections = gpd.sjoin(
            veg_gdf, 
            clearance_zones, 
            how='inner', 
            predicate='intersects'
        )
        
        return intersections
    
    def calculate_line_of_sight(self, powerline_segments, vegetation_heights, terrain_elevation=None):
        """
        Calculate line-of-sight between power lines and vegetation considering height.
        
        Args:
            powerline_segments: GeoDataFrame of power line segments
            vegetation_heights: Dictionary mapping coordinates to vegetation heights
            terrain_elevation: Optional terrain elevation data
            
        Returns:
            dict: Line-of-sight analysis results
        """
        # Simplified line-of-sight calculation
        # In a full implementation, this would use LiDAR DTM and DSM data
        
        results = {
            'blocked_segments': [],
            'sight_line_analysis': [],
            'height_clearances': []
        }
        
        # Assume power line height (typical transmission line height)
        powerline_height = 30.0  # meters above ground
        
        for idx, segment in powerline_segments.iterrows():
            segment_lat = segment['center_lat']
            segment_lon = segment['center_lon']
            
            # Find nearby vegetation
            nearby_vegetation = []
            for (veg_lat, veg_lon), veg_height in vegetation_heights.items():
                # Calculate approximate distance
                distance = np.sqrt((segment_lat - veg_lat)**2 + (segment_lon - veg_lon)**2) * 111000  # rough conversion
                
                if distance < 100:  # Within 100m
                    height_clearance = powerline_height - veg_height
                    nearby_vegetation.append({
                        'distance': distance,
                        'vegetation_height': veg_height,
                        'height_clearance': height_clearance,
                        'is_blocked': height_clearance < 5.0  # 5m minimum clearance
                    })
            
            if nearby_vegetation:
                results['height_clearances'].append({
                    'segment_id': segment['segment_id'],
                    'center_lat': segment_lat,
                    'center_lon': segment_lon,
                    'vegetation_count': len(nearby_vegetation),
                    'min_clearance': min([v['height_clearance'] for v in nearby_vegetation]),
                    'blocked': any([v['is_blocked'] for v in nearby_vegetation])
                })
        
        return results
    
    def transform_coordinates(self, points, from_crs, to_crs):
        """
        Transform coordinates between different coordinate reference systems.
        
        Args:
            points: List of coordinate tuples
            from_crs: Source CRS (e.g., 'EPSG:4326')
            to_crs: Target CRS (e.g., 'EPSG:3857')
            
        Returns:
            list: Transformed coordinates
        """
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        transformed_points = []
        
        for point in points:
            if len(point) == 2:
                x, y = transformer.transform(point[0], point[1])
                transformed_points.append((x, y))
            else:
                transformed_points.append(point)
        
        return transformed_points
