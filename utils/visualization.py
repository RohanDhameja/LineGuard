import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
import base64
import json
from datetime import datetime

class Visualizer:
    """
    Handles data visualization and report generation for the vegetation detection system.
    """
    
    def __init__(self):
        self.color_scales = {
            'risk': ['green', 'yellow', 'orange', 'red'],
            'vegetation': ['brown', 'yellow', 'lightgreen', 'darkgreen'],
            'height': ['lightblue', 'blue', 'darkblue']
        }
    
    def create_risk_heatmap(self, risk_data):
        """
        Create a heatmap visualization of fire risk across the analysis area.
        
        Args:
            risk_data: DataFrame with lat, lon, and risk_score columns
            
        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        if risk_data.empty:
            return self._create_empty_plot("No risk data available")
        
        fig = go.Figure(data=go.Densitymapbox(
            lat=risk_data['lat'],
            lon=risk_data['lon'],
            z=risk_data['risk_score'],
            radius=30,
            colorscale='Reds',
            zmin=0,
            zmax=1,
            hovertemplate='<b>Risk Score</b>: %{z:.3f}<br>' +
                         '<b>Location</b>: %{lat:.6f}, %{lon:.6f}<br>' +
                         '<extra></extra>'
        ))
        
        # Calculate center for map
        center_lat = risk_data['lat'].mean()
        center_lon = risk_data['lon'].mean()
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            title="Fire Risk Heatmap",
            height=600
        )
        
        return fig
    
    def create_vegetation_density_map(self, vegetation_data):
        """
        Create vegetation density visualization using NDVI data.
        
        Args:
            vegetation_data: DataFrame with vegetation metrics
            
        Returns:
            plotly.graph_objects.Figure: Vegetation density map
        """
        if vegetation_data.empty:
            return self._create_empty_plot("No vegetation data available")
        
        fig = px.scatter_mapbox(
            vegetation_data,
            lat="lat",
            lon="lon",
            color="ndvi_mean",
            size="vegetation_density",
            color_continuous_scale="Greens",
            size_max=15,
            zoom=12,
            mapbox_style="open-street-map",
            title="Vegetation Density (NDVI)",
            hover_data={
                'ndvi_mean': ':.3f',
                'vegetation_density': ':.2f',
                'avg_vegetation_height': ':.1f'
            }
        )
        
        fig.update_layout(height=600)
        return fig
    
    def create_clearance_analysis_chart(self, clearance_data):
        """
        Create visualization showing power line clearance distances.
        
        Args:
            clearance_data: DataFrame with clearance distance information
            
        Returns:
            plotly.graph_objects.Figure: Clearance analysis chart
        """
        if clearance_data.empty:
            return self._create_empty_plot("No clearance data available")
        
        # Create histogram of clearance distances
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distance Distribution', 'Risk Categories', 
                          'Geographic Distribution', 'Critical Segments'),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Distance distribution histogram
        fig.add_trace(
            go.Histogram(
                x=clearance_data['min_distance_to_powerline'],
                nbinsx=20,
                name='Distance Distribution',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Risk categories pie chart
        risk_categories = clearance_data['risk_score'].apply(self._categorize_risk_level)
        risk_counts = risk_categories.value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                name='Risk Categories'
            ),
            row=1, col=2
        )
        
        # Geographic scatter
        fig.add_trace(
            go.Scatter(
                x=clearance_data['lon'],
                y=clearance_data['lat'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=clearance_data['risk_score'],
                    colorscale='Reds',
                    showscale=True
                ),
                name='Risk Locations'
            ),
            row=2, col=1
        )
        
        # Critical segments bar chart
        critical_segments = clearance_data[clearance_data['min_distance_to_powerline'] < 2.0]
        if not critical_segments.empty:
            distance_bins = pd.cut(critical_segments['min_distance_to_powerline'], 
                                 bins=[0, 0.5, 1.0, 1.5, 2.0])
            bin_counts = distance_bins.value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=[str(bin) for bin in bin_counts.index],
                    y=bin_counts.values,
                    name='Critical Distance Bins',
                    marker_color='red'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Power Line Clearance Analysis",
            showlegend=False
        )
        
        return fig
    
    def create_priority_areas_table(self, priority_areas):
        """
        Create an interactive table of priority areas.
        
        Args:
            priority_areas: List of priority area dictionaries
            
        Returns:
            plotly.graph_objects.Figure: Interactive table
        """
        if not priority_areas:
            return self._create_empty_plot("No priority areas identified")
        
        df = pd.DataFrame(priority_areas)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Rank', 'Location', 'Risk Score', 'Distance (m)', 
                       'Height (m)', 'Category', 'Action Required'],
                fill_color='paleturquoise',
                align='center',
                font=dict(size=12)
            ),
            cells=dict(
                values=[
                    df['priority_rank'],
                    [f"{lat:.4f}, {lon:.4f}" for lat, lon in zip(df['lat'], df['lon'])],
                    [f"{score:.3f}" for score in df['risk_score']],
                    [f"{dist:.1f}" for dist in df['clearance_distance']],
                    [f"{height:.1f}" for height in df['vegetation_height']],
                    df['risk_category'],
                    df['action_required']
                ],
                fill_color=[
                    ['white' if rank <= 5 else 'lightgray' for rank in df['priority_rank']],
                    'white',
                    [self._get_risk_color(score) for score in df['risk_score']],
                    'white',
                    'white',
                    'white',
                    'white'
                ],
                align='center',
                font=dict(size=10)
            )
        )])
        
        fig.update_layout(
            title="Priority Areas for Vegetation Management",
            height=600
        )
        
        return fig
    
    def create_trend_analysis(self, historical_data):
        """
        Create trend analysis visualization (if historical data is available).
        
        Args:
            historical_data: DataFrame with historical analysis results
            
        Returns:
            plotly.graph_objects.Figure: Trend analysis chart
        """
        if historical_data is None or historical_data.empty:
            return self._create_empty_plot("No historical data available for trend analysis")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Trend', 'Vegetation Density', 
                          'Clearance Distance', 'Incident Frequency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add trend lines
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['avg_risk_score'],
                mode='lines+markers',
                name='Average Risk Score',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['vegetation_density'],
                mode='lines+markers',
                name='Vegetation Density',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['avg_clearance'],
                mode='lines+markers',
                name='Average Clearance',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=historical_data['date'],
                y=historical_data.get('incident_count', [0] * len(historical_data)),
                name='Incidents',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Historical Trend Analysis",
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, analysis_results, format='pdf', include_map=True, 
                       include_stats=True, include_recommendations=True):
        """
        Generate comprehensive analysis report in various formats.
        
        Args:
            analysis_results: Complete analysis results dictionary
            format: Output format ('pdf', 'csv', 'geojson', 'html')
            include_map: Whether to include map visualizations
            include_stats: Whether to include statistical summaries
            include_recommendations: Whether to include recommendations
            
        Returns:
            dict: Report content and metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            return self._generate_csv_report(analysis_results, timestamp)
        elif format.lower() == 'geojson':
            return self._generate_geojson_report(analysis_results, timestamp)
        elif format.lower() == 'html':
            return self._generate_html_report(
                analysis_results, timestamp, include_map, include_stats, include_recommendations
            )
        else:  # Default to JSON summary
            return self._generate_json_report(analysis_results, timestamp)
    
    def _generate_csv_report(self, results, timestamp):
        """Generate CSV report with priority areas and statistics."""
        # Create priority areas CSV
        if results.get('priority_areas'):
            df = pd.DataFrame(results['priority_areas'])
            csv_content = df.to_csv(index=False)
        else:
            csv_content = "No priority areas identified\n"
        
        # Add summary statistics
        stats_section = "\n\n# SUMMARY STATISTICS\n"
        stats_section += f"Analysis Date,{results.get('analysis_timestamp', 'Unknown')}\n"
        stats_section += f"Overall Risk Score,{results.get('overall_risk', 0):.3f}\n"
        stats_section += f"High Risk Areas,{results.get('high_risk_count', 0)}\n"
        stats_section += f"Vegetation Density,{results.get('vegetation_density', 0):.2%}\n"
        stats_section += f"Average Clearance Distance,{results.get('avg_clearance', 0):.1f}m\n"
        
        csv_content += stats_section
        
        return {
            'content': csv_content,
            'filename': f'vegetation_risk_analysis_{timestamp}.csv',
            'mime_type': 'text/csv'
        }
    
    def _generate_geojson_report(self, results, timestamp):
        """Generate GeoJSON report with spatial data."""
        features = []
        
        # Add priority areas as point features
        for area in results.get('priority_areas', []):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [area['lon'], area['lat']]
                },
                "properties": {
                    "risk_score": area['risk_score'],
                    "clearance_distance": area['clearance_distance'],
                    "vegetation_height": area['vegetation_height'],
                    "risk_category": area['risk_category'],
                    "action_required": area['action_required'],
                    "priority_rank": area['priority_rank']
                }
            }
            features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "metadata": {
                "analysis_date": str(results.get('analysis_timestamp', 'Unknown')),
                "overall_risk": results.get('overall_risk', 0),
                "total_features": len(features)
            },
            "features": features
        }
        
        return {
            'content': json.dumps(geojson_data, indent=2),
            'filename': f'vegetation_risk_analysis_{timestamp}.geojson',
            'mime_type': 'application/json'
        }
    
    def _generate_html_report(self, results, timestamp, include_map, include_stats, include_recommendations):
        """Generate comprehensive HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vegetation Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e6f3ff; border-radius: 5px; min-width: 150px; }}
                .priority-area {{ padding: 10px; margin: 5px 0; border-left: 4px solid #ff6b6b; 
                                background-color: #fff5f5; }}
                .recommendation {{ padding: 10px; margin: 5px 0; border-left: 4px solid #51cf66; 
                                 background-color: #f3fff3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔥 Vegetation Detection Near Power Lines - Analysis Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Date:</strong> {results.get('analysis_timestamp', 'Unknown')}</p>
            </div>
        """
        
        if include_stats:
            html_content += f"""
            <div class="section">
                <h2>📊 Key Metrics</h2>
                <div class="metric">
                    <h3>{results.get('overall_risk', 0):.3f}</h3>
                    <p>Overall Risk Score</p>
                </div>
                <div class="metric">
                    <h3>{results.get('high_risk_count', 0)}</h3>
                    <p>High Risk Areas</p>
                </div>
                <div class="metric">
                    <h3>{results.get('vegetation_density', 0):.1%}</h3>
                    <p>Vegetation Density</p>
                </div>
                <div class="metric">
                    <h3>{results.get('avg_clearance', 0):.1f}m</h3>
                    <p>Avg Clearance Distance</p>
                </div>
            </div>
            """
        
        # Priority areas section
        html_content += """
        <div class="section">
            <h2>🚨 Priority Areas</h2>
        """
        
        for area in results.get('priority_areas', [])[:10]:  # Top 10
            html_content += f"""
            <div class="priority-area">
                <strong>Rank {area['priority_rank']}</strong> - 
                Location: {area['lat']:.4f}, {area['lon']:.4f}<br>
                Risk Score: {area['risk_score']:.3f} | 
                Distance: {area['clearance_distance']:.1f}m | 
                Category: {area['risk_category']}<br>
                <em>{area['action_required']}</em>
            </div>
            """
        
        if include_recommendations:
            html_content += """
            <div class="section">
                <h2>💡 Recommendations</h2>
            """
            
            for rec in results.get('recommendations', []):
                html_content += f"""
                <div class="recommendation">{rec}</div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        return {
            'content': html_content,
            'filename': f'vegetation_risk_report_{timestamp}.html',
            'mime_type': 'text/html'
        }
    
    def _generate_json_report(self, results, timestamp):
        """Generate JSON summary report."""
        json_content = json.dumps(results, indent=2, default=str)
        
        return {
            'content': json_content,
            'filename': f'vegetation_analysis_{timestamp}.json',
            'mime_type': 'application/json'
        }
    
    def _categorize_risk_level(self, risk_score):
        """Categorize risk score into levels."""
        if risk_score >= 0.8:
            return 'Critical'
        elif risk_score >= 0.6:
            return 'High'
        elif risk_score >= 0.4:
            return 'Moderate'
        else:
            return 'Low'
    
    def _get_risk_color(self, risk_score):
        """Get color based on risk score."""
        if risk_score >= 0.8:
            return 'lightcoral'
        elif risk_score >= 0.6:
            return 'lightsalmon'
        elif risk_score >= 0.4:
            return 'khaki'
        else:
            return 'lightgreen'
    
    def _create_empty_plot(self, message):
        """Create an empty plot with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
