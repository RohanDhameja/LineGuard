import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import json
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

class DatabaseManager:
    """
    Handles database operations for the vegetation detection system.
    Stores analysis results, historical data, and monitoring information.
    """
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self._create_tables()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        create_tables_sql = """
        -- Analysis sessions table
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            center_lat DOUBLE PRECISION NOT NULL,
            center_lon DOUBLE PRECISION NOT NULL,
            radius_km DOUBLE PRECISION NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parameters JSONB,
            overall_risk_score DOUBLE PRECISION,
            high_risk_count INTEGER,
            vegetation_density DOUBLE PRECISION,
            avg_clearance_distance DOUBLE PRECISION,
            data_sources JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Risk areas table
        CREATE TABLE IF NOT EXISTS risk_areas (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) REFERENCES analysis_sessions(session_id),
            lat DOUBLE PRECISION NOT NULL,
            lon DOUBLE PRECISION NOT NULL,
            risk_score DOUBLE PRECISION NOT NULL,
            risk_category VARCHAR(20),
            clearance_distance DOUBLE PRECISION,
            vegetation_height DOUBLE PRECISION,
            ndvi_value DOUBLE PRECISION,
            action_required TEXT,
            priority_rank INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Power line segments table
        CREATE TABLE IF NOT EXISTS powerline_segments (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) REFERENCES analysis_sessions(session_id),
            segment_id VARCHAR(100),
            parent_line_id VARCHAR(100),
            center_lat DOUBLE PRECISION NOT NULL,
            center_lon DOUBLE PRECISION NOT NULL,
            voltage INTEGER,
            operator VARCHAR(100),
            structure_type VARCHAR(50),
            min_vegetation_distance DOUBLE PRECISION,
            is_critical BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Vegetation monitoring table
        CREATE TABLE IF NOT EXISTS vegetation_monitoring (
            id SERIAL PRIMARY KEY,
            lat DOUBLE PRECISION NOT NULL,
            lon DOUBLE PRECISION NOT NULL,
            ndvi_value DOUBLE PRECISION,
            vegetation_height DOUBLE PRECISION,
            measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_source VARCHAR(50),
            quality_score DOUBLE PRECISION
        );
        
        -- Historical trends table
        CREATE TABLE IF NOT EXISTS historical_trends (
            id SERIAL PRIMARY KEY,
            region_id VARCHAR(100),
            analysis_date DATE,
            avg_risk_score DOUBLE PRECISION,
            vegetation_density DOUBLE PRECISION,
            avg_clearance DOUBLE PRECISION,
            incident_count INTEGER DEFAULT 0,
            weather_conditions JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(region_id, analysis_date)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_analysis_sessions_date ON analysis_sessions(analysis_date);
        CREATE INDEX IF NOT EXISTS idx_risk_areas_session ON risk_areas(session_id);
        CREATE INDEX IF NOT EXISTS idx_risk_areas_location ON risk_areas(lat, lon);
        CREATE INDEX IF NOT EXISTS idx_powerline_segments_session ON powerline_segments(session_id);
        CREATE INDEX IF NOT EXISTS idx_vegetation_monitoring_location ON vegetation_monitoring(lat, lon);
        CREATE INDEX IF NOT EXISTS idx_vegetation_monitoring_date ON vegetation_monitoring(measurement_date);
        CREATE INDEX IF NOT EXISTS idx_historical_trends_date ON historical_trends(analysis_date);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_tables_sql)
                    conn.commit()
                    print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Error creating database tables: {str(e)}")
            raise
    
    def save_analysis_session(self, session_data):
        """Save analysis session results to database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert analysis session
                    insert_session_sql = """
                    INSERT INTO analysis_sessions 
                    (session_id, center_lat, center_lon, radius_km, parameters, 
                     overall_risk_score, high_risk_count, vegetation_density, 
                     avg_clearance_distance, data_sources)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        overall_risk_score = EXCLUDED.overall_risk_score,
                        high_risk_count = EXCLUDED.high_risk_count,
                        vegetation_density = EXCLUDED.vegetation_density,
                        avg_clearance_distance = EXCLUDED.avg_clearance_distance,
                        analysis_date = CURRENT_TIMESTAMP
                    """
                    
                    cursor.execute(insert_session_sql, (
                        session_data['session_id'],
                        session_data['center_lat'],
                        session_data['center_lon'],
                        session_data['radius_km'],
                        json.dumps(session_data.get('parameters', {})),
                        session_data.get('overall_risk_score'),
                        session_data.get('high_risk_count'),
                        session_data.get('vegetation_density'),
                        session_data.get('avg_clearance_distance'),
                        json.dumps(session_data.get('data_sources', {}))
                    ))
                    
                    # Save risk areas
                    if 'priority_areas' in session_data and session_data['priority_areas']:
                        self._save_risk_areas(cursor, session_data['session_id'], session_data['priority_areas'])
                    
                    conn.commit()
                    print(f"✅ Analysis session {session_data['session_id']} saved to database")
                    
        except Exception as e:
            print(f"❌ Error saving analysis session: {str(e)}")
            raise
    
    def _save_risk_areas(self, cursor, session_id, risk_areas):
        """Save risk areas for a session."""
        # Clear existing risk areas for this session
        cursor.execute("DELETE FROM risk_areas WHERE session_id = %s", (session_id,))
        
        # Insert new risk areas
        insert_risk_sql = """
        INSERT INTO risk_areas 
        (session_id, lat, lon, risk_score, risk_category, clearance_distance, 
         vegetation_height, ndvi_value, action_required, priority_rank)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for area in risk_areas:
            cursor.execute(insert_risk_sql, (
                session_id,
                area['lat'],
                area['lon'],
                area['risk_score'],
                area.get('risk_category'),
                area.get('clearance_distance'),
                area.get('vegetation_height'),
                area.get('ndvi'),
                area.get('action_required'),
                area.get('priority_rank')
            ))
    
    def save_powerline_segments(self, session_id, segments_data):
        """Save power line segments analysis results."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Clear existing segments for this session
                    cursor.execute("DELETE FROM powerline_segments WHERE session_id = %s", (session_id,))
                    
                    # Insert new segments
                    insert_segment_sql = """
                    INSERT INTO powerline_segments 
                    (session_id, segment_id, parent_line_id, center_lat, center_lon, 
                     voltage, operator, structure_type, min_vegetation_distance, is_critical)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    for _, segment in segments_data.iterrows():
                        cursor.execute(insert_segment_sql, (
                            session_id,
                            segment.get('segment_id'),
                            segment.get('parent_line_id'),
                            segment.get('center_lat'),
                            segment.get('center_lon'),
                            segment.get('voltage'),
                            segment.get('operator'),
                            segment.get('structure_type'),
                            segment.get('min_vegetation_distance'),
                            segment.get('min_vegetation_distance', 100) < 1.0  # Critical if < 1m
                        ))
                    
                    conn.commit()
                    print(f"✅ Power line segments saved for session {session_id}")
                    
        except Exception as e:
            print(f"❌ Error saving power line segments: {str(e)}")
            raise
    
    def save_vegetation_monitoring(self, monitoring_data):
        """Save vegetation monitoring data points."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    insert_monitoring_sql = """
                    INSERT INTO vegetation_monitoring 
                    (lat, lon, ndvi_value, vegetation_height, data_source, quality_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    
                    for point in monitoring_data:
                        cursor.execute(insert_monitoring_sql, (
                            point['lat'],
                            point['lon'],
                            point.get('ndvi_value'),
                            point.get('vegetation_height'),
                            point.get('data_source', 'unknown'),
                            point.get('quality_score', 1.0)
                        ))
                    
                    conn.commit()
                    print(f"✅ Vegetation monitoring data saved ({len(monitoring_data)} points)")
                    
        except Exception as e:
            print(f"❌ Error saving vegetation monitoring: {str(e)}")
            raise
    
    def get_historical_analysis(self, lat, lon, radius_km, days_back=30):
        """Get historical analysis data for trend analysis."""
        try:
            with self.get_connection() as conn:
                query_sql = """
                SELECT 
                    analysis_date::date as date,
                    AVG(overall_risk_score) as avg_risk_score,
                    AVG(vegetation_density) as vegetation_density,
                    AVG(avg_clearance_distance) as avg_clearance,
                    COUNT(*) as analysis_count
                FROM analysis_sessions
                WHERE analysis_date >= CURRENT_DATE - (%s * INTERVAL '1 day')
                AND (
                    6371 * 2 * asin(sqrt(
                        power(sin((radians(center_lat) - radians(%s)) / 2), 2) +
                        cos(radians(%s)) * cos(radians(center_lat)) *
                        power(sin((radians(center_lon) - radians(%s)) / 2), 2)
                    ))
                ) <= %s
                GROUP BY analysis_date::date
                ORDER BY date DESC
                """
                
                df = pd.read_sql_query(query_sql, conn, params=(
                    days_back, lat, lat, lon, radius_km
                ))
                
                return df
                
        except Exception as e:
            print(f"❌ Error getting historical analysis: {str(e)}")
            # Return empty DataFrame if query fails
            return pd.DataFrame(columns=['date', 'avg_risk_score', 'vegetation_density', 'avg_clearance', 'analysis_count'])
    
    def get_risk_areas_in_region(self, lat, lon, radius_km, min_risk_score=0.6):
        """Get high-risk areas within a geographic region."""
        try:
            with self.get_connection() as conn:
                query_sql = """
                SELECT ra.*, ras.analysis_date
                FROM risk_areas ra
                JOIN analysis_sessions ras ON ra.session_id = ras.session_id
                WHERE ra.risk_score >= %s
                AND (
                    6371 * 2 * asin(sqrt(
                        power(sin((radians(ra.lat) - radians(%s)) / 2), 2) +
                        cos(radians(%s)) * cos(radians(ra.lat)) *
                        power(sin((radians(ra.lon) - radians(%s)) / 2), 2)
                    ))
                ) <= %s
                ORDER BY ra.risk_score DESC, ras.analysis_date DESC
                LIMIT 100
                """
                
                df = pd.read_sql_query(query_sql, conn, params=(
                    min_risk_score, lat, lat, lon, radius_km
                ))
                
                return df
                
        except Exception as e:
            print(f"❌ Error getting risk areas: {str(e)}")
            return pd.DataFrame()
    
    def update_historical_trends(self, region_id, date, trend_data):
        """Update historical trends data for a region."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    insert_trend_sql = """
                    INSERT INTO historical_trends 
                    (region_id, analysis_date, avg_risk_score, vegetation_density, 
                     avg_clearance, incident_count, weather_conditions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (region_id, analysis_date) DO UPDATE SET
                        avg_risk_score = EXCLUDED.avg_risk_score,
                        vegetation_density = EXCLUDED.vegetation_density,
                        avg_clearance = EXCLUDED.avg_clearance,
                        incident_count = EXCLUDED.incident_count,
                        weather_conditions = EXCLUDED.weather_conditions
                    """
                    
                    cursor.execute(insert_trend_sql, (
                        region_id,
                        date,
                        trend_data.get('avg_risk_score'),
                        trend_data.get('vegetation_density'),
                        trend_data.get('avg_clearance'),
                        trend_data.get('incident_count', 0),
                        json.dumps(trend_data.get('weather_conditions', {}))
                    ))
                    
                    conn.commit()
                    print(f"✅ Historical trends updated for region {region_id}")
                    
        except Exception as e:
            print(f"❌ Error updating historical trends: {str(e)}")
            raise
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old analysis data to manage database size."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Clean up old analysis sessions and related data
                    # Clean up old risk areas
                    cursor.execute("""
                        DELETE FROM risk_areas 
                        WHERE session_id IN (
                            SELECT session_id FROM analysis_sessions 
                            WHERE analysis_date < CURRENT_DATE - (%s * INTERVAL '1 day')
                        )
                    """, (days_to_keep,))
                    
                    # Clean up old powerline segments
                    cursor.execute("""
                        DELETE FROM powerline_segments 
                        WHERE session_id IN (
                            SELECT session_id FROM analysis_sessions 
                            WHERE analysis_date < CURRENT_DATE - (%s * INTERVAL '1 day')
                        )
                    """, (days_to_keep,))
                    
                    # Clean up old analysis sessions
                    cursor.execute("""
                        DELETE FROM analysis_sessions 
                        WHERE analysis_date < CURRENT_DATE - (%s * INTERVAL '1 day')
                    """, (days_to_keep,))
                    
                    # Clean up old vegetation monitoring
                    cursor.execute("""
                        DELETE FROM vegetation_monitoring 
                        WHERE measurement_date < CURRENT_DATE - (%s * INTERVAL '1 day')
                    """, (days_to_keep,))
                    conn.commit()
                    print(f"✅ Cleaned up data older than {days_to_keep} days")
                    
        except Exception as e:
            print(f"❌ Error cleaning up old data: {str(e)}")