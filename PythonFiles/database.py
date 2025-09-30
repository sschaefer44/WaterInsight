import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import pandas as pd

DB_CONFIG = {
    'host': 'localhost',
    'database': 'waterinsight',
    'user': 'postgres',  
    'password': 'pgDB' 
}

def getConnection():
    """Create psychopg2 connection for inserts """
    return psycopg2.connect(**DB_CONFIG)

def getSQLAlchemyEngine():
    """Create SQLAlchemy engine for pandas operations"""
    connString = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    return create_engine(connString)

def testConnection():
    """Test if database connection works"""
    try:
        conn = getConnection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"Connected to PostgreSQL: {version[0]}")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def insertSiteLocation(site_info, state_code):
    """Insert site location data"""
    conn = getConnection()
    cur = conn.cursor()
    
    cur.execute('''
        INSERT INTO sitelocations (site_code, site_name, latitude, longitude, state) 
        VALUES (%s, %s, %s, %s, %s) ON CONFLICT (site_code) DO NOTHING
    ''', (site_info['siteCode'], site_info['siteName'], 
          site_info['latitude'], site_info['longitude'], state_code))
    
    conn.commit()
    cur.close()
    conn.close()

def insertWaterData(water_records):
    """Insert water measurement data"""
    conn = getConnection()
    cur = conn.cursor()
    
    values = [
        (record['site_code'], record['date'], record['discharge'],
         record['gage_height'], record['stream_elevation'], 
         record['temperature'], record['dissolved_oxygen'])
        for record in water_records
    ]
    
    execute_values(
        cur,
        '''INSERT INTO waterdata (site_code, date, discharge, gage_height, 
           stream_elevation, temperature, dissolved_oxygen) 
           VALUES %s ON CONFLICT (site_code, date) DO NOTHING''',
        values
    )
    
    conn.commit()
    cur.close() 
    conn.close()
    
    return len(values)

def queryWaterData(site_code, start_date, end_date):
    """Query water data for demonstration using SQLAlchemy"""
    engine = getSQLAlchemyEngine()
    
    query = '''
        SELECT * FROM waterdata 
        WHERE site_code = %s AND date BETWEEN %s AND %s
        ORDER BY date
    '''
    
    results = pd.read_sql(query, engine, params=[site_code, start_date, end_date])
    return results