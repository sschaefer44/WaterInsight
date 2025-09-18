import psycopg2
from psycopg2.extras import execute_values

def getConnection():
    """Create database connection"""
    return psycopg2.connect(
        host='localhost',
        database='waterinsight',
        user='postgres',      # Replace with your PostgreSQL username
        password='pgDB'   # Replace with your PostgreSQL password
    )

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
    """Query water data for demonstration"""
    conn = getConnection()
    cur = conn.cursor()
    
    cur.execute('''
        SELECT * FROM waterdata 
        WHERE site_code = %s AND date BETWEEN %s AND %s
        ORDER BY date
    ''', (site_code, start_date, end_date))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return results