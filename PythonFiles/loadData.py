import database
import pandas as pd
import numpy as np
from sqlalchemy import text

def loadData():
    """Load data from postgreSQL into dataframe"""

    engine = database.getSQLAlchemyEngine()

    query = """
        SELECT
            w.site_code,
            w.date,
            w.discharge,
            w.gage_height,
            w.stream_elevation,
            w.temperature,
            w.dissolved_oxygen,
            l.latitude,
            l.longitude,
            l.state
        FROM waterdata w
        LEFT JOIN sitelocations l on w.site_code = l.site_code
        WHERE w.discharge IS NOT NULL
        AND w.gage_height IS NOT NULL
        ORDER BY w.site_code, w.date
    """

    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df):,} records")
    print(f"Sites with location data: {df['latitude'].notna().sum() / len(df) * 100:.1f}%")

    return df