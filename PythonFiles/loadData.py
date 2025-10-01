import database
import pandas as pd
import numpy as np
from sqlalchemy import text
from Pipeline import timeSeriesTrainTestSplit
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

def loadModelReadyData(filepath):
    """Load the engineered features dataset from CSV to dataframe"""
    print(f"Loading Engineered Data")

    df = pd.read_csv(filepath)

    print(f"Loaded {len(df):,} rows.")
    print(f"Columns: {len(df.columns)}")

    return df

def loadFeatureNames(filepath):
    """Load Feature Column Names"""

    with open(filepath, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(features)} Feature Names")
    return features

def prepareDataForTraining(trainDF, valDF, testDF, featureCols, targetCol):
    print("\nPreparing features and target numpy arrays for model")

    X_train = trainDF[featureCols].values
    y_train = trainDF[targetCol].values

    X_val = valDF[featureCols].values
    y_val = valDF[targetCol].values

    X_test = testDF[featureCols].values
    y_test = testDF[targetCol].values   

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test