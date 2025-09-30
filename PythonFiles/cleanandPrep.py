import database
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import text
import time

def dataQualityCheck():
    """Checks data quality. Identify issues before data cleaning process"""

    engine = database.getSQLAlchemyEngine()

    print("----------Checking Data Quality----------")

    # ImpossibleQuery checks for impossible values: E.g. negative flow rate, negative gage height, ....
    impossibleQuery = """
        SELECT
            COUNT(CASE WHEN discharge < 0 THEN 1 END) as negativeDischarge,
            COUNT(CASE WHEN discharge < 0 THEN 1 END) as negativegage,
            COUNT(CASE WHEN temperature < -50 OR temperature > 50 THEN 1 END) as extremeTemp,
            COUNT(CASE WHEN dissolved_oxygen < 0 OR dissolved_oxygen > 50 THEN 1 END ) as extremeDO,
            MAX(discharge) as maxDischarge,
            MIN(discharge) as minDischarge,
            MAX(gage_height) as maxgageHeight,
            MIN(gage_height) as mingageHeight
        FROM waterdata
        WHERE discharge != -999 OR gage_height != -999 
        """
    
    qualityStats = pd.read_sql(impossibleQuery, engine)
    print("----------Potential Data Quality Problems:")
    print(qualityStats)

    duplicateCheckQuery = """
            SELECT COUNT(*) - COUNT(DISTINCT (site_code, date)) as dupeRecords
        """
    duplicates = pd.read_sql(duplicateCheckQuery, engine)
    print(f"\nDuplicate Records: {duplicates.iloc[0, 0]}")

    return qualityStats, duplicates

def handleMissingVals():
    engine = database.getSQLAlchemyEngine()
    
    missingPatternQuery = """
        SELECT
        site_code,
        COUNT(*) as totalRecords,
        COUNT(CASE WHEN discharge = -999 THEN 1 END) as missingDischarge,
        COUNT(CASE WHEN gage_height = -999 THEN 1 END) as missinggage,
        COUNT(CASE WHEN temperature = -999 THEN 1 END) as missingTemp,
        COUNT(CASE WHEN dissolved_oxygen = -999 THEN 1 END) as missingDO,
        ROUND(100.0 * COUNT(CASE WHEN discharge = -999 THEN 1 END)/ COUNT(*), 2) as missingPctDO
        FROM waterdata
        GROUP BY site_code
        HAVING COUNT(*) > 100
        ORDER BY missingPCTDO DESC
        LIMIT 20
    """

    missingPatterns = pd.read_sql(missingPatternQuery, engine)
    print("\n---------- MISSING VALUE PATTERNS (TOP 20 SITES BY MISSING DISCHARGE)")
    print(missingPatterns)
    return missingPatterns

def detectOutliers(parameters, threshold, limit):
    """Detect outliers for one or more parameters using statistical methods"""
    engine = database.getSQLAlchemyEngine()

    # Validate Parameters
    validParameters = ['discharge', 'gage_height', 'stream_elevation', 'temperature', 'dissolved_oxygen']
    parameters = [param for param in parameters if param in validParameters]

    if not parameters:
        print("No Valid Parameters Provided.")
        return {}
    
    results = {}

    for parameter in parameters:
        outlierQuery = f"""
            WITH stats AS (
                SELECT
                    AVG({parameter}) as meanVal,
                    STDDEV({parameter}) as stdVal
                FROM waterdata
                WHERE {parameter} != -999
                )
                SELECT 
                    site_code, date, {parameter},
                    ROUND(CAST(ABS({parameter} - meanVal) / stdVal AS NUMERIC), 2) as stdDevsFromMean
                FROM waterdata, stats
                WHERE {parameter} != -999
                AND ABS({parameter} - meanVal) > {threshold} * stdVal
                ORDER BY ABS({parameter} - meanVal) DESC
                LIMIT {limit}
    """
    outliers = pd.read_sql(outlierQuery, engine)
    results[parameter] = outliers

    print(f"\n---------- OUTLIERS FOR {parameter.upper()} (>{threshold} standard deviations from mean) ----------")
    print(f"Found {len(outliers)} extreme outliers.")
    if len(outliers) > 0:
        print(outliers.head(25))
    
    return results

def cleanDataset():
    """ Clean the dataset: -999 (missing Vals), duplicates, impossible values, outliers"""
    engine = database.getSQLAlchemyEngine()

    print("---------- Starting Dataset Cleaning ----------")

    # 1. Handle -999s
    convert999Query = """
    UPDATE waterdata
    SET
        discharge = CASE WHEN discharge = -999 THEN NULL ELSE discharge END,
        gage_height = CASE WHEN gage_height = -999 THEN NULL ELSE gage_height END,
        temperature = CASE WHEN temperature = -999 THEN NULL ELSE temperature END,
        dissolved_oxygen = CASE WHEN dissolved_oxygen = -999 THEN NULL ELSE dissolved_oxygen END,
        stream_elevation = CASE WHEN stream_elevation = -999 THEN NULL ELSE stream_elevation END
    """

    with engine.begin() as conn:
        conn.execute(text(convert999Query))
        print("Converted -999 Values to NULL")

    # 2. Remove Duplicates

    deleteDupeQuery = """
    DELETE FROM waterdata
    WHERE id NOT IN (
    SELECT MIN(id)
    FROM waterdata
    GROUP BY site_code, date
    )"""

    with engine.begin() as conn: 
        result = conn.execute(text(deleteDupeQuery))
        print("Removed Duplicate Records.")

    # 3. Remove impossible values

    impossibleQuery = """
        UPDATE waterdata
        SET
        discharge = CASE WHEN discharge < 0 THEN NULL ELSE discharge END,
        gage_height = CASE WHEN gage_height < 0 THEN NULL ELSE gage_height END,
        temperature = CASE WHEN temperature < -50 OR temperature > 50 THEN NULL ELSE temperature END,
        dissolved_oxygen = CASE WHEN dissolved_oxygen < 0 or dissolved_oxygen > 50 THEN NULL ELSE dissolved_oxygen END
    """

    with engine.begin() as conn:
        conn.execute(text(impossibleQuery))
        print("Removed Impossible Values")

    print("Cleaning Complete")
    return True

def verifyDatasetCleaning():
    """Verify the dataset cleaning worked"""
    engine = database.getSQLAlchemyEngine()
    
    print("Verficiation Report")
    # Check for records that still have -999
    check99Query = """
    SELECT 
        COUNT(CASE WHEN discharge = -999 THEN 1 END) as discharge999,
        COUNT(CASE WHEN gage_height = -999 THEN 1 END) as gage999,
        COUNT(CASE WHEN temperature = -999 THEN 1 END) as temp999,
        COUNT(CASE WHEN dissolved_oxygen = -999 THEN 1 END) as do999,
        COUNT(CASE WHEN stream_elevation = -999 THEN 1 END) as elev999
    FROM waterdata
    """
    check999 = pd.read_sql(check99Query, engine)

    print("----- Checking for remaining -999 values -----")
    print(check999)

    #Check for records that are duplicated

    dupeCheckQuery = """
    SELECT site_code, date, COUNT(*) as count
    FROM waterdata
    GROUP BY site_code, date
    HAVING COUNT(*) > 1
    LIMIT 25
    """

    duplicates = pd.read_sql(dupeCheckQuery, engine)
    print("----- Checking Duplicates -----")
    if len(duplicates) > 0:
        print(duplicates)
    else: 
        print("No duplicates")

    #Check for impossible values

    impossibleQuery = """
        SELECT
            COUNT(CASE WHEN discharge < 0 THEN 1 END) as negative_discharge,
            COUNT(CASE WHEN gage_height < 0 THEN 1 END) as negative_gage,
            COUNT(CASE WHEN temperature < -50 OR temperature > 50 THEN 1 END) as extreme_temp,
            COUNT(CASE WHEN dissolved_oxygen < 0 OR dissolved_oxygen > 50 THEN 1 END) as extreme_do
        FROM waterdata
    """

    impossible = pd.read_sql(impossibleQuery, engine)

    print("Checking Impossible Values")
    print(impossible)

    # Missing Value Summary

    missingQuery = """
        SELECT
            COUNT(*) as total_records,
            COUNT(discharge) as has_discharge,
            COUNT(gage_height) as has_gage,
            COUNT(temperature) as has_temp,
            COUNT(dissolved_oxygen) as has_do,
            ROUND(100.0 * (COUNT(*) - COUNT(discharge)) / COUNT(*), 2) as pct_missing_discharge,
            ROUND(100.0 * (COUNT(*) - COUNT(gage_height)) / COUNT(*), 2) as pct_missing_gage,
            ROUND(100.0 * (COUNT(*) - COUNT(temperature)) / COUNT(*), 2) as pct_missing_temp,
            ROUND(100.0 * (COUNT(*) - COUNT(dissolved_oxygen)) / COUNT(*), 2) as pct_missing_do
        FROM waterdata
    """
    missing = pd.read_sql(missingQuery, engine)
    print("----- Missing value summary -----")
    print(missing)
    print()

    #Check ohio river. Had extremely high discharge values, but it lines up with flood season in the spring so I want to make sure it wasnt removed

    ohioRiverQuery = """
    SELECT COUNT(*) AS count
    FROM waterdata
    WHERE site_code = '03612600'
    AND discharge > 1000000
    """

    ohioRiver = pd.read_sql(ohioRiverQuery, engine)

    print(f"----- Ohio River Extreme Events (>1Million cf/s flow): {ohioRiver.iloc[0, 0]} records")
    print("Expecting ~25 legitimate flood records")

    # Overall Summary
    summaryQuery = """
    SELECT
        COUNT(*) as total_records,
        COUNT(DISTINCT site_code) as num_sites,
        MIN(date) as earliest_date,
        MAX(date) as latest_date
    FROM waterdata
    """

    summ = pd.read_sql(summaryQuery, engine)
    print("----- OVERALL DATASET SUMMARY -----")
    print(summ)

    print("---------- Verification Complete -----------")
    return True

if __name__ == "__main__":
    cleanDataset()

    verifyDatasetCleaning()