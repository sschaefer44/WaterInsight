import database
import pandas as pd
import matplotlib.pyplot as plt
import warnings

def getDatasetOverview():
    """Get comprehensive overview of entire water data set"""
    import time
    
    conn = database.getConnection()
    
    overviewQuery = """
        SELECT
            COUNT(*) as totalRecords,
            COUNT(DISTINCT site_code) as totalSites,
            MIN(date) as firstDate,
            MAX(date) as latestDate,
            EXTRACT(YEAR FROM MAX(date)) - EXTRACT(YEAR FROM MIN(date)) as yearRange
        FROM waterdata
    """
    
    overview = pd.read_sql(overviewQuery, conn)

    stateSummaryQuery = """
        SELECT
            sl.state,
            COUNT(wd.*) as totalRecords,
            COUNT(DISTINCT wd.site_code) as uniqueSites,
            MIN(wd.date) as earliestDate,
            MAX(wd.date) as latestDate
        FROM waterdata wd
        JOIN sitelocations sl ON wd.site_code = sl.site_code
        GROUP BY sl.state
        ORDER BY totalRecords DESC
    """
    
    stateSummary = pd.read_sql(stateSummaryQuery, conn)

    parameterAvailQuery = """
        SELECT
            COUNT(*) as totalRecords,
            COUNT(CASE WHEN discharge != -999 THEN 1 END) as dischargeAvailable,
            COUNT(CASE WHEN gage_height != -999 THEN 1 END) as gageHeightAvailable,
            COUNT(CASE WHEN stream_elevation != -999 THEN 1 END) as streamElevationAvailable,
            COUNT(CASE WHEN temperature != -999 THEN 1 END) as temperatureAvailable,
            COUNT(CASE WHEN dissolved_oxygen != -999 THEN 1 END) as dissolvedOxygenAvailable,
            ROUND(100.0 * COUNT(CASE WHEN discharge != -999 THEN 1 END) / COUNT(*), 2) as dischargePct,
            ROUND(100.0 * COUNT(CASE WHEN temperature != -999 THEN 1 END) / COUNT(*), 2) as temperaturePct,
            ROUND(100.0 * COUNT(CASE WHEN dissolved_oxygen != -999 THEN 1 END) / COUNT(*), 2) as dissolvedOxygenPct
        FROM waterdata
    """
    
    paramAvailability = pd.read_sql(parameterAvailQuery, conn)

    conn.close()
    
    return overview, stateSummary, paramAvailability

def displayOverview():
    """Display formatted overview of dataset"""

    overview, stateSummary, parameterAvail = getDatasetOverview()
    
    print("=" * 80)
    print("WaterInsight Dataset Overview")
    print("=" * 80)    

    # PostgreSQL returns lowercase column names, so this needs to be all lowercase
    totalRecords = overview['totalrecords'].iloc[0]
    totalSites = overview['totalsites'].iloc[0]
    earliest = overview['firstdate'].iloc[0]
    latest = overview['latestdate'].iloc[0]
    years = overview['yearrange'].iloc[0]

    print(f"Total Records: {totalRecords:,}")
    print(f"Total Sites: {totalSites:,}")
    print(f"Date Range: {earliest} to {latest} ({years} years)")
    print(f"Average Records per Site: {totalRecords/totalSites:.0f}")

    print(f"\nRECORDS BY STATE:")
    print("-" * 50)
    for _, row in stateSummary.iterrows():
        print(f"{row['state']}: {row['totalrecords']:,} records from {row['uniquesites']} sites")
        print(f"    Date range: {row['earliestdate']} to {row['latestdate']}")

    print(f"\nPARAMETER AVAILABILITY:")
    print("-" * 50)
    params = parameterAvail.iloc[0]
    print(f"Discharge: {params['dischargeavailable']:,} records ({params['dischargepct']}%)")
    print(f"Gage Height: {params['gageheightavailable']:,} records ({100.0 * params['gageheightavailable'] / params['totalrecords']:.1f}%)")
    print(f"Stream Elevation: {params['streamelevationavailable']:,} records ({100.0 * params['streamelevationavailable'] / params['totalrecords']:.1f}%)")
    print(f"Temperature: {params['temperatureavailable']:,} records ({params['temperaturepct']}%)")
    print(f"Dissolved Oxygen: {params['dissolvedoxygenavailable']:,} records ({params['dissolvedoxygenpct']}%)")

if __name__ == "__main__":
    displayOverview()