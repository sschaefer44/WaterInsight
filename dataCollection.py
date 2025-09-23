import database
from API import getSites, getStandardizedWaterData, demonstrateAPICollection
import time
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')

def collectStateHistorical(stateCode, startYear, endYear):
    """Collect data for all sites in a state within a specified year range"""

    sites = getSites(stateCode, f"{startYear}-01-01", f"{endYear}-12-31")
    print(f"Found {len(sites)} sites in {stateCode}")

    successSites = 0
    total = 0

    for i, site in enumerate(sites):
        print(f"Obtaining data from site {i + 1}/{len(sites)}: {site['siteName'][:50]}:")

        database.insertSiteLocation(site, stateCode) # Insert site location data into siteLocation table in pgDB

        siteRecords = 0
        for year in range(startYear, endYear + 1):
            startDate = f'{year}-01-01'
            endDate = f'{year}-12-31'

            try:
                waterData = getStandardizedWaterData(site['siteCode'], startDate, endDate)

                if waterData:
                    insertedRecords = database.insertWaterData(waterData)
                    siteRecords += insertedRecords
                    print(f"    Year {year}: {insertedRecords} records inserted.")

                    time.sleep(1) # 1 second sleep to ensure the number of requests to api doesn't overwhelm the api

            except Exception as e: 
                print(f"Error collecting {year} data: {e}")

        if siteRecords > 0:
            successSites += 1
            total += siteRecords
            print(f"Site Records Total: {siteRecords} Records")

        time.sleep(2)

    print(f"State {stateCode}: {successSites} sites, {total} records")
    return total

def collectMultiStateHistorical(states, startYear, endYear):
    """Collect historical data from all state codes passed as a list"""

    for i, state in enumerate(states):
        print(f"Collecting {states[i]}")
        
        try:
            collectStateHistorical(state, startYear, endYear)
        except Exception as e:
            print(f"Failed to collect {state} data: {e}")
            continue

        time.sleep(5)

if __name__ == "__main__":
    demonstrationSites, totalRecords = demonstrateAPICollection('WI', 3, '2025-07-14', '2025-07-15')

    if demonstrationSites:
        engine = database.getSQLAlchemyEngine()
        
        siteCodes = [site['siteCode'] for site in demonstrationSites]
        siteCodesStr = "', '".join(siteCodes)
        
        recentRecordsQuery = f"""
            SELECT wd.site_code, sl.site_name, wd.date, wd.discharge, 
                   wd.gage_height, wd.stream_elevation, wd.temperature, wd.dissolved_oxygen
            FROM waterdata wd
            JOIN sitelocations sl ON wd.site_code = sl.site_code
            WHERE wd.site_code IN ('{siteCodesStr}')
            AND wd.date BETWEEN '2025-07-14' AND '2025-07-15'
            ORDER BY wd.site_code, wd.date
        """
        
        recentRecords = pd.read_sql(recentRecordsQuery, engine)
        
        print(f"\nShowing the {len(recentRecords)} records just added:")
        print("=" * 100)
        for _, row in recentRecords.iterrows():
            print(f"Site: {row['site_name'][:50]} | Date: {row['date']} | Discharge: {row['discharge']} | Gage Height: {row['gage_height']}")