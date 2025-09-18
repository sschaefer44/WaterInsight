import pandas
import requests
from datetime import datetime
import database

def getSites(stateCode, startDate, endDate):
    """Get all water monitoring sites within a state by state code. E.g. Wisconsin = WI, Minnesota = MN...."""
    initURL = "https://waterservices.usgs.gov/nwis/dv/"

    parameters = {
        'format': 'json',
        'stateCd': stateCode,
        'startDT': startDate,
        'endDT': endDate,
        'siteStatus': 'active'
    }

    try:
        response = requests.get(initURL, params=parameters)
        response.raise_for_status()
        data = response.json()

        # Use a set to track unique site codes and avoid duplicates
        sites = []
        seen_sites = set()
        
        if 'value' in data and 'timeSeries' in data['value']:
            for siteInfo in data['value']['timeSeries']:
                site_code = siteInfo['sourceInfo']['siteCode'][0]['value']
                
                # Only add if we haven't seen this site before
                if site_code not in seen_sites:
                    siteData = {
                        'siteCode': site_code,
                        'siteName': siteInfo['sourceInfo']['siteName'],
                        'latitude': float(siteInfo['sourceInfo']['geoLocation']['geogLocation']['latitude']),
                        'longitude': float(siteInfo['sourceInfo']['geoLocation']['geogLocation']['longitude'])
                    }
                    sites.append(siteData)
                    seen_sites.add(site_code)
        
        return sites
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sites for {stateCode}: {e}")
        return []
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        return []

def getWaterData(siteCode, startDate, endDate, metrics):
    """Get specified water metrics from specified sites within a date range."""

    initURL = "https://waterservices.usgs.gov/nwis/dv/"
    
    parameters = {
        'format': 'json',
        'sites': siteCode,
        'startDT': startDate,
        'endDT': endDate,
        'parameterCd': metrics,
        'siteStatus': 'active'
    }

    try: 
        response = requests.get(initURL, params = parameters)
        response.raise_for_status()
        data = response.json()

        waterData = []
        if 'value' in data and 'timeSeries' in data['value']:
            for timeseries in data['value']['timeSeries']:
                parameterInfo = timeseries['variable']
                values = timeseries['values'][0]['value']

                for valueEntry in values:
                    if valueEntry['value'] != '-999999':
                        waterData.append({
                            'siteCode': siteCode,
                            'date': valueEntry['dateTime'][:10],
                            'parameterCode': parameterInfo['variableCode'][0]['value'],
                            'parameterName': parameterInfo['variableDescription'],
                            'value': float(valueEntry['value'])
                        })      
        return waterData
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for site {siteCode}: {e}")
        return []
    
def getWaterDataContinuous(siteCode, startDate, endDate, metrics):
    """Get continuous water metrics from specified sites within a date range."""
    
    # Use instantaneous values service instead of daily values
    initURL = "https://waterservices.usgs.gov/nwis/iv/"  # 'iv' = instantaneous values
    
    parameters = {
        'format': 'json',
        'sites': siteCode,
        'startDT': startDate,
        'endDT': endDate,
        'parameterCd': metrics,
        'siteStatus': 'active'
    }

    try: 
        response = requests.get(initURL, params=parameters)
        response.raise_for_status()
        data = response.json()

        waterData = []
        if 'value' in data and 'timeSeries' in data['value']:
            for timeSeries in data['value']['timeSeries']:
                parameterInfo = timeSeries['variable']
                values = timeSeries['values'][0]['value']

                for valueEntry in values:
                    if valueEntry['value'] != '-999999':
                        waterData.append({
                            'siteCode': siteCode,
                            'date': valueEntry['dateTime'][:10],  # Extract date
                            'datetime': valueEntry['dateTime'],   # Full timestamp
                            'parameterCode': parameterInfo['variableCode'][0]['value'],
                            'parameterName': parameterInfo['variableDescription'],
                            'value': float(valueEntry['value'])
                        })      
        return waterData
    except requests.exceptions.RequestException as e:
        print(f"Error fetching continuous data for site {siteCode}: {e}")
        return []    

def getWaterDataDaily(siteCode, startDate, endDate, metrics):
    """Get daily water metrics (for suspended sediment data)"""
    initURL = "https://waterservices.usgs.gov/nwis/dv/"  # Daily values
    
    parameters = {
        'format': 'json',
        'sites': siteCode,
        'startDT': startDate,
        'endDT': endDate,
        'parameterCd': metrics,
        'siteStatus': 'active'
    }

    try: 
        response = requests.get(initURL, params=parameters)
        response.raise_for_status()
        data = response.json()

        waterData = []
        if 'value' in data and 'timeSeries' in data['value']:
            for timeSeries in data['value']['timeSeries']:
                parameterInfo = timeSeries['variable']
                values = timeSeries['values'][0]['value']

                for valueEntry in values:
                    if valueEntry['value'] != '-999999':
                        waterData.append({
                            'siteCode': siteCode,
                            'date': valueEntry['dateTime'][:10],
                            'parameterCode': parameterInfo['variableCode'][0]['value'],
                            'parameterName': parameterInfo['variableDescription'],
                            'value': float(valueEntry['value'])
                        })      
        return waterData
    except requests.exceptions.RequestException as e:
        print(f"Error fetching daily data for site {siteCode}: {e}")
        return []

def getCompleteWaterData(siteCode, startDate, endDate):
    """Get both continuous and daily data for a site"""
    
    # Get continuous data (discharge, gage height, elevation)
    continuous_params = '00060,00065,63160'  # discharge, gage height, stream elevation
    continuous_data = getWaterDataContinuous(siteCode, startDate, endDate, continuous_params)
    
    # Get daily data (suspended sediment)
    daily_params = '80154,80155'  # suspended sediment concentration and discharge
    daily_data = getWaterDataDaily(siteCode, startDate, endDate, daily_params)
    
    # Combine both datasets
    all_data = continuous_data + daily_data
    
    return all_data

def exploreAllParameters(siteCode, startDate, endDate):
    """Get all available parameters for a site to see what's actually there"""
    initURL = "https://waterservices.usgs.gov/nwis/iv/"
    
    # Don't specify parameterCd - get ALL available parameters
    parameters = {
        'format': 'json',
        'sites': siteCode,
        'startDT': startDate,
        'endDT': endDate,
        'siteStatus': 'active'
        # No parameterCd specified = get everything available
    }

    try: 
        response = requests.get(initURL, params=parameters)
        response.raise_for_status()
        data = response.json()

        parameters_found = {}
        if 'value' in data and 'timeSeries' in data['value']:
            for timeSeries in data['value']['timeSeries']:
                param_info = timeSeries['variable']
                param_code = param_info['variableCode'][0]['value']
                param_name = param_info['variableDescription']
                
                if param_code not in parameters_found:
                    parameters_found[param_code] = param_name
        
        return parameters_found
    except requests.exceptions.RequestException as e:
        print(f"Error exploring parameters for site {siteCode}: {e}")
        return {}

def validateDataStructure(water_records):
    """Ensure data structure matches database expectations"""
    required_fields = ['site_code', 'date', 'discharge', 'gage_height', 
                      'stream_elevation', 'temperature', 'dissolved_oxygen']
    
    for record in water_records[:3]:  # Check first few records
        for field in required_fields:
            if field not in record:
                print(f"Missing field: {field}")
                return False
        print(f"Valid record: {record['date']} - {record['site_code']}")
    return True

def getStandardizedWaterData(siteCode, startDate, endDate):
    """
    Collect 5 standard water metrics, filling missing metrics with -999.
    Returns consistent data structure regardless of site parameter availability.
    """
    
    TARGET_PARAMS = {
        '00060': 'Discharge, cubic feet per second',
        '00065': 'Gage height, feet', 
        '63160': 'Stream water level elevation above NAVD 1988, in feet',
        '00010': 'Temperature, water, degrees Celsius',
        '00300': 'Dissolved oxygen, water, unfiltered, milligrams per liter'
    }
    
    # Try to get all parameters
    all_params = ','.join(TARGET_PARAMS.keys())
    water_data = getWaterDataContinuous(siteCode, startDate, endDate, all_params)
    
    if not water_data:
        return []
    
    # Group data by date and parameter
    data_by_date = {}
    for entry in water_data:
        date = entry['date']
        param = entry['parameterCode']
        
        if date not in data_by_date:
            data_by_date[date] = {}
        
        data_by_date[date][param] = entry['value']
    
    # Create standardized records with consistent structure
    standardized_data = []
    for date, params in data_by_date.items():
        record = {
            'site_code': siteCode,
            'date': date,
            'discharge': params.get('00060', -999),
            'gage_height': params.get('00065', -999),
            'stream_elevation': params.get('63160', -999),
            'temperature': params.get('00010', -999),
            'dissolved_oxygen': params.get('00300', -999)
        }
        standardized_data.append(record)
    
    return standardized_data
