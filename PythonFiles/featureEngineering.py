import database
import pandas as pd
import numpy as np
import loadData
import time

def temporalFeatures(df):
    """Create features based on time"""

    # Break down date and derive features (day of week, day of year, etc)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayOfYear'] = df['date'].dt.dayofyear
    df['dayOfWeek'] = df['date'].dt.dayofweek
    df['weekOfYear'] = df['date'].dt.isocalendar().week

    # Seasonal features, based on metorological seasons
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    })

    # Cycle encoding. Raw month number will be treated as continuous numbers by model. 
    # By encoding in the fomrat of: (sin, cosine), it fixes the continous number issue.
    # E.G. December (12) and January (1) are treated as very far apart months when using 1-12.
    # When they are encoded to (sin, cosine) the last day of December is next to the first day of January.
    df['dayOfYearSine'] = np.sin(2 * np.pi * df['dayOfYear'] /365.25)
    df['dayOfYearCosine'] = np.cos(2 * np.pi * df['dayOfYear'] / 365.25)
    df['monthSine'] = np.sin(2 * np.pi * df['month'] / 12)
    df['monthCosine'] = np.cos(2 * np.pi * df['month'] / 12)

    print(f"    Added Temporal Features")
    return df

def lagFeatures(df):
    """Create lagged features by site"""

    lagVals = [1, 7, 14, 30] # Day, week, 2 weeks, month

    for lagVal in lagVals: # Create lag feature for each of lagVals for discharge and gage height
        df[f'dischargeLag{lagVal}'] = df.groupby('site_code')['discharge'].shift(lagVal)
        df[f'gageHeightLag{lagVal}'] = df.groupby('site_code')['gage_height'].shift(lagVal)

    print(f" Added {len(lagVals) * 2} lagged features")
    return df

def rollingFeatures(df):
    """Create rolling window statistics by site"""

    windows = [7, 14, 30] # week, 2 weeks, month

    for window in windows:
        # Discharge rolling statistics 
        df[f'dischargeRollingMean{window}'] = ( # Rolling Mean
            df.groupby('site_code')['discharge'].transform(lambda x: x.rolling(window, min_periods = 1).mean())
        )

        df[f'dischargeRollingStd{window}'] = ( # Rolling standard deviation
            df.groupby('site_code')['discharge'].transform(lambda x: x.rolling(window, min_periods = 1).std())
        )

        df[f'dischargeRollingMin{window}'] = ( # Rolling Minimum
            df.groupby('site_code')['discharge'].transform(lambda x: x.rolling(window, min_periods = 1).min())
        )
        
        df[f'dischargeRollingMax{window}'] = ( # Rolling Maximum
            df.groupby('site_code')['discharge'].transform(lambda x: x.rolling(window, min_periods = 1).max())
        )
        
        # Gage Height rolling statistics
        df[f'gageHeightRollingMean{window}'] = ( # Rolling Mean
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).mean())
        )

        df[f'gageHeightRollingStd{window}'] = ( # Rolling standard deviation
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).std())
        )

        df[f'gageHeightRollingMin{window}'] = ( # Rolling Minimum
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).min())
        )
        
        df[f'gageHeightRollingMax{window}'] = ( # Rolling Maximum
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).max())
        )

        print(f"   Added {len(windows) * 8} rolling features.")
        return df

def calculateTrend(series):
    if len(series < 2):
        return 0
    x = np.arrange(len(series))
    y = series.values
    if np.any(np.isnan(y)):
        return 0
    try:
        slope = np.polyfit(x, y, 1)[0]
        return slope
    except:
        return 0

def trendFeatures(df):
    """Create trend and delta features"""

    df['dischargeDelta1D'] = df.groupby('site_code')['discharge'].diff(1)
    df['dischargeDelta7D'] = df.groupby('site_code')['discharge'].diff(7)
    df['dischargeDelta14D'] = df.groupby('site_code')['discharge'].diff(14)
    df['dischargeDelta30D'] = df.groupby('site_code')['discharge'].diff(30)
    df['gageHeightDelta1D'] = df.groupby('site_code')['gage_height'].diff(1)
    df['gageHeightDelta7D'] = df.groupby('site_code')['gage_height'].diff(7)
    df['gageHeightDelta14D'] = df.groupby('site_code')['gage_height'].diff(14)
    df['gageHeightDelta30D'] = df.groupby('site_code')['gage_height'].diff(30)

    df['dischargeTrend7D'] = df.groupby('site_code')['discharge'].transform(
        lambda x: x.rolling(7, min_periods = 2).apply(calculateTrend, raw= False)
    )

    df['gageHeightTrend7D'] = df.groupby('site_code')['gage_height'].transform(
        lambda x: x.rolling(7, min_periods = 2).apply(calculateTrend, raw= False)
    )

    print("Added trend features")
    return df

def climatologyFeatures(df): 
    climatologyDischarge = df.groupby(['site_code', 'dayOfYear'])['discharge'].agg([
        ('dischargeClimateMean', 'mean'),
        ('dischargeClimateStd', 'std'),
        ('dischargeClimateMin', 'min'),
        ('dischargeClimateMax', 'max'),
        ('dischargeClimateQuant25', lambda x: x.quantile(0.25)),
        ('dischargeClimateQuant75', lambda x: x.quantile(0.75))
    ]).reset_index()

    climatologyGageHeight = df.groupby(['site_code', 'dayOfYear'])['gage_height'].agg([
        ('gageHClimateMean', 'mean'),
        ('gageHClimateStd', 'std'),
        ('gageHClimateMin', 'min'),
        ('gageHClimateMax', 'max'),
        ('gageHClimateQuant25', lambda x: x.quantile(0.25)),
        ('gageHClimateQuant75', lambda x: x.quantile(0.75))
    ]).reset_index()

    df = df.merge(climatologyDischarge, on=['site_code', 'dayOfYear'], how = 'left')
    df = df.merge(climatologyGageHeight, on = ['site_code', 'dayOfYear'], how = 'left')

    # Anomaly Columns, checking Vs mean. Z scores
    df['dischargeAnomaly'] = df['discharge'] - df['dischargeClimateMean']
    df['dischargeZScore'] = (df['discharge'] - df['dischargeClimateMean']) / (df['dischargeClimateStd'] + 0.000001) # adding 0.000001 ensures no / by 0
    
    df['gageHeightAnomaly'] = df['gage_height'] - df['gageHClimateMean']
    df['gageHeightZScore'] = (df['gage_height'] - df['gageHClimateMean']) / (df['gageHClimateStd'] + 0.000001) # adding 0.000001 ensures no / by 0

    print("     Added 16 climatology features")
    return df

def siteFeatures(df):
    """Create site based features"""

    siteStats = df.groupby('site_code').agg({
        'discharge': ['mean', 'std', 'min', 'max'],
        'gage_height': ['mean', 'std', 'min', 'max'],
        'stream_elevation': 'first',
        'latitude': 'first',
        'longitude': 'first'
    })

    siteStats.columns = ['siteDischargeMean', 'siteDischargeStd', 'siteDischargeMin', 'siteDischargeMax',
                       'siteGageHeightMean', 'siteGageHeightStd', 'siteGageHeightMin', 'siteGageHeightMax',
                       'siteElevation', 'siteLatitude', 'siteLongitude']
    
    siteStats = siteStats.reset_index()

    df = df.merge(siteStats, on = 'site_code', how = 'left')

    # Seasonality Strength By Site
    monthlyDischarge = df.groupby(['site_code', 'month'])['discharge'].mean().reset_index()
    seasonalityDischarge = monthlyDischarge.groupby('site_code')['discharge'].std().reset_index()
    seasonalityDischarge.columns = ['site_code', 'siteDischargeSeasonality']
    df = df.merge(seasonalityDischarge, on = 'site_code', how = 'left')

    monthlyGageHeight = df.groupby(['site_code', 'month'])['gage_height'].mean().reset_index()
    seasonalityGageHeight = monthlyGageHeight.groupby('site_code')['gage_height'].std().reset_index()
    seasonalityGageHeight.columns = ['site_code', 'siteGageHeightSeasonality']
    df = df.merge(seasonalityGageHeight, on = 'site_code', how = 'left')

    # Geographic features cycle encoding (as done in temporalFeatures)
    df['latitudeSine'] = np.sin(np.radians(df['siteLatitude']))
    df['latitudeCosine'] = np.cos(np.radians(df['siteLatitude']))
    df['longitudeSine'] = np.sin(np.radians(df['siteLongitude']))
    df['longitudeCosine'] = np.cos(np.radians(df['siteLongitude']))

    # Calculate centroid (lat. and long.) of sites
    centroidLatitude = df['siteLatitude'].mean()
    centroidLongitude = df['siteLongitude'].mean()
    print(f"    Sites Centroid: {centroidLatitude:.2f}°N, {centroidLongitude:.2f}°W")

    df['distanceFromCentroid'] = np.sqrt( # Calculate using Euclidean Distance Formula: Distance = √((x₂ - x₁)² + (y₂ - y₁)²)
        (df['siteLatitude'] - centroidLatitude) **2 +
        (df['siteLongitude'] - centroidLongitude) **2
    )

    print(f"    Added by-site features")
    return df

def crossVarFeatures(df):
    """Create features based on variable interactions"""

    df['dischargeGageHeightRatio'] = df['discharge'] / (df['gage_height'] + 0.01) # + 0.01 to prevent division by 0

    # Temperature features (NOTE: most monitoring locations don't provide temperature data)
    df['temperatureLag1'] = df.groupby('site_code')['temperature'].shift(1)
    df['temperatureRolling7'] = df.groupby('site_code')['temperature'].transform(
        lambda x: x.rolling(7, min_periods = 1).mean()
    )
    df['tempAboveFreezing'] = (df['temperature'] > 0).astype(float)

    print("     Added cross variable features")

    return df

def categoricalEnconding(df):
    """
    Encode sites by average discharge cfs
    MLP can't handle catergorical site codes. 
    By average cfs, it tells the model that any given site is a high, low, moderate, etc. flow site
    """
    siteEncoding = df.groupby('site_code')['discharge'].mean().to_dict()
    df['encodedSiteCode'] = df['site_code'].map(siteEncoding)

    print("     Added encoded feature")
    return df