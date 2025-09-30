from featureEngineering import *
import loadData
import pandas as pd 
import numpy as np

def createFeatures(df):
    """Master function that creates all features using featureEngineering.py functions"""

    print("\n" + "-" * 15)
    print("Feature Engineering Pipeline")
    print("\n" + "-" * 15)
    print(f"Intital DF Shape: {df.shape}\n")

    df = df.sort_values(['site_code', 'date']).reset_index(drop = True)

    df = temporalFeatures(df)
    df = lagFeatures(df)
    df = rollingFeatures(df)
    df = trendFeatures(df)
    df = climatologyFeatures(df)
    df = siteFeatures(df)
    df = crossVarFeatures(df)
    df = categoricalEnconding(df)    

    print(f"Final Shape: {df.shape}\n")
    print(f"Features Added: {df.shape[1] - 10}")

    return df

def getFeatureCols(df):
    """Identify then return feature columns Vs. metadata columns"""

    excludedCols = ['site_code', 'date', 'discharge', 'temperature', 
                    'dissolved_oxygen', 'latitude', 'longitude', 'state']
    
    featureCols = [col for col in df.columns if col not in excludedCols]
    return featureCols

def cleanMissingFeatures(df, featureCols):
    """Handle missing values in engineered features"""

    print("\n" + "-" * 15)
    print("Handling Missing Feature Values")
    print("\n" + "-" * 15)

    # Check missing Vals
    missing = df[getFeatureCols].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending = False)

    if len(missing) > 0:
        print(f"\nFeatures with missing Values:")
        print(missing.head(15))

        criticalLagFeatures = ['dischargeLag1', 'dischargeLag7', 'gageHeightLag1', 'gageHeightLag7']

        print(f"\nRows before dropping NaN lag features: {len(df):,}")
        cleanedDf = df.dropna(subset = criticalLagFeatures)
        print(f"\nRows after dropping NaN lag features: {len(cleanedDf):,}")
        print(f"Dropped: {len(df) - len(cleanedDf):,} rows ({(len(df) - len(cleanedDf))/len(df)*100:.2f}%")
        
        cleanedDf[featureCols] = cleanedDf[featureCols].fillna(0)

        remainingNaN = cleanedDf[featureCols].isnull().sum().sum()
        print(f"Remaining NaN: {remainingNaN}")
        
        return cleanedDf
    
def timeSeriesTrainTestSplit(df, targetCol):
    """
    Split data by year for training, testing, etc. 
    TrainTestSplit from scikit-learn can not be used because this is timeseries data. 
    Sequential data is needed to learn meterological, seasonal, etc. patterns
    """

    print("\n" + "-" * 15)
    print("Splitting into Train/Val/Test sets")
    print("\n" + "-" * 15)

    trainDF = df[df['year'] <= 2022]
    valDF = df[df['year'] == 2023]
    testDF = df[df['year'] == 2024]

    print(f"\nTrain (2016-2022): {len(trainDF):,} rows ({len(trainDF)/len(df)*100:.1f}%)")
    print(f"\nValidation (2023): {len(valDF):,}   rows ({len(valDF)/len(df)*100:.1f}%)")
    print(f"\nTest (2024):       {len(testDF):,}  rows ({len(testDF)/len(df)*100:.1f}%)")

    print(f"\nTarget ({targetCol}) Statistics")
    print(f"  Train: mean={trainDF[targetCol].mean():.2f}, std={trainDF[targetCol].std():.2f}")
    print(f"  Val:   mean={valDF[targetCol].mean():.2f}, std={valDF[targetCol].std():.2f}")
    print(f"  Test:  mean={testDF[targetCol].mean():.2f}, std={testDF[targetCol].std():.2f}")
    
    return trainDF, valDF, testDF

def saveFeatureData(df, featureCols, filename = '/Users/sschaefer/Desktop/WaterInsight/CSV Backups/engineeredFeatures.csv'):
    """Save"""

    print(f"\nSaving to {filename}...")

    df.to_csv(filename, index = False)

    with open('featureColumns.txt', 'w') as f:
        for col in featureCols:
            f.write(f"{col}\n")
    
    print(f"Saved {len(featureCols)} feature names to featureColumns.txt")

def summarizeFeatures(feature_cols):
    """Print feature summary"""
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    temporal = [c for c in feature_cols if any(t in c for t in ['year', 'month', 'day', 'season', 'week'])]
    lagged = [c for c in feature_cols if 'lag' in c or 'Lag' in c]
    rolling = [c for c in feature_cols if 'rolling' in c or 'Rolling' in c]
    trend = [c for c in feature_cols if 'trend' in c or 'Trend' in c or 'Delta' in c]
    climatology = [c for c in feature_cols if 'climate' in c or 'Climate' in c or 'anomaly' in c or 'Anomaly' in c or 'ZScore' in c]
    site = [c for c in feature_cols if 'site_' in c or 'latitude' in c or 'longitude' in c or 'distance' in c or 'elevation' in c or 'seasonality' in c]
    cross = [c for c in feature_cols if 'ratio' in c or 'temp_' in c or 'freezing' in c]
    encoded = [c for c in feature_cols if 'encoded' in c]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"  Temporal:     {len(temporal)}")
    print(f"  Lagged:       {len(lagged)}")
    print(f"  Rolling:      {len(rolling)}")
    print(f"  Trend:        {len(trend)}")
    print(f"  Climatology:  {len(climatology)}")
    print(f"  Site:         {len(site)}")
    print(f"  Cross-var:    {len(cross)}")
    print(f"  Encoded:      {len(encoded)}")


# MAIN

if __name__ == "__main__":
    startTime = time.time()

    print("\n" + "-" * 15)
    print("Water Discharge Feature Engineering")
    print("\n" + "-" * 15)

    df = loadData.loadData()
    
    dfEngineered = createFeatures(df)
    
    featureCols = getFeatureCols(dfEngineered)

    cleanedDF = cleanMissingFeatures(dfEngineered, featureCols)

    trainDf, valDf, testDf = timeSeriesTrainTestSplit(cleanedDF, 'discharge')
    
    saveFeatureData(cleanedDF, featureCols)

    summarizeFeatures(featureCols)

    elapsed = time.time() - startTime

    print(f"\nTotal time: {elapsed/60:.2f} minutes")
    print("\n" + "-" * 15)
    print("Feature Engineering Complete")
    print("\n" + "-" * 15)

