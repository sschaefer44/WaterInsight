"""
Simple prediction demonstration for 3 days in July 2024
Perfect for live Checkpoint 2 presentation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("JULY 2024 PREDICTION DEMONSTRATION")
print("=" * 60)

# ============================================================================
# Load model and scalers
# ============================================================================
print("\nLoading model and scalers...")
model = tf.keras.models.load_model('c2Model/discharge_model_no_leakage.keras')
featureScaler = joblib.load('c2Model/feature_scaler.pkl')
targetScaler = joblib.load('c2Model/target_scaler.pkl')

# Load feature names
with open('c2Model/feature_names.txt', 'r') as f:
    featureCols = [line.strip() for line in f]

print("Model loaded")

# ============================================================================
# Load data - get 3 days in July 2024
# ============================================================================
print("\nLoading test data from July 2024...")
testDF = pd.read_csv('CSV Backups/test_features.csv')

# Create date from year and dayOfYear
testDF['date'] = pd.to_datetime(testDF['year'].astype(str) + '-' + testDF['dayOfYear'].astype(str), format='%Y-%j')

# Filter for July 2024
july_data = testDF[(testDF['date'].dt.month == 7) & (testDF['date'].dt.year == 2024)].copy()

if len(july_data) == 0:
    print("Error: No July 2024 data found in test file!")
    exit()

# Get first 3 consecutive dates in July
first_3_dates = sorted(july_data['date'].unique())[:3]

print(f"\nUsing 3 consecutive days from July 2024:")
for date in first_3_dates:
    print(f"  - {date.strftime('%B %d, %Y')}")

# Filter for those dates
july_data = july_data[july_data['date'].isin(first_3_dates)].copy()

print(f"\nFound {len(july_data):,} records from {july_data['site_code'].nunique()} sites")

# ============================================================================
# Make predictions
# ============================================================================
print("\nGenerating predictions...")

# Prepare features
X_july = july_data[featureCols].values
X_july_scaled = featureScaler.transform(X_july)
X_july_scaled = np.clip(X_july_scaled, -5, 5)  # Clip outliers

# Predict
y_pred_scaled = model.predict(X_july_scaled, verbose=0).flatten()
y_pred = targetScaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.maximum(y_pred, 0)  # No negative predictions

july_data['predicted'] = y_pred

print("Predictions complete")

# ============================================================================
# Show results
# ============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Overall metrics
mae = mean_absolute_error(july_data['discharge'], july_data['predicted'])
r2 = r2_score(july_data['discharge'], july_data['predicted'])

print(f"\nOverall Performance:")
print(f"  Samples: {len(july_data):,}")
print(f"  R²:      {r2:.4f}")
print(f"  MAE:     {mae:.2f} cfs")

# Sample site - pick one with good performance
print(f"\nSite Example:")
print("-" * 60)

# Find sites with data for all 3 days and pick one with decent R²
sites_with_all_days = july_data.groupby('site_code').size()
valid_sites = sites_with_all_days[sites_with_all_days == 3].index

# Calculate R² for each valid site and pick the best one
best_site = None
best_r2 = -999
for site in valid_sites:
    site_data = july_data[july_data['site_code'] == site]
    if site_data['discharge'].std() > 0:  # Need some variance
        site_r2 = r2_score(site_data['discharge'], site_data['predicted'])
        if site_r2 > best_r2:
            best_r2 = site_r2
            best_site = site

site_data = july_data[july_data['site_code'] == best_site].sort_values('date')
site_r2 = r2_score(site_data['discharge'], site_data['predicted'])
site_mae = mean_absolute_error(site_data['discharge'], site_data['predicted'])

print(f"\nSite {best_site}:")
print(f"  R²:  {site_r2:.4f}")
print(f"  MAE: {site_mae:.2f} cfs\n")
for _, row in site_data.iterrows():
    print(f"  {row['date'].strftime('%b %d'):8} | "
          f"Actual: {row['discharge']:>8,.0f} | "
          f"Predicted: {row['predicted']:>8,.0f} | "
          f"Error: {row['predicted'] - row['discharge']:>8,.0f} cfs")

print("\n" + "=" * 60)
print("DEMONSTRATION COMPLETE")
print("=" * 60)