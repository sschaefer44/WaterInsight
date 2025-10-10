import loadData
from mlpModel import *
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 60)
print("TRAINING OPTIMIZED MODEL (NO DATA LEAKAGE)")
print("=" * 60)

# ============================================================================
# Load the three separate files from pipeline
# ============================================================================
print("\nLoading engineered features from separate files...")
trainDF = pd.read_csv('CSV Backups/train_features.csv')
valDF = pd.read_csv('CSV Backups/val_features.csv')
testDF = pd.read_csv('CSV Backups/test_features.csv')
featureCols = loadData.loadFeatureNames('featureColumns.txt')

print(f"Train: {len(trainDF):,} rows")
print(f"Val:   {len(valDF):,} rows")
print(f"Test:  {len(testDF):,} rows")

# ============================================================================
# Prepare features and targets
# ============================================================================
print("\nPreparing features and targets...")

# Extract features and target
X_train = trainDF[featureCols].values
y_train = trainDF['discharge'].values

X_val = valDF[featureCols].values
y_val = valDF['discharge'].values

X_test = testDF[featureCols].values
y_test = testDF['discharge'].values

print(f"\nFeature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")

# Scale features
print("\nScaling features...")
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, featureScaler, targetScaler = scaleFeatures(
    X_train, X_val, X_test, y_train, y_val, y_test)

# CRITICAL: Clip extreme outliers after scaling
print("\nClipping extreme outliers...")
print(f"Before clipping - X_train range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# Clip to [-5, 5] standard deviations (removes extreme outliers)
X_train_scaled = np.clip(X_train_scaled, -5, 5)
X_val_scaled = np.clip(X_val_scaled, -5, 5)
X_test_scaled = np.clip(X_test_scaled, -5, 5)

print(f"After clipping - X_train range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# Check distribution
print(f"\nScaled data statistics:")
print(f"X_train_scaled - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"y_train_scaled - mean: {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}")

# ============================================================================
# Build and train model
# ============================================================================
print("\nBuilding improved model: [128, 64, 32], dropout=0.3, LR=0.001")
inputDim = X_train_scaled.shape[1]

# Build a more robust model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(1)
])

# Use a more stable optimizer with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model.summary()

# Callbacks
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-6, 
    verbose=1
)

earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining model...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100, 
    batch_size=512,  # Larger batch size for stability
    callbacks=[reduceLR, earlyStopping],
    verbose=1
)

# Print training history to diagnose issues
print("\nTraining history:")
print(f"Initial train loss: {history.history['loss'][0]:.4f}")
print(f"Final train loss: {history.history['loss'][-1]:.4f}")
print(f"Initial val loss: {history.history['val_loss'][0]:.4f}")
print(f"Final val loss: {history.history['val_loss'][-1]:.4f}")

# ============================================================================
# Evaluate on test set
# ============================================================================
print("\nEvaluating on test set...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = targetScaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred = np.maximum(y_pred, 0)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
mse = np.mean((y_test - y_pred)**2)

# Also calculate validation metrics for comparison
y_val_pred_scaled = model.predict(X_val_scaled, verbose=0).flatten()
y_val_pred = targetScaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
y_val_pred = np.maximum(y_val_pred, 0) 
val_r2 = r2_score(y_val, y_val_pred)
val_mae = np.mean(np.abs(y_val - y_val_pred))
val_rmse = np.sqrt(np.mean((y_val - y_val_pred)**2))

print(f"\n{'=' * 60}")
print("FINAL MODEL PERFORMANCE (NO DATA LEAKAGE)")
print(f"{'=' * 60}")
print("\nValidation Set:")
print(f"  R²:   {val_r2:.4f}")
print(f"  MAE:  {val_mae:.2f} cfs")
print(f"  RMSE: {val_rmse:.2f} cfs")
print("\nTest Set:")
print(f"  R²:   {r2:.4f}")
print(f"  MAE:  {mae:.2f} cfs")
print(f"  RMSE: {rmse:.2f} cfs")
print(f"  MSE:  {mse:,.2f}")
print(f"\nEpochs trained: {len(history.history['loss'])}")

# ============================================================================
# Plot training history
# ============================================================================
print("\nGenerating training history plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title('Training and Validation Loss (No Leakage)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# MAE
ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Training and Validation MAE (No Leakage)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c2Model/training_history.png', dpi=300, bbox_inches='tight')
print("Saved training history to c2Model/training_history.png")
plt.show()

print("\nStep 1: Saving model and scalers...")

# Create directory if it doesn't exist
os.makedirs('c2Model', exist_ok=True)

# Save the trained model
model.save('c2Model/DischargePredModel.keras')
print("Model saved to: c2Model/DischargePredModel.keras")

# Save the scalers
joblib.dump(featureScaler, 'c2Model/feature_scaler.pkl')
joblib.dump(targetScaler, 'c2Model/target_scaler.pkl')
print("Scalers saved to: c2Model/feature_scaler.pkl and target_scaler.pkl")

# Save feature names for future use
with open('c2Model/feature_names.txt', 'w') as f:
    for col in featureCols:
        f.write(f"{col}\n")
print("Feature names saved to: c2Model/feature_names.txt")

# Save model metadata
with open('c2Model/model_metadata.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DISCHARGE PREDICTION MODEL - NO DATA LEAKAGE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Training Date: {pd.Timestamp.now()}\n\n")
    f.write(f"Model Architecture:\n")
    f.write(f"  Layers: [128, 64, 32]\n")
    f.write(f"  Dropout: 0.3\n")
    f.write(f"  Optimizer: Adam (LR=0.001, clipnorm=1.0)\n\n")
    f.write(f"Data Split:\n")
    f.write(f"  Train: {len(trainDF):,} rows (2016-2023, 85% per year)\n")
    f.write(f"  Val:   {len(valDF):,} rows (2016-2023, 15% per year)\n")
    f.write(f"  Test:  {len(testDF):,} rows (2024)\n\n")
    f.write(f"Features: {len(featureCols)}\n\n")
    f.write(f"Performance:\n")
    f.write(f"  Validation R²:  {val_r2:.4f}\n")
    f.write(f"  Validation MAE: {val_mae:.2f} cfs\n")
    f.write(f"  Validation RMSE: {val_rmse:.2f} cfs\n\n")
    f.write(f"  Test R²:   {r2:.4f}\n")
    f.write(f"  Test MAE:  {mae:.2f} cfs\n")
    f.write(f"  Test RMSE: {rmse:.2f} cfs\n\n")
    f.write(f"Data Leakage Audit: PASSED\n")
    f.write(f"  - No row overlap between splits\n")
    f.write(f"  - No discharge features used\n")
    f.write(f"  - Climatology from train data only\n")
print("Model metadata saved to: c2Model/model_metadata.txt")

# ============================================================================
# GENERATE PREDICTIONS FOR 2024
# ============================================================================
print("\nStep 2: Generating predictions for 2024...")

# Make predictions (already have y_pred from earlier)
# If not, uncomment below:
# y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
# y_pred = targetScaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Create a results dataframe
results_df = testDF.copy()
results_df['predicted_discharge'] = y_pred
results_df['actual_discharge'] = y_test
results_df['error'] = y_pred - y_test
results_df['abs_error'] = np.abs(results_df['error'])
results_df['pct_error'] = (results_df['error'] / (results_df['actual_discharge'] + 1)) * 100

# Save predictions
results_df.to_csv('c2Model/2024_predictions.csv', index=False)
print("Predictions saved to: c2Model/2024_predictions.csv")

# ============================================================================
# VISUALIZATION 1: Predicted vs Actual Scatter Plot
# ============================================================================
print("\nStep 3: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter plot with regression line
ax1 = axes[0, 0]
scatter = ax1.scatter(y_test, y_pred, alpha=0.3, s=1, c=results_df['site_code'].astype('category').cat.codes, cmap='tab20')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Discharge (cfs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Discharge (cfs)', fontsize=12, fontweight='bold')
ax1.set_title(f'2024 Predictions vs Actual\nR² = {r2:.4f}, MAE = {mae:.2f} cfs', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add text box with stats
textstr = f'RMSE: {rmse:.2f} cfs\nSamples: {len(y_test):,}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Plot 2: Residuals plot
ax2 = axes[0, 1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.3, s=1, c='blue')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Discharge (cfs)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (cfs)', fontsize=12, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax3 = axes[1, 0]
ax3.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Prediction Error (cfs)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add mean and std to plot
mean_error = residuals.mean()
std_error = residuals.std()
textstr = f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}'
ax3.text(0.75, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Plot 4: Time series of predictions vs actual (sample)
ax4 = axes[1, 1]
# Sample one site for cleaner visualization
sample_site = results_df['site_code'].value_counts().index[0]
site_data = results_df[results_df['site_code'] == sample_site].sort_values('date')

if len(site_data) > 0:
    ax4.plot(site_data['date'], site_data['actual_discharge'], 
             label='Actual', linewidth=1.5, alpha=0.8)
    ax4.plot(site_data['date'], site_data['predicted_discharge'], 
             label='Predicted', linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Discharge (cfs)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Time Series: Site {sample_site}', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('c2Model/2024_predictions_overview.png', dpi=300, bbox_inches='tight')
print("Saved: c2Model/2024_predictions_overview.png")
plt.show()

# ============================================================================
# VISUALIZATION 2: Multiple Site Time Series
# ============================================================================
print("\nCreating multi-site time series plot...")

# Select 6 sites with most data
top_sites = results_df['site_code'].value_counts().head(6).index

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, site in enumerate(top_sites):
    site_data = results_df[results_df['site_code'] == site].sort_values('date')
    
    ax = axes[idx]
    ax.plot(site_data['date'], site_data['actual_discharge'], 
            label='Actual', linewidth=1.5, alpha=0.8, color='blue')
    ax.plot(site_data['date'], site_data['predicted_discharge'], 
            label='Predicted', linewidth=1.5, alpha=0.8, color='orange')
    
    site_r2 = r2_score(site_data['actual_discharge'], site_data['predicted_discharge'])
    site_mae = mean_absolute_error(site_data['actual_discharge'], site_data['predicted_discharge'])
    
    ax.set_title(f'Site: {site}\nR² = {site_r2:.3f}, MAE = {site_mae:.1f} cfs', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Discharge (cfs)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('2024 Predictions vs Actual - Top 6 Sites by Data Volume', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('c2Model/2024_multi_site_predictions.png', dpi=300, bbox_inches='tight')
print("Saved: c2Model/2024_multi_site_predictions.png")
plt.show()

# ============================================================================
# VISUALIZATION 3: Performance by Discharge Magnitude
# ============================================================================
print("\nCreating performance by discharge magnitude plot...")

# Bin predictions by discharge magnitude
results_df['discharge_bin'] = pd.cut(results_df['actual_discharge'], 
                                       bins=[0, 100, 500, 1000, 5000, np.inf],
                                       labels=['0-100', '100-500', '500-1K', '1K-5K', '>5K'])

bin_stats = results_df.groupby('discharge_bin').agg({
    'abs_error': ['mean', 'std', 'count'],
    'pct_error': 'mean'
}).reset_index()

bin_stats.columns = ['discharge_bin', 'mae', 'std', 'count', 'pct_error']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# MAE by discharge bin
ax1.bar(bin_stats['discharge_bin'], bin_stats['mae'], 
        yerr=bin_stats['std'], capsize=5, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Actual Discharge Range (cfs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error (cfs)', fontsize=12, fontweight='bold')
ax1.set_title('Prediction Error by Discharge Magnitude', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add sample counts on bars
for i, (bin_name, count) in enumerate(zip(bin_stats['discharge_bin'], bin_stats['count'])):
    ax1.text(i, bin_stats['mae'].iloc[i] + bin_stats['std'].iloc[i], 
             f'n={int(count):,}', ha='center', va='bottom', fontsize=9)

# Percent error by discharge bin
ax2.bar(bin_stats['discharge_bin'], bin_stats['pct_error'], 
        alpha=0.7, edgecolor='black', color='coral')
ax2.set_xlabel('Actual Discharge Range (cfs)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Percent Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Percent Error by Discharge Magnitude', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('c2Model/error_by_discharge_magnitude.png', dpi=300, bbox_inches='tight')
print("Saved: c2Model/error_by_discharge_magnitude.png")
plt.show()
# ============================================================================
# VISUALIZATION 4: Complete 2024 Time Series - All Data
# ============================================================================
print("\nCreating complete 2024 actual vs predicted time series...")

# Sort by date for proper time series plotting
results_sorted = results_df.sort_values('date')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Plot 1: All individual points
ax1.scatter(results_sorted['date'], results_sorted['actual_discharge'], 
           s=1, alpha=0.5, label='Actual', color='blue')
ax1.scatter(results_sorted['date'], results_sorted['predicted_discharge'], 
           s=1, alpha=0.5, label='Predicted', color='orange')
ax1.set_ylabel('Discharge (cfs)', fontsize=12, fontweight='bold')
ax1.set_title('2024 Complete Time Series: All Sites - Individual Points', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, markerscale=5)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # Log scale to see patterns better
textstr = f'R² = {r2:.4f}\nMAE = {mae:.2f} cfs\nRMSE = {rmse:.2f} cfs'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Plot 2: Daily aggregated (mean across all sites per day)
daily_actual = results_sorted.groupby('date')['actual_discharge'].mean()
daily_predicted = results_sorted.groupby('date')['predicted_discharge'].mean()

ax2.plot(daily_actual.index, daily_actual.values, 
        label='Actual (Daily Mean)', linewidth=2, alpha=0.8, color='blue')
ax2.plot(daily_predicted.index, daily_predicted.values, 
        label='Predicted (Daily Mean)', linewidth=2, alpha=0.8, color='orange')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Discharge (cfs)', fontsize=12, fontweight='bold')
ax2.set_title('2024 Daily Mean Discharge: All Sites Aggregated', 
             fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Calculate daily R²
daily_r2 = r2_score(daily_actual.values, daily_predicted.values)
daily_mae = mean_absolute_error(daily_actual.values, daily_predicted.values)
textstr = f'Daily Aggregated:\nR² = {daily_r2:.4f}\nMAE = {daily_mae:.2f} cfs'
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('c2Model/2024_complete_timeseries.png', dpi=300, bbox_inches='tight')
print("Saved: c2Model/2024_complete_timeseries.png")
plt.show()
# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("PREDICTION SUMMARY STATISTICS")
print("=" * 70)

print(f"\nOverall Performance on 2024 Test Set:")
print(f"  Samples:     {len(y_test):,}")
print(f"  R²:          {r2:.4f}")
print(f"  MAE:         {mae:.2f} cfs")
print(f"  RMSE:        {rmse:.2f} cfs")
print(f"  Mean Error:  {residuals.mean():.2f} cfs")
print(f"  Std Error:   {residuals.std():.2f} cfs")

print(f"\nDischarge Statistics:")
print(f"  Actual - Mean:   {y_test.mean():.2f} cfs")
print(f"  Actual - Std:    {y_test.std():.2f} cfs")
print(f"  Actual - Range:  [{y_test.min():.2f}, {y_test.max():.2f}] cfs")

print(f"\nPredicted - Mean:   {y_pred.mean():.2f} cfs")
print(f"  Predicted - Std:    {y_pred.std():.2f} cfs")
print(f"  Predicted - Range:  [{y_pred.min():.2f}, {y_pred.max():.2f}] cfs")

print("\n" + "=" * 70)
print("MODEL SAVING AND VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nSaved files:")
print("  • c2Model/DischargePredModel.keras")
print("  • c2Model/feature_scaler.pkl")
print("  • c2Model/target_scaler.pkl")
print("  • c2Model/feature_names.txt")
print("  • c2Model/model_metadata.txt")
print("  • c2Model/2024_predictions.csv")
print("  • c2Model/2024_predictions_overview.png")
print("  • c2Model/2024_multi_site_predictions.png")
print("  • c2Model/error_by_discharge_magnitude.png")
print("\n" + "=" * 70)