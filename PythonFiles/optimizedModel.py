import loadData
from mlpModel import *
import numpy as np
import random
from sklearn.metrics import r2_score

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 60)
print("TRAINING AND SAVING OPTIMIZED MODEL")
print("=" * 60)

# Load data with balanced split
df = loadData.loadModelReadyData('CSV Backups/engineeredFeatures.csv')
featureCols = loadData.loadFeatureNames('CSV Backups/featureColumns.txt')
trainDF, valDF, testDF = loadData.balancedTimeSeriesSplit(df, 'discharge')
X_train, y_train, X_val, y_val, X_test, y_test = loadData.prepareDataForTraining(
    trainDF, valDF, testDF, featureCols, 'discharge')

# Scale features
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, featureScaler, targetScaler = scaleFeatures(
    X_train, X_val, X_test, y_train, y_val, y_test)

# Build optimal model: [64, 32], dropout=0.2, LR=0.0005
print("\nBuilding optimal model: [64, 32], dropout=0.2, LR=0.0005")
inputDim = X_train_scaled.shape[1]
model = buildMLP(inputDim, [64, 32], 0.2, 0.0005)

# Train
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)


print("\nTraining model...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=50, batch_size=256,
    callbacks=[reduceLR],
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = targetScaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
mse = np.mean((y_test - y_pred)**2)

print(f"\n{'=' * 60}")
print("FINAL MODEL PERFORMANCE")
print(f"{'=' * 60}")
print(f"Test R²:   {r2:.4f}")
print(f"Test MAE:  {mae:.2f} cfs")
print(f"Test RMSE: {rmse:.2f} cfs")
print(f"Test MSE:  {mse:,.2f}")
print(f"Epochs trained: {len(history.history['loss'])}")

# Save model and scalers
model.save('FinalModel/optimized_discharge_model.keras')
print("\nModel saved to FinalModel/optimized_discharge_model.keras")

joblib.dump(featureScaler, 'FinalModel/feature_scaler.pkl')
joblib.dump(targetScaler, 'FinalModel/target_scaler.pkl')

print("Scalers saved to FinalModel/feature_scaler.pkl and FinalModel/target_scaler.pkl")

print(f"\n{'=' * 60}")
print("MODEL TRAINING AND SAVING COMPLETE")
print(f"{'=' * 60}")

# Plot training history
print("\nGenerating training history plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# MAE
ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('FinalModel/optimized_model_training_history.png', dpi=300, bbox_inches='tight')
print("Saved training history to FinalModel/optimized_model_training_history.png")
plt.show()