import loadData
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import joblib
from mlpModel import *

print("-" * 15)
print("MLP ARCHITECTURE EXPERIMENTATION")
print("-" * 15)

# =========== LOAD DATA ===========

print("\nLoading Engineered Data")

df = loadData.loadModelReadyData('CSV Backups/engineeredFeatures.csv')
featureCols = loadData.loadFeatureNames('CSV Backups/featureColumns.txt')

trainDF, valDF, testDF = loadData.balancedTimeSeriesSplit(df, 'discharge')
X_train, y_train, X_val, y_val, X_test, y_test = loadData.prepareDataForTraining(trainDF, valDF, testDF, featureCols, 'discharge')


# Scale features
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, featureScaler, targetScaler = scaleFeatures(
    X_train, X_val, X_test, y_train, y_val, y_test)



# =========== EXPERIMENT FUNCTION ===========
def trainAndEval(arch, dropoutRate, learningRate, epochs, batchSize):
    """Train a model with given architecture and hyperparameters => return metrics"""

    inputDim = X_train_scaled.shape[1]
    model = buildMLP(inputDim, arch, dropoutRate, learningRate)

    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 20,
        restore_best_weights = True,
        verbose = 0
    )

    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,
        patience = 3,
        min_lr = 1e-6,
        verbose = 0
    )

    startTime = time.time()

    history = model.fit(
        X_train_scaled, y_train_scaled, validation_data = (X_val_scaled, y_val_scaled),
        epochs = epochs, batch_size = batchSize, callbacks = [earlyStopping, reduceLR],
        verbose = 0
    )

    trainingTime = time.time() - startTime

    y_pred_scaled = model.predict(X_test_scaled, verbose = 0).flatten()
    y_pred = targetScaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mse = np.mean((y_test - y_pred)**2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)

    bestValLoss = min(history.history['val_loss'])
    epochsStopped = len(history.history['loss'])

    print(f"    Completed in {trainingTime/60:.2f} minutes ({epochsStopped} epochs)")
    print(f"    Test MAE: {mae:.2f} cfs, RMSE: {rmse:.2f} cfs")

    return {
        'architecture': str(arch),
        'dropout': dropoutRate,
        'learning_rate': learningRate,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_mse': mse,
        'best_val_loss': bestValLoss,
        'epochs_trained': epochsStopped,
        'training_time_min': trainingTime/60,
        'total_params': model.count_params()
    }

# =========== RUN EXPERIMENTS ===========
print("\nRunning Experiments")
experiments = []

# Baseline Model
print("\n1. Baseline [128, 64]")
experiments.append(trainAndEval([128, 64], 0.2, 0.0005, 50, 256))  # Changed

# Experiment with architecture size
print("\n2. Smaller [64, 32]")
experiments.append(trainAndEval([64, 32], 0.2, 0.0005, 50, 256))  # Changed

print("\n3. Larger [256, 128, 64]")
experiments.append(trainAndEval([256, 128, 64], 0.2, 0.0005, 50, 256))  # Changed

print("\n4. Deeper [128, 64, 32]")
experiments.append(trainAndEval([128, 64, 32], 0.2, 0.0005, 50, 256))  # Changed

# Experiment with dropout rate
print("\n5. Higher Dropout [128, 64], dropoutRate = 0.3")
experiments.append(trainAndEval([128, 64], 0.3, 0.0005, 50, 256))  # Changed

print("\n6. Lower Dropout [128, 64], dropout = 0.1")
experiments.append(trainAndEval([128, 64], 0.1, 0.0005, 50, 256))  # Changed

# Experiment with learning rate - keep these as they are testing different LRs
print("\n7. Higher Learning Rate [128, 64], LR = 0.01")
experiments.append(trainAndEval([128, 64], 0.2, 0.01, 50, 256))

print("\n8. Lower Learning Rate [128, 64], LR = 0.0001")
experiments.append(trainAndEval([128, 64], 0.2, 0.0001, 50, 256))

# =========== ANALYZE RESULTS ===========
resultsDF = pd.DataFrame(experiments)
resultsDF = resultsDF.sort_values('test_mae').reset_index(drop = True)

print("-" * 15)
print("EXPERIMENT RESULTS (SORTED BY MAE)")
print("-" * 15)
print(resultsDF)

resultsDF.to_csv('Experiments/architecture_experiment_results.csv', index=False)
print("Results Saved to Experiments/architecture_experiment_results.csv")

# =========== VISUALIZE COMPARISON ===========
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

x = range(len(resultsDF))
labels = [f"Exp {i+1}" for i in range(len(resultsDF))]


ax1.bar(x, resultsDF['test_mae'], color='steelblue')
ax1.set_xlabel('Experiment')
ax1.set_ylabel('Test MAE (cfs)')
ax1.set_title('Test MAE by Configuration', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# RMSE comparison
ax2.bar(x, resultsDF['test_rmse'], color='coral')
ax2.set_xlabel('Experiment')
ax2.set_ylabel('Test RMSE (cfs)')
ax2.set_title('Test RMSE by Configuration', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Training time comparison
ax3.bar(x, resultsDF['training_time_min'], color='green', alpha=0.7)
ax3.set_xlabel('Experiment')
ax3.set_ylabel('Training Time (minutes)')
ax3.set_title('Training Time by Configuration', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Model complexity vs performance
ax4.scatter(resultsDF['total_params'], resultsDF['test_mae'], s=100, alpha=0.6)
for i, row in resultsDF.iterrows():
    ax4.annotate(f"Exp {i+1}", (row['total_params'], row['test_mae']), 
                fontsize=8, ha='right')
ax4.set_xlabel('Total Parameters')
ax4.set_ylabel('Test MAE (cfs)')
ax4.set_title('Model Complexity vs Performance', fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Graphs/architecture_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comparison plot to Graphs/architecture_comparison.png")
plt.show()

# =========== SUMMARY ===========
print("-" * 15)
print("BEST MODEL")
print("-" * 15)
best = resultsDF.iloc[0]
print(f"Architecture: {best['architecture']}")
print(f"Dropout: {best['dropout']}")
print(f"Learning Rate: {best['learning_rate']}")
print(f"Test MAE: {best['test_mae']:.2f} cfs")
print(f"Test RMSE: {best['test_rmse']:.2f} cfs")
print(f"Parameters: {best['total_params']:,}")
print(f"Training Time: {best['training_time_min']:.2f} minutes")

improvement = ((experiments[0]['test_mae'] - best['test_mae']) / experiments[0]['test_mae']) * 100
print(f"\nImprovement over baseline: {improvement:.1f}%")
print("-" * 15)