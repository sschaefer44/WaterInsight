import loadData
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time 
import joblib

print("-" * 15)
print("MLP Model for Discharge (cf/s)")
print("-" * 15)

# =========== LOAD DATA ===========
print(f"\nLoading engineered data")
df = loadData.loadModelReadyData('CSV Backups/engineeredFeatures.csv')
featureCols = loadData.loadFeatureNames('CSV Backups/featureColumns.txt')

trainDF, valDF, testDF = loadData.timeSeriesTrainTestSplit(df, 'discharge')

# Split data into X and Y sets for train/val/test
X_train, y_train, X_val, y_val, X_test, y_test = loadData.prepareDataForTraining(trainDF, valDF, testDF, featureCols, 'discharge')


def scaleFeatures(X_train, X_val, X_test, y_train, y_val, y_test):
    """Standardize scaling of X sets. Mean = 0, STD = 1"""

    print(f"\nScaling Features")

    featureScaler = StandardScaler()
    X_train_scaled = featureScaler.fit_transform(X_train)  # Fit on train
    X_val_scaled = featureScaler.transform(X_val)          # Transform val
    X_test_scaled = featureScaler.transform(X_test)        # Transform test
    
    # Scale target - separate scaler
    targetScaler = StandardScaler()
    y_train_scaled = targetScaler.fit_transform(y_train.reshape(-1, 1)).flatten()  # Fit on train
    y_val_scaled = targetScaler.transform(y_val.reshape(-1, 1)).flatten()          # Transform val
    y_test_scaled = targetScaler.transform(y_test.reshape(-1, 1)).flatten()        # Transform test
    
    print("Scaling complete")

    joblib.dump(featureScaler, 'feature_scaler.pkl')
    joblib.dump(targetScaler, 'target_scaler.pkl')
    
    print("Saved scalers")
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, featureScaler, targetScaler

# Scale
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, featureScaler, targetScaler = scaleFeatures(X_train, X_val, X_test, y_train, y_val, y_test)

# =========== BUILD MODEL ===========
def buildMLP(inputDim, hiddenLayers, dropoutRate, LR):

    """
        Build MLP model for discharge prediction
    
    Parameters:
    -----------
    inputDim : int
        Number of input features
    hiddenLayers : list
        Number of neurons in each hidden layer
    dropoutRate : float
        Dropout rate for regularization
    learningRate : float
        Learning rate for Adam optimizer
    """

    print("-" * 15)
    print(f"Building MLP")
    print("-" * 15)
    print(f"Architecture: {hiddenLayers}")
    print(f"Dropout Rate: {dropoutRate}")
    print(f"Learning Rate: {LR}")

    model = tf.keras.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Input(shape=(inputDim,)))

    # Add Hidden Layers
    for i, units in enumerate(hiddenLayers):
        model.add(tf.keras.layers.Dense(units, activation = 'relu', name=f'hidden{i+1}'))
        model.add(tf.keras.layers.Dropout(dropoutRate, name = f'dropout{i+1}'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1, name='output'))

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = LR),
        loss = 'mse',
        metrics = ['mae']
    )

    print("\nModel summary:")
    model.summary()
    
    return model

inputDim = X_train_scaled.shape[1]
MLP = buildMLP(inputDim, [128, 64], dropoutRate=0.2, LR = 0.001)

# =========== TRAIN MODEL ===========
def trainModel(model, X_train, y_train, X_val, y_val, epochs, batchSize):
    """Train MLP model"""
    print("-" * 15)
    print(f"Training MLP")
    print("-" * 15)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batchSize}")

    # Callbacks
    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        restore_best_weights=True,
        verbose=1
    )

    reducleLR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor = 0.5,
        patience = 5,
        min_lr = 1e-6,
        verbose = 1
    )

    startTime = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs = epochs,
        batch_size = batchSize,
        callbacks = [earlyStopping, reducleLR],
        verbose = 1
    )

    elapsed = time.time() - startTime
    print(f"\nTraining Completed in {elapsed/60:.2f} Minutes")
    return history

history = trainModel(MLP, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 50, 256)

# =========== EVAL. AND VISUALIZATION ===========

def plotTrainingHistory(history):
    """Plot loss curves"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth = 2)
    ax1.plot(history.history['val_loss'], label = 'Val Loss', linewidth = 2)
    ax1.set_xlabel('Epoch', fontsize = 12)
    ax1.set_ylabel('MSE Loss', fontsize = 12)
    ax1.set_title('Training and Validation Loss', fontsize = 14, fontweight = 'bold')
    ax1.legend(fontsize = 11)
    ax1.grid(True, alpha = 0.3)

    # MAE
    ax2.plot(history.history['mae'], label = 'Train MAE', linewidth = 2)
    ax2.plot(history.history['val_mae'], label = 'Val MAE', linewidth = 2)
    ax2.set_xlabel('Epoch', fontsize = 12)
    ax2.set_ylabel('MAE', fontsize = 12)
    ax2.set_title('Training and Validation MAE', fontsize = 14, fontweight = 'bold')
    ax2.legend(fontsize = 11)
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig('Graphs/trainingHistory.png', dpi = 300, bbox_inches = 'tight')
    print("Saved training history plot to Graphs/trainingHistory.png")
    plt.show()

def evalModel(model, X_test, y_test_original, targetScaler):
    """Evaluate model on test data set"""
    print("-" * 15)
    print("Evaluating Model")
    print("-" * 15)

    # Predict on scaled data
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Unscale predictions back to original units
    y_pred = targetScaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics in original units (cfs)
    mse = np.mean((y_test_original - y_pred)**2)
    mae = np.mean(np.abs(y_test_original - y_pred))
    rmse = np.sqrt(mse)
    
    print(f"\nTest Metrics (original units - cfs):")
    print(f"  MSE:  {mse:,.2f}")
    print(f"  MAE:  {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    
    return y_pred


y_pred_test = evalModel(MLP, X_test_scaled, y_test, targetScaler)

plotTrainingHistory(history)

# Save model
MLP.save('SavedModels/discharge_mlp_model.keras')
print("\nModel saved to discharge_mlp_model.keras")

print('-' * 15)
print("MODEL TRAINING COMPLETE")
print('-' * 15)