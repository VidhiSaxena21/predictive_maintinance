import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print("Loading models and preparing data...")
# Load dataset
df = pd.read_csv("dataset.csv")
df = df.drop(['UDI', 'Product ID'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Re-fit scaler (normally this scaler should be saved with joblib during training and loaded here)
scaler = StandardScaler()
scaler.fit(X) # fit on all data for simulation purposes

try:
    autoencoder = tf.keras.models.load_model('autoencoder.keras')
    print("Autoencoder loaded.")
except Exception as e:
    print(f"Could not load autoencoder (perhaps it's not trained yet): {e}")
    autoencoder = None

try:
    mlp_model = tf.keras.models.load_model('mlp_model.keras')
    print("MLP Model loaded.")
except Exception as e:
    print(f"Could not load MLP model: {e}")
    mlp_model = None

# For this simulation, we'll pick a slice of the data that contains some failures to make it interesting
failure_indices = y[y == 1].index
if len(failure_indices) > 0:
    start_idx = max(0, failure_indices[0] - 10) # Start 10 rows before the first failure
    end_idx = start_idx + 20
else:
    start_idx = 0
    end_idx = 20

print("\n--- Starting Real-Time Inference Simulation ---")
print("Simulating sensor data stream...")

# We need a threshold for the autoencoder (we'd normally save/load this too)
# For the simulation, we'll set a dummy threshold or recalculate it quickly
if autoencoder:
    X_normal = X[y == 0].head(1000)
    X_normal_scaled = scaler.transform(X_normal)
    recon = autoencoder.predict(X_normal_scaled, verbose=0)
    train_mse = np.mean(np.power(X_normal_scaled - recon, 2), axis=1)
    threshold = np.percentile(train_mse, 95)
else:
    threshold = 0.5 # fallback dummy threshold

for i in range(start_idx, end_idx):
    # Simulate data arriving row by row
    current_row = X.iloc[i].to_frame().T
    actual_label = y.iloc[i]
    
    # Scale current row
    current_scaled = scaler.transform(current_row)
    
    print(f"\n[Time Step {i}] Receiving data... "
          f"Temp: {current_row['Air temperature [K]'].values[0]:.1f}K, "
          f"Speed: {current_row['Rotational speed [rpm]'].values[0]}rpm, "
          f"Torque: {current_row['Torque [Nm]'].values[0]:.1f}Nm")
    
    # 1. Anomaly Detection Phase
    is_anomaly = False
    if autoencoder:
        recon_curr = autoencoder.predict(current_scaled, verbose=0)
        mse = np.mean(np.power(current_scaled - recon_curr, 2), axis=1)[0]
        if mse > threshold:
            is_anomaly = True
            print(f"   --> ⚠️ ANOMALY DETECTED (Reconstruction Error: {mse:.4f} > {threshold:.4f})")
    
    # 2. Failure Prediction Phase
    if mlp_model:
        prob = mlp_model.predict(current_scaled, verbose=0)[0][0]
        pred_label = 1 if prob > 0.5 else 0
        
        status = "✅ Normal"
        if pred_label == 1:
            status = f"🚨 PREDICTING FAILURE (Confidence: {prob*100:.1f}%)"
            
        print(f"   --> MLP Status: {status}")
        
    print(f"   --> Actual State: {'Failure' if actual_label == 1 else 'Normal'}")
    
    time.sleep(1) # wait 1 second to simulate stream

print("\n--- Simulation Complete ---")
