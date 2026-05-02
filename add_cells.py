import nbformat
import json

notebook_path = "main.ipynb"

# Load the existing notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

new_cells = []

# Phase 1: MLP
new_cells.append(nbformat.v4.new_markdown_cell("## Phase 1: Neural Network (MLP)"))
new_cells.append(nbformat.v4.new_code_cell("""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Neural networks require scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# Build MLP Model
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classification
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train MLP
print("Training MLP Model...")
history = mlp_model.fit(X_train_scaled, y_train_sm, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
mlp_loss, mlp_acc = mlp_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"MLP Test Accuracy: {mlp_acc:.4f}")

# Save MLP model
mlp_model.save('mlp_model.keras')
"""))

# Phase 2: Autoencoder
new_cells.append(nbformat.v4.new_markdown_cell("## Phase 2: Anomaly Detection (Autoencoder)"))
new_cells.append(nbformat.v4.new_code_cell("""
import numpy as np

# Filter only "normal" data for training the autoencoder (Machine failure == 0)
X_train_normal = X_train[y_train == 0]
X_train_normal_scaled = scaler.transform(X_train_normal) # use the same scaler

# Build Autoencoder
input_dim = X_train_normal_scaled.shape[1]
autoencoder = Sequential([
    # Encoder
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    # Decoder
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
print("Training Autoencoder on normal data...")
autoencoder.fit(X_train_normal_scaled, X_train_normal_scaled, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Save Autoencoder
autoencoder.save('autoencoder.keras')

# Predict and calculate reconstruction error on test set
X_test_recon = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - X_test_recon, 2), axis=1)

# Define an anomaly threshold (e.g., 95th percentile of training normal errors)
train_recon = autoencoder.predict(X_train_normal_scaled)
train_mse = np.mean(np.power(X_train_normal_scaled - train_recon, 2), axis=1)
threshold = np.percentile(train_mse, 95)
print(f"Anomaly Detection Threshold (MSE): {threshold:.4f}")

# Flag anomalies
anomalies = mse > threshold
print(f"Number of anomalies detected in test set: {np.sum(anomalies)} out of {len(anomalies)}")
"""))

# Phase 3: LSTM
new_cells.append(nbformat.v4.new_markdown_cell("## Phase 3: Sequence Modeling (LSTM) for Advance Prediction\n*Note: Since the dataset is tabular, we synthesize a sliding window to simulate time-series data.*"))
new_cells.append(nbformat.v4.new_code_cell("""
# Synthesize sequential data
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5
X_seq, y_seq = create_sequences(X, y, time_steps)

# Split sequential data
from sklearn.model_selection import train_test_split
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False) # Important: shuffle=False for time series!

# Scale sequential data (flatten, scale, reshape back)
num_samples_train, seq_len, num_features = X_train_seq.shape
X_train_seq_flat = X_train_seq.reshape(-1, num_features)
X_train_seq_scaled_flat = scaler.fit_transform(X_train_seq_flat)
X_train_seq_scaled = X_train_seq_scaled_flat.reshape(num_samples_train, seq_len, num_features)

num_samples_test = X_test_seq.shape[0]
X_test_seq_flat = X_test_seq.reshape(-1, num_features)
X_test_seq_scaled_flat = scaler.transform(X_test_seq_flat)
X_test_seq_scaled = X_test_seq_scaled_flat.reshape(num_samples_test, seq_len, num_features)


from tensorflow.keras.layers import LSTM

# Build LSTM
lstm_model = Sequential([
    LSTM(32, activation='relu', input_shape=(seq_len, num_features)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM
print("Training LSTM Model...")
# Note: dealing with class imbalance is harder here, but we will train it simply for demonstration
lstm_model.fit(X_train_seq_scaled, y_train_seq, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_seq_scaled, y_test_seq, verbose=0)
print(f"LSTM Test Accuracy: {lstm_acc:.4f}")

# Save LSTM
lstm_model.save('lstm_model.keras')
"""))

# Append new cells
nb.cells.extend(new_cells)

# Write back to notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully appended new cells to notebook!")
