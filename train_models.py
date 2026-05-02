import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

print("Loading dataset...")
df = pd.read_csv("dataset.csv")
df = df.drop(['UDI', 'Product ID'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights for imbalance
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
print(f"Computed Class Weights: {class_weights_dict}")

# 1. Train MLP with Class Weights
print("\n--- Training MLP ---")
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, 
              class_weight=class_weights_dict, verbose=0)
mlp_model.save('mlp_model.keras')
print("MLP Model saved.")

# 2. Train Autoencoder (Only on healthy data)
print("\n--- Training Autoencoder ---")
X_train_normal = X_train[y_train == 0]
X_train_normal_scaled = scaler.transform(X_train_normal)

autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(X_train_scaled.shape[1], activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_normal_scaled, X_train_normal_scaled, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
autoencoder.save('autoencoder.keras')
print("Autoencoder saved.")

# 3. Train LSTM
print("\n--- Training LSTM ---")
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, 5)
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

num_samples_train, seq_len, num_features = X_train_seq.shape
X_train_seq_flat = X_train_seq.reshape(-1, num_features)
X_train_seq_scaled_flat = scaler.fit_transform(X_train_seq_flat)
X_train_seq_scaled = X_train_seq_scaled_flat.reshape(num_samples_train, seq_len, num_features)

# Recompute class weights for sequential train set
cw_seq_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_seq), y=y_train_seq)
cw_seq_dict = {0: cw_seq_arr[0], 1: cw_seq_arr[1]}

lstm_model = Sequential([
    LSTM(32, activation='relu', input_shape=(seq_len, num_features)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_seq_scaled, y_train_seq, epochs=10, batch_size=32, validation_split=0.2, 
               class_weight=cw_seq_dict, verbose=0)
lstm_model.save('lstm_model.keras')
print("LSTM saved.")

print("\nAll models trained and saved successfully using Class Weights!")
