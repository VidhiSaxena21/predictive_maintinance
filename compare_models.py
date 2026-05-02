import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

print("Loading dataset for evaluation...")
df = pd.read_csv("dataset.csv")
df = df.drop(['UDI', 'Product ID'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

report_lines = []
report_lines.append("="*50)
report_lines.append(" PREDICTIVE MAINTENANCE MODEL COMPARISON REPORT")
report_lines.append("="*50 + "\n")

# 1. Random Forest Evaluation
try:
    rf_model = joblib.load("model.pkl")
    rf_pred = rf_model.predict(X_test)
    report_lines.append("--- 1. RANDOM FOREST (Original Baseline) ---")
    report_lines.append(classification_report(y_test, rf_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, rf_pred).ravel()
    report_lines.append(f"Confusion Matrix: True Negatives={tn}, False Positives={fp}, False Negatives={fn}, True Positives={tp}\n")
except Exception as e:
    report_lines.append(f"Could not load Random Forest: {e}\n")

# 2. MLP Evaluation
try:
    mlp_model = tf.keras.models.load_model("mlp_model.keras")
    mlp_prob = mlp_model.predict(X_test_scaled, verbose=0)
    mlp_pred = (mlp_prob > 0.5).astype(int)
    report_lines.append("--- 2. NEURAL NETWORK (MLP with Class Weights) ---")
    report_lines.append(classification_report(y_test, mlp_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, mlp_pred).ravel()
    report_lines.append(f"Confusion Matrix: True Negatives={tn}, False Positives={fp}, False Negatives={fn}, True Positives={tp}\n")
except Exception as e:
    report_lines.append(f"Could not load MLP: {e}\n")

# 3. Autoencoder Evaluation (Anomaly Detection)
try:
    autoencoder = tf.keras.models.load_model("autoencoder.keras")
    # Threshold defined from train set
    X_train_normal = X_train[y_train == 0]
    X_train_normal_scaled = scaler.transform(X_train_normal)
    train_recon = autoencoder.predict(X_train_normal_scaled, verbose=0)
    train_mse = np.mean(np.power(X_train_normal_scaled - train_recon, 2), axis=1)
    threshold = np.percentile(train_mse, 95)
    
    test_recon = autoencoder.predict(X_test_scaled, verbose=0)
    test_mse = np.mean(np.power(X_test_scaled - test_recon, 2), axis=1)
    auto_pred = (test_mse > threshold).astype(int)
    
    report_lines.append(f"--- 3. AUTOENCODER (Unsupervised Anomaly Detection) ---")
    report_lines.append(f"Using Reconstruction Error Threshold: {threshold:.4f}\n")
    report_lines.append(classification_report(y_test, auto_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, auto_pred).ravel()
    report_lines.append(f"Confusion Matrix: True Negatives={tn}, False Positives={fp}, False Negatives={fn}, True Positives={tp}\n")
except Exception as e:
    report_lines.append(f"Could not load Autoencoder: {e}\n")

# 4. LSTM Evaluation
try:
    def create_sequences(X, y, time_steps=5):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X.iloc[i:(i + time_steps)].values)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(X, y, 5)
    _, X_test_seq, _, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)
    
    num_samples_test = X_test_seq.shape[0]
    seq_len = X_test_seq.shape[1]
    num_features = X_test_seq.shape[2]
    X_test_seq_flat = X_test_seq.reshape(-1, num_features)
    X_test_seq_scaled_flat = scaler.transform(X_test_seq_flat) # Note: this scaler was fit on non-sequential train set, which is close enough for demonstration
    X_test_seq_scaled = X_test_seq_scaled_flat.reshape(num_samples_test, seq_len, num_features)
    
    lstm_model = tf.keras.models.load_model("lstm_model.keras")
    lstm_prob = lstm_model.predict(X_test_seq_scaled, verbose=0)
    lstm_pred = (lstm_prob > 0.5).astype(int)
    
    report_lines.append("--- 4. LSTM (Sequence Forecasting) ---")
    report_lines.append(classification_report(y_test_seq, lstm_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test_seq, lstm_pred).ravel()
    report_lines.append(f"Confusion Matrix: True Negatives={tn}, False Positives={fp}, False Negatives={fn}, True Positives={tp}\n")
except Exception as e:
    report_lines.append(f"Could not load LSTM: {e}\n")

# Write report to file
with open("model_comparison_report.txt", "w") as f:
    f.write("\n".join(report_lines))

print("Report generated successfully as 'model_comparison_report.txt'.")
