import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    # simple rmse calculation
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    # simple mae calculation
    return mean_absolute_error(y_true, y_pred)

def drop_constant_columns(train_data, test_data):
    # find columns that never change
    constant_cols = [c for c in train_data.columns if train_data[c].std() < 0.00001]
    
    # drop them from both datasets
    train_data = train_data.drop(columns=constant_cols)
    test_data = test_data.drop(columns=constant_cols, errors='ignore')
    
    return train_data, test_data

def remove_highly_correlated_features(train_data, test_data, exclude_cols):
    # get features to check
    features = [c for c in train_data.columns if c not in exclude_cols]
    
    # find correlations
    corr_matrix = train_data[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [c for c in upper.columns if any(upper[c] > 0.9)]
    
    # drop them from both datasets
    train_data = train_data.drop(columns=highly_correlated)
    test_data = test_data.drop(columns=highly_correlated, errors='ignore')
    
    return train_data, test_data
