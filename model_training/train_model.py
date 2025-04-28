import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(datapath):
    """
    Load dataset from path and display basic info
    """
    data = pd.read_csv(datapath)
    print(f'Loaded data from {datapath}, Shape: {data.shape}')
    return data

def preprocess_data(train_data, test_data=None):
    """
    Preprocess data by creating additional features
    """
    # Make copies to avoid modifying original data
    train_processed = train_data.copy()
    
    # Convert date to datetime
    train_processed['date'] = pd.to_datetime(train_processed['date'])
    
    # Extract date features
    train_processed['month'] = train_processed['date'].dt.month
    train_processed['day'] = train_processed['date'].dt.dayofweek
    train_processed['year'] = train_processed['date'].dt.year
    train_processed['day_of_month'] = train_processed['date'].dt.day
    
    # If we have test data, apply the same transformations
    if test_data is not None:
        test_processed = test_data.copy()
        test_processed['date'] = pd.to_datetime(test_processed['date'])
        test_processed['month'] = test_processed['date'].dt.month
        test_processed['day'] = test_processed['date'].dt.dayofweek
        test_processed['year'] = test_processed['date'].dt.year
        test_processed['day_of_month'] = test_processed['date'].dt.day
        return train_processed, test_processed
    
    return train_processed

def prepare_training_data(train_data):
    """
    Prepare data for training
    """
    # Identify features to use
    features = ['store', 'item', 'month', 'day', 'year', 'day_of_month']
    target = 'sales'
    
    # Split training data
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_data[features],
        train_data[target], 
        test_size=0.2, 
        random_state=2018
    )
    
    return train_x, valid_x, train_y, valid_y, features

def train_lgb_model(train_x, train_y, valid_x, valid_y, features):
    """
    Train LightGBM model
    """
    params = {
        'nthread': 4,
        'max_depth': 5,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mape',
        'num_leaves': 64,
        'learning_rate': 0.2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
    }
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(valid_x, valid_y)
    
    # Train model
    print("Training model...")
    model = lgb.train(
        params, 
        lgb_train, 
        num_boost_round=1000,  # Reduced for quicker training
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50, 
        verbose_eval=100
    )
    
    # Evaluate model
    valid_pred = model.predict(valid_x)
    mae = mean_absolute_error(valid_y, valid_pred)
    rmse = np.sqrt(mean_squared_error(valid_y, valid_pred))
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('/app/models/feature_importance.png')
    plt.close()
    
    return model

def save_model(model, features, output_dir='/app/models'):
    """
    Save the trained model and feature list
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'demand_forecasting_model.joblib')
    joblib.dump(model, model_path)
    
    # Save the feature list
    features_path = os.path.join(output_dir, 'features.joblib')
    joblib.dump(features, features_path)
    
    print(f"Model saved to {model_path}")
    print(f"Features saved to {features_path}")

def main():
    # Load data
    print("Loading data...")
    train_df = load_data('/data/train.csv')
    
    # Preprocess data
    print("Preprocessing data...")
    train_df_processed = preprocess_data(train_df)
    
    # Prepare training data
    print("Preparing training data...")
    train_x, valid_x, train_y, valid_y, features = prepare_training_data(train_df_processed)
    
    # Train model
    print("Training model...")
    model = train_lgb_model(train_x, train_y, valid_x, valid_y, features)
    
    # Save model
    save_model(model, features)
    
    print("Done!")

if __name__ == "__main__":
    main()

    