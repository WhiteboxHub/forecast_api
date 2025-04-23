
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import joblib
# from datetime import datetime

# def load_data(train_path):
#     """Load and preprocess the training data"""
#     train_df = pd.read_csv(train_path)
#     train_df['date'] = pd.to_datetime(train_df['date'])
    
#     # Feature engineering
#     train_df['month'] = train_df['date'].dt.month
#     train_df['day'] = train_df['date'].dt.dayofweek
#     train_df['year'] = train_df['date'].dt.year
    
#     return train_df

# def train_model(train_df, model_path='model/sales_model.pkl'):
#     """Train and save the LightGBM model"""
#     # Prepare features and target
#     cols = [col for col in train_df.columns if col not in ['date', 'sales']]
#     X = train_df[cols]
#     y = train_df['sales']
    
#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=2018
#     )
    
#     # Model parameters (from your original code)
#     params = {
#         'nthread': 10,
#         'max_depth': 5,
#         'boosting_type': 'gbdt',
#         'objective': 'regression_l1',
#         'metric': 'mape',
#         'num_leaves': 64,
#         'learning_rate': 0.2,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'lambda_l1': 3.097,
#         'lambda_l2': 2.948,
#         'verbose': 1,
#         'min_child_weight': 6.996,
#         'min_split_gain': 0.037,
#     }
    
#     # Train model
#     lgb_train = lgb.Dataset(X_train, y_train)
#     lgb_valid = lgb.Dataset(X_val, y_val)
    
#     model = lgb.train(
#         params,
#         lgb_train,
#         3000,
#         valid_sets=[lgb_train, lgb_valid],
#         early_stopping_rounds=50,
#         verbose_eval=50
#     )
    
#     # Save model
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")
    
#     return model

# if __name__ == "__main__":
#     # Load data - adjust the path to your dataset
#     train_df = load_data("data/train.csv")
    
#     # Train and save model
#     model = train_model(train_df)





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import joblib
# from datetime import datetime

# def load_data(train_path):
#     """Load and preprocess the training data"""
#     train_df = pd.read_csv(train_path)
#     train_df['date'] = pd.to_datetime(train_df['date'])
    
#     # Feature engineering
#     train_df['month'] = train_df['date'].dt.month
#     train_df['day'] = train_df['date'].dt.dayofweek
#     train_df['year'] = train_df['date'].dt.year
    
#     return train_df

# def train_model(train_df, model_path='model/sales_model.pkl'):
#     """Train and save the LightGBM model"""
#     # Prepare features and target
#     cols = [col for col in train_df.columns if col not in ['date', 'sales']]
#     X = train_df[cols]
#     y = train_df['sales']
    
#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=2018
#     )
    
#     # Model parameters (updated with correct parameter names)
#     params = {
#         'nthread': 10,
#         'max_depth': 5,
#         'boosting_type': 'gbdt',
#         'objective': 'regression_l1',
#         'metric': 'mape',
#         'num_leaves': 64,
#         'learning_rate': 0.2,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'lambda_l1': 3.097,
#         'lambda_l2': 2.948,
#         'verbose': 1,
#         'min_child_weight': 6.996,
#         'min_split_gain': 0.037,
#     }
    
#     # Train model with correct parameter names
#     lgb_train = lgb.Dataset(X_train, y_train)
#     lgb_valid = lgb.Dataset(X_val, y_val)
    
#     model = lgb.train(
#         params,
#         lgb_train,
#         num_boost_round=3000,
#         valid_sets=[lgb_train, lgb_valid],
#         early_stopping_rounds=50,  # This is the correct parameter name
#         verbose_eval=50
#     )
    
#     # Save model
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")
    
#     return model

# if __name__ == "__main__":
#     try:
#         # Load data - adjust the path to your dataset
#         train_df = load_data("data/train.csv")
        
#         # Create model directory if it doesn't exist
#         import os
#         os.makedirs('model', exist_ok=True)
        
#         # Train and save model
#         model = train_model(train_df)
#         print("Model training completed successfully!")
#     except Exception as e:
#         print(f"Error during training: {str(e)}")







# import pandas as pd
# import numpy as np
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objs as go
# import plotly.offline as py
# import statsmodels.api as sm
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import warnings

# # Set plotting style
# plt.style.use('fivethirtyeight')
# sns.set()

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Optional imports - uncomment if you need them
# # from fbprophet import Prophet

# def load_data(datapath):
#     """
#     Load dataset from path and display basic info
#     """
#     data = pd.read_csv(datapath)
#     print('Shape:', data.shape)
#     print(data.sample(10))
#     return data

# def sales_distribution_analysis(data):
#     """
#     Analyze sales distribution across the data
#     """
#     sales_df = data.copy(deep=True)
#     sales_df['sales_bins'] = pd.cut(sales_df.sales, [0, 50, 100, 150, 200, 250])
    
#     print('Sales Statistics:')
#     print(f'Max sale: {sales_df.sales.max()}')
#     print(f'Min sale: {sales_df.sales.min()}')
#     print(f'Avg sale: {sales_df.sales.mean():.2f}')
#     print()
    
#     # Calculate percentages
#     total_points = pd.value_counts(sales_df.sales_bins).sum()
#     sales_pct = pd.value_counts(sales_df.sales_bins).apply(lambda s: (s/total_points)*100)
#     print('Sales bucket v/s Total percentage:')
#     print(sales_pct)
    
#     # Visualization
#     plt.figure(figsize=(12, 6))
#     sales_count = pd.value_counts(sales_df.sales_bins)
#     sales_count.sort_values(ascending=True).plot(
#         kind='barh', 
#         title='Sales distribution'
#     )
#     plt.tight_layout()
#     plt.savefig('sales_distribution.png')
#     plt.close()
    
#     return sales_df

# def store_item_analysis(data):
#     """
#     Analyze sales data across stores and items
#     """
#     store_df = data.copy()
    
#     # Create pivot table
#     sales_pivoted_df = pd.pivot_table(
#         store_df, 
#         index='store', 
#         values=['sales', 'date'], 
#         columns='item', 
#         aggfunc=np.mean
#     )
    
#     # Plot histogram of sales
#     plt.figure(figsize=(20, 10))
#     sales_pivoted_df.plot(kind="hist", figsize=(20, 10))
#     plt.tight_layout()
#     plt.savefig('store_item_histogram.png')
#     plt.close()
    
#     return store_df, sales_pivoted_df

# def analyze_store_sales(sales_pivoted_df):
#     """
#     Calculate and visualize average sales across stores
#     """
#     sales_across_store_df = sales_pivoted_df.copy()
#     sales_across_store_df['avg_sale'] = sales_across_store_df.apply(lambda r: r.mean(), axis=1)
    
#     # Create scatter plot
#     sales_store_data = go.Scatter(
#         y=sales_across_store_df.avg_sale.values,
#         mode='markers',
#         marker=dict(
#             size=sales_across_store_df.avg_sale.values,
#             color=sales_across_store_df.avg_sale.values,
#             colorscale='Viridis',
#             showscale=True
#         ),
#         text=sales_across_store_df.index.values
#     )
    
#     sales_store_layout = go.Layout(
#         autosize=True,
#         title='Scatter plot of avg sales per store',
#         hovermode='closest',
#         xaxis=dict(
#             title='Stores',
#             ticklen=10,
#             zeroline=False,
#             gridwidth=1,
#         ),
#         yaxis=dict(
#             title='Avg Sales',
#             ticklen=10,
#             zeroline=False,
#             gridwidth=1,
#         ),
#         showlegend=False
#     )
    
#     fig = go.Figure(data=[sales_store_data], layout=sales_store_layout)
#     # Save as HTML file
#     py.plot(fig, filename='scatter_sales_store.html', auto_open=False)
    
#     return sales_across_store_df

# def analyze_item_sales(sales_pivoted_df):
#     """
#     Calculate and visualize average sales across items
#     """
#     sales_across_item_df = sales_pivoted_df.copy()
#     # Add row with average sales per item
#     sales_across_item_df.loc[11] = sales_across_item_df.apply(lambda r: r.mean(), axis=0)
    
#     # Create dataframe with item averages
#     avg_sales_per_item = pd.DataFrame(
#         data=[[i+1, a] for i, a in enumerate(sales_across_item_df.loc[11:].values[0])],
#         columns=['item', 'avg_sale']
#     )
    
#     # Sort by average sales
#     avg_sales_per_item.sort_values(by='avg_sale', ascending=False, inplace=True)
    
#     # Create bar chart
#     sales_item_data = go.Bar(
#         x=[i for i in range(0, len(avg_sales_per_item))],
#         y=avg_sales_per_item.avg_sale.values,
#         marker=dict(
#             color=avg_sales_per_item.avg_sale.values,
#             colorscale='Blackbody',
#             showscale=True
#         ),
#         text=avg_sales_per_item.item.values
#     )
    
#     sales_item_layout = go.Layout(
#         autosize=True,
#         title='Bar plot of avg sales per item',
#         hovermode='closest',
#         xaxis=dict(
#             title='Items',
#             ticklen=5,
#             zeroline=False,
#             gridwidth=1,
#         ),
#         yaxis=dict(
#             title='Avg Sales',
#             ticklen=10,
#             zeroline=False,
#             gridwidth=1,
#         ),
#         showlegend=False
#     )
    
#     fig = go.Figure(data=[sales_item_data], layout=sales_item_layout)
#     # Save as HTML file
#     py.plot(fig, filename='bar_sales_item.html', auto_open=False)
    
#     return avg_sales_per_item

# def visualize_time_series(data, store_id=10, item_id=40):
#     """
#     Visualize time series data for a specific store and item
#     """
#     store_item_df = data.copy()
    
#     # Filter data
#     print('Before filter:', store_item_df.shape)
#     store_item_df = store_item_df[(store_item_df.store == store_id) & 
#                                  (store_item_df.item == item_id)]
#     print('After filter:', store_item_df.shape)
    
#     # Plot time series
#     plt.figure(figsize=(15, 7))
#     plt.plot(store_item_df.date, store_item_df.sales)
#     plt.title(f'Sales for Store {store_id}, Item {item_id}')
#     plt.xlabel('Date')
#     plt.ylabel('Sales')
#     plt.tight_layout()
#     plt.savefig(f'time_series_store{store_id}_item{item_id}.png')
#     plt.close()
    
#     return store_item_df

# def visualize_multiple_series(data, store_ids=[1, 1, 1, 1], item_ids=[10, 20, 30, 40]):
#     """
#     Visualize multiple time series for comparison
#     """
#     multi_store_item_df = data.copy()
    
#     # Filter data
#     print('Before filter:', multi_store_item_df.shape)
#     multi_store_item_df = multi_store_item_df[
#         (multi_store_item_df.store.isin(store_ids)) & 
#         (multi_store_item_df.item.isin(item_ids))
#     ]
#     print('After filter:', multi_store_item_df.shape)
    
#     # Create plotly figure
#     multi_store_item_ts_data = []
#     for st, it in zip(store_ids, item_ids):
#         flt = multi_store_item_df[
#             (multi_store_item_df.store == st) & 
#             (multi_store_item_df.item == it)
#         ]
#         multi_store_item_ts_data.append(
#             go.Scatter(
#                 x=flt.date, 
#                 y=flt.sales, 
#                 name=f"Store:{st}, Item:{it}"
#             )
#         )
    
#     fig = go.Figure(data=multi_store_item_ts_data)
#     fig.update_layout(
#         title="Sales across multiple store-item combinations",
#         xaxis_title="Date",
#         yaxis_title="Sales"
#     )
    
#     # Save as HTML file
#     py.plot(fig, filename='multiple_time_series.html', auto_open=False)
    
#     return multi_store_item_df

# def preprocess_data(train_data, test_data):
#     """
#     Preprocess data by creating additional features
#     """
#     # Convert date to datetime
#     train_data['date'] = pd.to_datetime(train_data['date'])
#     test_data['date'] = pd.to_datetime(test_data['date'])
    
#     # Extract date features
#     for df in [train_data, test_data]:
#         df['month'] = df['date'].dt.month
#         df['day'] = df['date'].dt.dayofweek
#         df['year'] = df['date'].dt.year
#         df['day_of_month'] = df['date'].dt.day
#         df['week_of_year'] = df['date'].dt.isocalendar().week
        
#         # Create store-item combination feature
#         df['store_item'] = df['store'].astype(str) + '_' + df['item'].astype(str)
    
#     # Create lag features for training data if possible
#     if 'sales' in train_data.columns:
#         # Sort by date to ensure correct lag calculation
#         train_data = train_data.sort_values(['store', 'item', 'date'])
        
#         # Create lag features by store-item
#         for lag in [1, 2, 3, 7, 14]:
#             train_data[f'sales_lag_{lag}'] = train_data.groupby(['store', 'item'])['sales'].shift(lag)
        
#         # Create rolling statistics
#         for window in [7, 14, 28]:
#             train_data[f'sales_rolling_mean_{window}'] = train_data.groupby(['store', 'item'])['sales'].transform(
#                 lambda x: x.rolling(window, min_periods=1).mean()
#             )
#             train_data[f'sales_rolling_std_{window}'] = train_data.groupby(['store', 'item'])['sales'].transform(
#                 lambda x: x.rolling(window, min_periods=1).std()
#             )
    
#     # Drop missing values or fill them
#     train_data = train_data.fillna(0)
    
#     return train_data, test_data

# def prepare_training_data(train_data, test_data):
#     """
#     Prepare data for training
#     """
#     # Identify features to use
#     features = [col for col in test_data.columns if col not in ['date', 'id', 'sales']]
#     target = 'sales'
    
#     # Split training data
#     train_x, valid_x, train_y, valid_y = train_test_split(
#         train_data[features],
#         train_data[target], 
#         test_size=0.2, 
#         random_state=2018
#     )
    
#     return train_x, valid_x, train_y, valid_y, features

# def time_based_cv_split(train_data, n_splits=5):
#     """
#     Create time-based cross-validation splits
#     """
#     # Sort by date
#     train_data = train_data.sort_values('date')
    
#     # Define features and target
#     features = [col for col in train_data.columns if col not in ['date', 'id', 'sales']]
#     target = 'sales'
    
#     # Create TimeSeriesSplit
#     tscv = TimeSeriesSplit(n_splits=n_splits)
    
#     # Return split indices and data
#     return [(train_data.iloc[train_idx][features], 
#              train_data.iloc[train_idx][target],
#              train_data.iloc[test_idx][features],
#              train_data.iloc[test_idx][target]) 
#             for train_idx, test_idx in tscv.split(train_data)]

# def train_lgb_model(train_x, train_y, valid_x, valid_y, features, params=None):
#     """
#     Train LightGBM model
#     """
#     if params is None:
#         params = {
#             'nthread': 10,
#             'max_depth': 5,
#             'task': 'train',
#             'boosting_type': 'gbdt',
#             'objective': 'regression_l1',
#             'metric': 'mape',
#             'num_leaves': 64,
#             'learning_rate': 0.2,
#             'feature_fraction': 0.9,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 5,
#             'lambda_l1': 3.097758978478437,
#             'lambda_l2': 2.9482537987198496,
#             'verbose': 1,
#             'min_child_weight': 6.996211413900573,
#             'min_split_gain': 0.037310344962162616,
#         }
    
#     # Create LightGBM datasets
#     lgb_train = lgb.Dataset(train_x, train_y)
#     lgb_valid = lgb.Dataset(valid_x, valid_y)
    
#     # Train model
#     model = lgb.train(
#         params, 
#         lgb_train, 
#         num_boost_round=3000, 
#         valid_sets=[lgb_train, lgb_valid],
#         early_stopping_rounds=50, 
#         verbose_eval=50
#     )
    
#     # Evaluate model
#     valid_pred = model.predict(valid_x)
#     mae = mean_absolute_error(valid_y, valid_pred)
#     rmse = np.sqrt(mean_squared_error(valid_y, valid_pred))
    
#     print(f"Validation MAE: {mae:.4f}")
#     print(f"Validation RMSE: {rmse:.4f}")
    
#     # Plot feature importance
#     plt.figure(figsize=(12, 8))
#     lgb.plot_importance(model, max_num_features=20)
#     plt.title('Feature Importance')
#     plt.tight_layout()
#     plt.savefig('feature_importance.png')
#     plt.close()
    
#     return model

# def make_predictions(model, test_data, features):
#     """
#     Make predictions using trained model
#     """
#     predictions = model.predict(test_data[features])
#     return predictions

# def create_submission(test_df, predictions, filename="submission.csv"):
#     """
#     Create submission file
#     """
#     submission = test_df[['id']].copy()
#     submission['sales'] = predictions
#     submission.to_csv(filename, index=False)
#     return submission

# def ensemble_predictions(predictions_list, weights=None):
#     """
#     Create ensemble predictions from multiple models
#     """
#     if weights is None:
#         weights = [1/len(predictions_list)] * len(predictions_list)
    
#     assert len(predictions_list) == len(weights), "Number of predictions must match number of weights"
#     assert abs(sum(weights) - 1.0) < 1e-10, "Weights must sum to 1"
    
#     # Weighted average
#     final_predictions = np.zeros_like(predictions_list[0])
#     for pred, weight in zip(predictions_list, weights):
#         final_predictions += pred * weight
    
#     return final_predictions

# def analyze_errors(y_true, y_pred):
#     """
#     Analyze prediction errors
#     """
#     errors = y_true - y_pred
    
#     plt.figure(figsize=(12, 6))
#     plt.hist(errors, bins=50)
#     plt.title('Error Distribution')
#     plt.xlabel('Error')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.savefig('error_distribution.png')
#     plt.close()
    
#     plt.figure(figsize=(12, 6))
#     plt.scatter(y_true, y_pred, alpha=0.5)
#     plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
#     plt.title('Actual vs Predicted')
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.tight_layout()
#     plt.savefig('actual_vs_predicted.png')
#     plt.close()
    
#     return errors

# def main():
#     # Initialize plotly
#     py.init_notebook_mode(connected=True)
    
#     # Load data
#     print("Loading data...")
#     train_df = load_data('./input/demand-forecasting-kernels-only/train.csv')
#     test_df = load_data('./input/demand-forecasting-kernels-only/test.csv')
#     sample_df = load_data('./input/demand-forecasting-kernels-only/sample_submission.csv')
    
#     # EDA
#     print("\nAnalyzing sales distribution...")
#     sales_df = sales_distribution_analysis(train_df)
    
#     print("\nAnalyzing store-item relationships...")
#     store_df, sales_pivoted_df = store_item_analysis(train_df)
    
#     print("\nAnalyzing store sales...")
#     sales_across_store_df = analyze_store_sales(sales_pivoted_df)
    
#     print("\nAnalyzing item sales...")
#     avg_sales_per_item = analyze_item_sales(sales_pivoted_df)
    
#     print("\nVisualizing time series for a single store-item...")
#     store_item_df = visualize_time_series(train_df)
    
#     print("\nVisualizing multiple time series...")
#     multi_store_item_df = visualize_multiple_series(train_df)
    
#     # Preprocess data
#     print("\nPreprocessing data...")
#     train_df_processed, test_df_processed = preprocess_data(train_df, test_df)
    
#     # Prepare training data
#     print("\nPreparing training data...")
#     train_x, valid_x, train_y, valid_y, features = prepare_training_data(train_df_processed, test_df_processed)
    
#     # Alternative: time-based CV
#     print("\nCreating time-based CV splits...")
#     cv_splits = time_based_cv_split(train_df_processed)
    
#     # Train model
#     print("\nTraining LightGBM model...")
#     model = train_lgb_model(train_x, train_y, valid_x, valid_y, features)
    
#     # Make predictions
#     print("\nMaking predictions...")
#     predictions = make_predictions(model, test_df_processed, features)
    
#     # Create submission
#     print("\nCreating submission file...")
#     submission = create_submission(test_df, predictions, "lgb_submission.csv")
    
#     # Optional: Analyze errors on validation set
#     print("\nAnalyzing prediction errors...")
#     valid_pred = model.predict(valid_x)
#     errors = analyze_errors(valid_y.values, valid_pred)
    
#     print("\nDone!")
#     return submission

# if __name__ == "__main__":
#     main()




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