
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class DemandForecastingModel:
    """
    Service class for demand forecasting model
    """
    def __init__(self, model_path, features_path):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.features = None
    
    def load_model(self):
        """
        Load the trained model and features
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Loading features from {self.features_path}")
            self.features = joblib.load(self.features_path)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, date_str, store, item):
        """
        Preprocess input data for prediction
        """
        # Convert date string to datetime
        date = pd.to_datetime(date_str)
        
        # Create a dataframe with a single row
        input_data = pd.DataFrame({
            'store': [store],
            'item': [item],
            'month': [date.month],
            'day': [date.dayofweek],
            'year': [date.year],
            'day_of_month': [date.day]
        })
        
        # Make sure the dataframe has all required features in the correct order
        input_data = input_data[self.features]
        
        return input_data
    
    def predict(self, date_str, store, item):
        """
        Make a prediction based on input data
        """
        if self.model is None:
            loaded = self.load_model()
            if not loaded:
                raise Exception("Failed to load model")
        
        try:
            # Preprocess input
            input_data = self.preprocess_input(date_str, store, item)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Ensure non-negative prediction for sales
            prediction = max(0, prediction)
            
            return float(prediction)
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

            