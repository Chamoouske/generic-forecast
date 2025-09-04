# src/infrastructure/forecasting_models/anomaly_detection_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetectionModel:
    def __init__(self, contamination=0.01, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.feature_columns = []

    def _create_features(self, df: pd.DataFrame):
        """
        Creates features for anomaly detection from a multivariate DataFrame.
        This includes rolling window aggregates for each numerical column and time-based features.
        """
        # Ensure the DataFrame index is datetime for time-based features
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        features_df = df.copy() # Work on a copy to avoid modifying original df

        # Identify numerical columns to apply rolling features
        numerical_cols = features_df.select_dtypes(include=np.number).columns.tolist()

        # Rolling window features for each numerical column
        windows = [5, 15, 60] # 5-min, 15-min, 1-hour windows
        for col in numerical_cols:
            for window in windows:
                features_df[f'{col}_rolling_mean_{window}'] = features_df[col].rolling(window=window, min_periods=1).mean()
                features_df[f'{col}_rolling_std_{window}'] = features_df[col].rolling(window=window, min_periods=1).std()
                features_df[f'{col}_rolling_sum_{window}'] = features_df[col].rolling(window=window, min_periods=1).sum()

        # Time-based features
        features_df['hour'] = features_df.index.hour
        features_df['dayofweek'] = features_df.index.dayofweek
        features_df['dayofyear'] = features_df.index.dayofyear
        features_df['month'] = features_df.index.month
        features_df['year'] = features_df.index.year
        
        features_df.fillna(0, inplace=True) # Fill NaNs from rolling std of first element

        # Store feature columns for consistency during prediction
        self.feature_columns = features_df.columns.tolist()
        
        return features_df

    def train(self, df: pd.DataFrame):
        """
        Trains the Isolation Forest model with the given multivariate DataFrame.
        """
        if df.empty:
            raise ValueError("The DataFrame for training cannot be empty.")

        features_df = self._create_features(df)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Train model
        self.model.fit(scaled_features)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicts anomalies in a new multivariate DataFrame.
        Returns a Series with -1 for anomalies and 1 for normal points.
        """
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError("The model has not been trained. Call .train() first.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        features_df = self._create_features(df)
        
        # Ensure feature consistency: add missing columns, reorder
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0 # Add missing columns with default value
        features_df = features_df[self.feature_columns] # Reorder columns
        
        scaled_features = self.scaler.transform(features_df)
        
        predictions = self.model.predict(scaled_features)
        
        return pd.Series(predictions, index=df.index, name='anomaly')