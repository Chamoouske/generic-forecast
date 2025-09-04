# src/use_cases/detect_anomalies_use_case.py

import pandas as pd
from src.domain.models import ForecastInputData
from src.infrastructure.persistence.model_repository import load_model, load_production_model_info
from typing import List

class DetectAnomaliesUseCase:
    def execute(self, app_id: str, series_data: List[ForecastInputData]) -> pd.DataFrame:
        """
        Executes the use case for detecting anomalies in a time series.

        Args:
            app_id (str): The base ID of the application.
            series_data (List[ForecastInputData]): The time series data to check for anomalies.

        Returns:
            pd.DataFrame: A DataFrame with the original data and an 'anomaly' column.
        """
        if not series_data:
            raise ValueError("The input data for anomaly detection cannot be empty.")

        df = pd.DataFrame([item.model_dump() for item in series_data])

        try:
            df['ds'] = pd.to_datetime(df['DIA'] + ' ' + df['HORA'], format='%d/%m/%Y %H')
        except Exception as e:
            raise ValueError(f"Erro ao parsear 'DIA' e 'HORA': {e}.")

        # Convert all numerical columns to float
        for col in df.columns:
            if col not in ['DIA', 'HORA', 'ds']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True) # Fill any NaNs created by coercion

        df = df.set_index('ds') # Set 'ds' as index for the model

        # Drop original string columns that are not features
        df = df.drop(columns=['DIA', 'HORA', 'DIA_DA_SEMANA'], errors='ignore')

        model_id = f"{app_id.lower()}_multivariate_anomaly"
        
        # Load production model information
        prod_model_info = load_production_model_info(model_id)
        if prod_model_info is None or "version" not in prod_model_info:
            raise ValueError(f"No production model found for '{model_id}'. Please train a model first.")
            
        # Load the actual model using the version info
        model_version = prod_model_info["version"]
        model = load_model(model_id, model_version)
        
        if model is None:
            raise ValueError(f"Could not load model '{model_id}' version '{model_version}'.")

        anomaly_predictions = model.predict(df)
        
        result_df = df.copy()
        result_df['anomaly'] = anomaly_predictions.values
        
        return result_df
