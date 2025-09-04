# src/use_cases/train_anomaly_model_use_case.py

import pandas as pd
from src.domain.models import ForecastInputData, TrainResponse
from src.infrastructure.forecasting_models.anomaly_detection_model import AnomalyDetectionModel
from src.infrastructure.persistence.model_repository import save_model
from typing import List

class TrainAnomalyModelUseCase:
    def execute(self, app_id: str, series_data: List[ForecastInputData]) -> TrainResponse:
        """
        Executes the use case for training an anomaly detection model.

        Args:
            app_id (str): The base ID of the application.
            series_data (List[ForecastInputData]): Historical data for training.

        Returns:
            TrainResponse: Response from the training operation.
        """
        if not series_data:
            raise ValueError("The data for training cannot be empty.")

        MAX_TRAINING_SAMPLES = 10000
        df = pd.DataFrame([item.model_dump() for item in series_data])

        if len(df) > MAX_TRAINING_SAMPLES:
            df = df.sample(n=MAX_TRAINING_SAMPLES, random_state=42)

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

        model = AnomalyDetectionModel()
        model.train(df)

        # For anomaly detection, evaluation is different. We can't easily calculate
        # metrics like RMSE without labels. We will just save the model.
        # The "metrics" can be statistics about the training data if needed.
        metrics = {"training_samples": len(df)}

        model_version, saved_path = save_model(model, model_id, metrics)

        message = f"Anomaly detection model '{model_id}' trained and saved (version: {model_version})."

        return TrainResponse(
            message=message,
            model_id=model_id,
            model_version=model_version,
            metrics=metrics
        )
