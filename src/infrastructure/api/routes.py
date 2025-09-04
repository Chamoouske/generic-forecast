# src/infrastructure/api/routes.py

from fastapi import APIRouter, HTTPException, status, File, UploadFile, BackgroundTasks
from src.domain.models import TrainRequest, ForecastInputData
from src.use_cases.train_anomaly_model_use_case import TrainAnomalyModelUseCase
from src.use_cases.detect_anomalies_use_case import DetectAnomaliesUseCase
import pandas as pd
from io import StringIO
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

def run_anomaly_training_in_background(app_id: str, series_data: List[ForecastInputData]):
    """Function that runs the anomaly detection training use case and logs the result."""
    try:
        logging.info(f"[BACKGROUND-TRAIN] Starting anomaly model training for app_id: {app_id}")
        use_case = TrainAnomalyModelUseCase()
        response = use_case.execute(
            app_id=app_id,
            series_data=series_data
        )
        logging.info(f"[BACKGROUND-TRAIN] Anomaly model training completed for app_id: {app_id}. Result: {response.message}")
    except Exception as e:
        logging.error(f"[BACKGROUND-TRAIN] Error during anomaly model training for app_id: {app_id}. Error: {e}", exc_info=True)

@router.post("/train-anomaly-model-csv/{app_id}", status_code=status.HTTP_202_ACCEPTED)
async def train_anomaly_model_from_csv(
    app_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description=".csv file with historical data for training. Separator must be ';'.")
):
    """
    Starts training an anomaly detection model in the background from a CSV file.
    The response is immediate, and the training result is logged.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The file must be a .csv")

    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        df = pd.read_csv(StringIO(content_str), sep=';', dtype=str)
        series_data = [ForecastInputData(**row) for row in df.to_dict(orient='records')]

        background_tasks.add_task(
            run_anomaly_training_in_background,
            app_id=app_id,
            series_data=series_data
        )
        return {"message": f"Anomaly detection model training process for '{app_id}' from file '{file.filename}' started in the background."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/detect-anomalies-csv/{app_id}", status_code=status.HTTP_200_OK)
async def detect_anomalies_from_csv(
    app_id: str,
    file: UploadFile = File(..., description=".csv file with data to check for anomalies. Separator must be ';'.")
):
    """
    Detects anomalies in data provided via a CSV file.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The file must be a .csv")

    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        df = pd.read_csv(StringIO(content_str), sep=';', dtype=str)
        series_data = [ForecastInputData(**row) for row in df.to_dict(orient='records')]

        use_case = DetectAnomaliesUseCase()
        result_df = use_case.execute(
            app_id=app_id,
            series_data=series_data
        )
        
        anomalies = result_df[result_df['anomaly'] == -1]
        
        return {
            "message": f"Anomaly detection complete. Found {len(anomalies)} potential anomalies.",
            "anomalies": anomalies.to_dict(orient='records')
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")