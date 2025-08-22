# src/domain/models.py

from pydantic import BaseModel, Field
from typing import List, Dict

class TimeSeriesData(BaseModel):
    timestamp: str
    value: float

class TrainRequest(BaseModel):
    series_data: List[TimeSeriesData] = Field(..., description="Dados históricos da série temporal para treino (timestamp e valor).")

class TrainResponse(BaseModel):
    message: str
    model_id: str
    model_version: str | None = None
    metrics: dict | None = None

class PredictRequest(BaseModel):
    series_data: List[TimeSeriesData] = Field(..., description="Dados históricos da série temporal para previsão (timestamp e valor).")
    n_predict_steps: int = Field(..., gt=0, description="Número de passos futuros para prever.")

class PredictResponse(BaseModel):
    forecast: Dict[str, float]
    model_used: str = "Prophet"
    model_id: str
    model_version: str | None = None
