# src/domain/models.py
from pydantic import BaseModel, Field
from typing import List, Dict

class ForecastInputData(BaseModel):
    DIA: str
    HORA: str
    DIA_DA_SEMANA: str
    RASCUNHO: int
    AGUARDANDO_PAGAMENTO: int
    A_TRANSMITIR: int
    EM_PROCESSAMENTO: int
    AUTORIZADA: int
    DENEGADA: int
    CANCELADA: int
    REJEITADA: int
    ERRO_SCHEMA: int

class TrainRequest(BaseModel):
    series_data: List[ForecastInputData] = Field(..., description="Dados históricos para treino no formato de colunas.")

class TrainResponse(BaseModel):
    message: str
    model_id: str
    model_version: str | None = None
    metrics: dict | None = None

class PredictRequest(BaseModel):
    series_data: List[ForecastInputData] = Field(..., description="Dados históricos para previsão no formato de colunas.")
    n_predict_steps: int = Field(..., gt=0, description="Número de passos futuros para prever.")

class PredictResponse(BaseModel):
    forecast: Dict[str, float]
    model_used: str = "Prophet"
    model_id: str
    model_version: str | None = None