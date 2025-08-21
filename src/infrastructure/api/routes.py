# src/infrastructure/api/routes.py

from fastapi import APIRouter, HTTPException, status
from src.domain.models import TrainRequest, TrainResponse, PredictRequest, PredictResponse
from src.use_cases.train_model_use_case import TrainModelUseCase
from src.use_cases.predict_forecast_use_case import PredictForecastUseCase

router = APIRouter()

@router.post("/train/{app_id}", response_model=TrainResponse, status_code=status.HTTP_200_OK)
async def train_model(app_id: str, request: TrainRequest):
    """
    Treina um modelo de previsão para uma aplicação específica.

    - **app_id**: Identificador único da aplicação.
    - **series_data**: Dados históricos da série temporal para treino.
    """
    try:
        use_case = TrainModelUseCase()
        response = use_case.execute(app_id=app_id, series_data=request.series_data)
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro interno do servidor: {e}")

@router.post("/predict/{app_id}", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_forecast(app_id: str, request: PredictRequest):
    """
    Realiza a previsão para uma aplicação específica usando um modelo previamente treinado.

    - **app_id**: Identificador único da aplicação.
    - **series_data**: Dados históricos da série temporal para previsão (usados para contextualizar a previsão).
    - **n_predict_steps**: Número de passos futuros para prever.
    """
    try:
        use_case = PredictForecastUseCase()
        response = use_case.execute(
            app_id=app_id,
            series_data=request.series_data,
            n_predict_steps=request.n_predict_steps
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro interno do servidor: {e}")
