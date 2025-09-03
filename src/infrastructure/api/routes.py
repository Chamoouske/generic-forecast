# src/infrastructure/api/routes.py

from fastapi import APIRouter, HTTPException, status, Query, File, UploadFile, BackgroundTasks
from src.domain.models import TrainRequest, PredictRequest, PredictResponse, ForecastInputData
from src.use_cases.train_model_use_case import TrainModelUseCase
from src.use_cases.predict_forecast_use_case import PredictForecastUseCase
import pandas as pd
from io import StringIO
import logging
from typing import List

# Configuração básica de logging para imprimir os resultados do treino
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

def run_training_in_background(app_id: str, series_data: List[ForecastInputData], target_column: str):
    """Função que executa o caso de uso de treinamento e loga o resultado."""
    try:
        logging.info(f"[BACKGROUND-TRAIN] Iniciando treinamento para app_id: {app_id}, target: {target_column}")
        use_case = TrainModelUseCase()
        response = use_case.execute(
            app_id=app_id,
            series_data=series_data,
            target_column=target_column
        )
        logging.info(f"[BACKGROUND-TRAIN] Treinamento concluído para app_id: {app_id}, target: {target_column}. Resultado: {response.message}")
    except Exception as e:
        logging.error(f"[BACKGROUND-TRAIN] Erro durante treinamento para app_id: {app_id}, target: {target_column}. Erro: {e}", exc_info=True)

@router.post("/train/{app_id}", status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    app_id: str, 
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    target_column: str = Query(..., description="A coluna alvo para o treinamento do modelo (ex: 'Autorizada').")
):
    """
    Inicia o treinamento de um modelo em segundo plano a partir de um corpo JSON.
    A resposta é imediata e o resultado do treino é logado na aplicação.
    """
    background_tasks.add_task(
        run_training_in_background, 
        app_id=app_id, 
        series_data=request.series_data, 
        target_column=target_column
    )
    return {"message": f"Processo de treinamento para '{app_id}_{target_column}' iniciado em segundo plano."}

@router.post("/train-csv/{app_id}", status_code=status.HTTP_202_ACCEPTED)
async def train_model_from_csv(
    app_id: str,
    background_tasks: BackgroundTasks,
    target_column: str = Query(..., description="A coluna alvo para o treinamento (ex: 'Autorizada')."),
    file: UploadFile = File(..., description="Arquivo .csv com os dados históricos para treino. Separador deve ser ';'.")
):
    """
    Inicia o treinamento de um modelo em segundo plano a partir de um arquivo CSV.
    A resposta é imediata e o resultado do treino é logado na aplicação.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="O arquivo deve ser um .csv")

    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        df = pd.read_csv(StringIO(content_str), sep=';', dtype=str)
        series_data = [ForecastInputData(**row) for row in df.to_dict(orient='records')]

        background_tasks.add_task(
            run_training_in_background, 
            app_id=app_id, 
            series_data=series_data, 
            target_column=target_column
        )
        return {"message": f"Processo de treinamento para '{app_id}_{target_column}' a partir do arquivo '{file.filename}' iniciado em segundo plano."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao processar o arquivo CSV antes de iniciar o treinamento: {e}")


@router.post("/predict/{app_id}", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_forecast(
    app_id: str, 
    request: PredictRequest,
    target_column: str = Query(..., description="A coluna alvo para a previsão (ex: 'Autorizada').")
):
    """
    Realiza a previsão para uma aplicação e métrica específicas.
    """
    try:
        use_case = PredictForecastUseCase()
        response = use_case.execute(
            app_id=app_id,
            series_data=request.series_data,
            n_predict_steps=request.n_predict_steps,
            target_column=target_column
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro interno do servidor: {e}")
