# src/use_cases/predict_forecast_use_case.py

import pandas as pd
from src.domain.models import TimeSeriesData, PredictResponse
from src.infrastructure.persistence.model_repository import load_model, load_production_model_info
from typing import List, Dict

class PredictForecastUseCase:
    def execute(self, app_id: str, series_data: List[TimeSeriesData], n_predict_steps: int) -> PredictResponse:
        """
        Executa o caso de uso de previsão de uma série temporal.

        Args:
            app_id (str): O ID da aplicação para o qual o modelo foi treinado.
            series_data (List[TimeSeriesData]): Dados históricos da série temporal para previsão.
            n_predict_steps (int): Número de passos futuros para prever.

        Returns:
            PredictResponse: Resposta contendo a previsão, o modelo usado, ID do modelo e versão.
        """
        if not series_data:
            raise ValueError("Os dados da série temporal para previsão não podem ser vazios.")

        # Carregar o modelo treinado
        try:
            production_model_info = load_production_model_info(app_id)
            if production_model_info is None:
                raise ValueError(f"Nenhum modelo em produção encontrado para app_id '{app_id}'. Treine e promova um modelo primeiro.")
            
            prophet_model = load_model(app_id, production_model_info["version"])
            model_version_used = production_model_info["version"]
        except FileNotFoundError:
            raise ValueError(f"Modelo para app_id '{app_id}' e versão '{production_model_info["version"]}' não encontrado. Treine o modelo primeiro.")
        
        # Converter os dados de entrada para um pandas Series
        timestamps = [item.timestamp for item in series_data]
        values = [item.value for item in series_data]
        
        try:
            series = pd.Series(values, index=pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S"))
        except Exception as e:
            raise ValueError(f"Erro ao parsear timestamps: {e}. Verifique o formato.")

        print("Previsão iniciada para app_id:", app_id)
        # Realizar a previsão
        forecast_series = prophet_model.predict(
            series=series,
            n_predict=n_predict_steps
        )
        
        # Converter o resultado para um dicionário com timestamps como strings
        forecast_dict = {str(k): v for k, v in forecast_series.to_dict().items()}

        return PredictResponse(
            forecast=forecast_dict,
            model_id=app_id,
            model_version=model_version_used
        )

if __name__ == "__main__":
    # Exemplo de uso (requer que um modelo 'test_app_predict' tenha sido treinado e salvo)
    print("Testando PredictForecastUseCase:")
    from src.domain.models import TimeSeriesData
    from src.use_cases.train_model_use_case import TrainModelUseCase
    import os

    # Certificar que a pasta models existe
    os.makedirs("models", exist_ok=True)

    # Primeiro, treinar e salvar um modelo para testar a previsão
    sample_data_train = [
        TimeSeriesData(timestamp="2023-01-01", value=10.0),
        TimeSeriesData(timestamp="2023-01-02", value=12.0),
        TimeSeriesData(timestamp="2023-01-03", value=13.0),
        TimeSeriesData(timestamp="2023-01-04", value=15.0),
        TimeSeriesData(timestamp="2023-01-05", value=14.0),
        TimeSeriesData(timestamp="2023-01-06", value=16.0),
        TimeSeriesData(timestamp="2023-01-07", value=18.0),
        TimeSeriesData(timestamp="2023-01-08", value=17.0),
        TimeSeriesData(timestamp="2023-01-09", value=19.0),
        TimeSeriesData(timestamp="2023-01-10", value=20.0),
    ]
    train_use_case = TrainModelUseCase()
    try:
        train_response = train_use_case.execute(app_id="test_app_predict", series_data=sample_data_train)
        print(f"\nModelo de teste treinado: {train_response.message}")
    except Exception as e:
        print(f"\nErro ao treinar modelo de teste: {e}")

    # Dados para previsão (pode ser os mesmos ou uma continuação)
    sample_data_predict = [
        TimeSeriesData(timestamp="2023-01-08", value=17.0),
        TimeSeriesData(timestamp="2023-01-09", value=19.0),
        TimeSeriesData(timestamp="2023-01-10", value=20.0),
    ]

    predict_use_case = PredictForecastUseCase()
    try:
        response = predict_use_case.execute(
            app_id="test_app_predict", 
            series_data=sample_data_predict, 
            n_predict_steps=3
        )
        print(f"\nResposta da previsão: {response.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"\nErro ao executar caso de uso de previsão: {e}")

    # Testar com modelo não encontrado
    try:
        predict_use_case.execute(app_id="non_existent_app", series_data=sample_data_predict, n_predict_steps=1)
    except ValueError as e:
        print(f"\nErro esperado para modelo não encontrado: {e}")

    # Testar com dados vazios
    try:
        predict_use_case.execute(app_id="test_app_predict", series_data=[], n_predict_steps=1)
    except ValueError as e:
        print(f"\nErro esperado para dados vazios na previsão: {e}")
