# src/use_cases/train_model_use_case.py

import pandas as pd
from src.domain.models import TimeSeriesData, TrainResponse
from src.infrastructure.forecasting_models.prophet_model import ProphetModel
from src.infrastructure.persistence.model_repository import save_model
from typing import List

class TrainModelUseCase:
    def execute(self, app_id: str, series_data: List[TimeSeriesData]) -> TrainResponse:
        """
        Executa o caso de uso de treinamento de um modelo Prophet.

        Args:
            app_id (str): O ID da aplicação para o qual o modelo será treinado.
            series_data (List[TimeSeriesData]): Dados históricos da série temporal para treino.

        Returns:
            TrainResponse: Resposta contendo mensagem de sucesso, ID do modelo e versão.
        """
        if not series_data:
            raise ValueError("Os dados da série temporal para treino não podem ser vazios.")

        # Converter os dados de entrada para um pandas Series
        timestamps = [item.timestamp for item in series_data]
        values = [item.value for item in series_data]
        
        try:
            series = pd.Series(values, index=pd.to_datetime(timestamps))
        except Exception as e:
            raise ValueError(f"Erro ao parsear timestamps: {e}. Verifique o formato.")

        # Instanciar e treinar o modelo Prophet
        prophet_model = ProphetModel()
        prophet_model.train(series)

        # Salvar o modelo treinado
        saved_path = save_model(prophet_model, app_id)

        return TrainResponse(
            message=f"Modelo {app_id} treinado e salvo com sucesso em {saved_path}",
            model_id=app_id
        )

if __name__ == "__main__":
    # Exemplo de uso
    print("Testando TrainModelUseCase:")
    from src.domain.models import TimeSeriesData

    # Dados de exemplo
    sample_data = [
        TimeSeriesData(timestamp="2023-01-01", value=10.0),
        TimeSeriesData(timestamp="2023-01-02", value=12.0),
        TimeSeriesData(timestamp="2023-01-03", value=13.0),
        TimeSeriesData(timestamp="2023-01-04", value=15.0),
        TimeSeriesData(timestamp="2023-01-05", value=14.0),
        TimeSeriesData(timestamp="2023-01-06", value=16.0),
    ]

    use_case = TrainModelUseCase()
    try:
        response = use_case.execute(app_id="test_app_train", series_data=sample_data)
        print(f"\nResposta do treino: {response.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"\nErro ao executar caso de uso de treino: {e}")

    # Testar com dados vazios
    try:
        use_case.execute(app_id="test_empty", series_data=[])
    except ValueError as e:
        print(f"\nErro esperado para dados vazios: {e}")
