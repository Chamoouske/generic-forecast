# src/use_cases/train_model_use_case.py

import pandas as pd
from src.domain.models import TimeSeriesData, TrainResponse
from src.infrastructure.forecasting_models.prophet_model import ProphetModel
from src.infrastructure.persistence.model_repository import save_model, load_production_model_info, update_production_model_info
from src.infrastructure.persistence.data_persistance import load_csv
from typing import List

class TrainModelUseCase:
    def execute(self, app_id: str, series_data: List[TimeSeriesData], csv_local: str = '') -> TrainResponse:
        """
        Executa o caso de uso de treinamento de um modelo Prophet.

        Args:
            app_id (str): O ID da aplicação para o qual o modelo será treinado.
            series_data (List[TimeSeriesData]): Dados históricos da série temporal para treino.

        Returns:
            TrainResponse: Resposta contendo mensagem de sucesso, ID do modelo e versão.
        """
        if not series_data and not csv_local:
            raise ValueError("Os dados da série temporal para treino não podem ser vazios.")
        if csv_local:
            # Carregar os dados do CSV local
            df = load_csv(csv_local)
            series = pd.Series(df['value'].values, index=df['timestamp'])
        elif series_data:
            # Converter os dados de entrada para um pandas Series
            timestamps = [item.timestamp for item in series_data]
            values = [item.value for item in series_data]
            series = pd.Series(values, index=pd.to_datetime(timestamps))

        # Instanciar e treinar o modelo Prophet
        prophet_model = ProphetModel()
        prophet_model.train(series)

        # Avaliar o novo modelo treinado
        new_model_metrics = prophet_model.evaluate(series)

        # Carregar informações do modelo em produção
        production_model_info = load_production_model_info(app_id)

        promote_new_model = False
        promotion_reason = ""
        saved_path = ""
        model_version = ""

        if production_model_info is None:
            promote_new_model = True
            promotion_reason = "Não há modelo em produção. Novo modelo promovido."
        else:
            current_production_rmse = production_model_info["metrics"]["rmse"]
            new_model_rmse = new_model_metrics["rmse"]

            if new_model_rmse < current_production_rmse:
                promote_new_model = True
                promotion_reason = f"Métricas do novo modelo ({new_model_rmse:.2f} RMSE) são melhores que o modelo em produção ({current_production_rmse:.2f} RMSE)."
            else:
                promotion_reason = f"Métricas do novo modelo ({new_model_rmse:.2f} RMSE) não são melhores que o modelo em produção ({current_production_rmse:.2f} RMSE)."

        if promote_new_model:
            # Salvar o novo modelo e promovê-lo para produção
            model_version, saved_path = save_model(prophet_model, app_id, new_model_metrics)
            update_production_model_info(app_id, model_version, saved_path, new_model_metrics)
            message = f"Modelo {app_id} treinado e PROMOVIDO para produção (versão: {model_version}). {promotion_reason}"
        else:
            # O modelo foi treinado, mas não promovido
            message = f"Modelo {app_id} treinado, mas NÃO PROMOVIDO para produção. {promotion_reason}"

        return TrainResponse(
            message=message,
            model_id=app_id,
            model_version=model_version if promote_new_model else None, # Retorna a versão apenas se promovido
            metrics=new_model_metrics
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
