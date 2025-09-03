# src/use_cases/train_model_use_case.py

import pandas as pd
from src.domain.models import ForecastInputData, TrainResponse
from src.infrastructure.forecasting_models.prophet_model import ProphetModel
from src.infrastructure.persistence.model_repository import save_model, load_production_model_info, update_production_model_info
from typing import List

class TrainModelUseCase:
    def execute(self, app_id: str, series_data: List[ForecastInputData], target_column: str) -> TrainResponse:
        """
        Executa o caso de uso de treinamento de um modelo Prophet com dados tabulares.

        Args:
            app_id (str): O ID base da aplicação.
            series_data (List[ForecastInputData]): Dados históricos para treino.
            target_column (str): A coluna alvo para o treinamento (ex: 'Autorizada').

        Returns:
            TrainResponse: Resposta da operação de treinamento.
        """
        if not series_data:
            raise ValueError("Os dados para treino não podem ser vazios.")

        # Converter Pydantic models para um DataFrame do Pandas
        df = pd.DataFrame([item.model_dump() for item in series_data])

        # Validar se a coluna alvo existe
        if target_column not in df.columns:
            raise ValueError(f"A coluna alvo '{target_column}' não foi encontrada nos dados de entrada.")

        # Pré-processamento dos dados
        try:
            df['ds'] = pd.to_datetime(df['DIA'] + ' ' + df['HORA'], format='%d/%m/%Y %H')
        except Exception as e:
            raise ValueError(f"Erro ao parsear 'DIA' e 'HORA': {e}.")

        df.rename(columns={target_column: 'y'}, inplace=True)
        df_prophet = df[['ds', 'y']]
        series = df_prophet.set_index('ds')['y']

        # O ID do modelo agora é uma combinação do app_id e da coluna alvo
        model_id = f"{app_id}_{target_column}"

        # Instanciar e treinar o modelo Prophet
        prophet_model = ProphetModel()
        prophet_model.train(series)

        # Avaliar o novo modelo
        new_model_metrics = prophet_model.evaluate(series)

        # Carregar informações do modelo em produção para este ID específico
        production_model_info = load_production_model_info(model_id)

        promote_new_model = False
        if production_model_info is None:
            promote_new_model = True
            promotion_reason = "Não há modelo em produção. Novo modelo promovido."
        else:
            current_production_rmse = production_model_info["metrics"]["rmse"]
            new_model_rmse = new_model_metrics["rmse"]
            if new_model_rmse < current_production_rmse:
                promote_new_model = True
                promotion_reason = f"Métricas do novo modelo ({new_model_rmse:.2f} RMSE) são melhores que o atual ({current_production_rmse:.2f} RMSE)."
            else:
                promotion_reason = f"Métricas do novo modelo ({new_model_rmse:.2f} RMSE) não são melhores que o atual ({current_production_rmse:.2f} RMSE)."

        if promote_new_model:
            model_version, saved_path = save_model(prophet_model, model_id, new_model_metrics)
            update_production_model_info(model_id, model_version, saved_path, new_model_metrics)
            message = f"Modelo '{model_id}' treinado e PROMOVIDO para produção (versão: {model_version}). {promotion_reason}"
        else:
            model_version = None # Nenhuma nova versão foi promovida
            message = f"Modelo '{model_id}' treinado, mas NÃO PROMOVIDO. {promotion_reason}"

        return TrainResponse(
            message=message,
            model_id=model_id,
            model_version=model_version,
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
