# src/use_cases/predict_forecast_use_case.py

import pandas as pd
from src.domain.models import ForecastInputData, PredictResponse
from src.infrastructure.persistence.model_repository import load_model, load_production_model_info
from typing import List

class PredictForecastUseCase:
    def execute(self, app_id: str, series_data: List[ForecastInputData], n_predict_steps: int, target_column: str) -> PredictResponse:
        """
        Executa o caso de uso de previsão de uma série temporal com base em dados tabulares.

        Args:
            app_id (str): O ID da aplicação para o qual o modelo foi treinado.
            series_data (List[ForecastInputData]): Dados históricos no formato de colunas.
            n_predict_steps (int): Número de passos futuros para prever.
            target_column (str): A coluna alvo para a previsão (ex: 'Autorizada').

        Returns:
            PredictResponse: Resposta contendo a previsão.
        """
        if not series_data:
            raise ValueError("Os dados da série temporal para previsão não podem ser vazios.")

        # Converter Pydantic models para um DataFrame do Pandas
        df = pd.DataFrame([item.model_dump() for item in series_data])

        # Validar se a coluna alvo existe no DataFrame
        if target_column not in df.columns:
            raise ValueError(f"A coluna alvo '{target_column}' não foi encontrada nos dados de entrada.")

        # Pré-processamento dos dados
        try:
            # Combinar DIA e HORA para criar o timestamp 'ds'
            df['ds'] = pd.to_datetime(df['DIA'] + ' ' + df['HORA'], format='%d/%m/%Y %H')
        except Exception as e:
            raise ValueError(f"Erro ao parsear 'DIA' e 'HORA': {e}. Verifique o formato.")

        # Renomear a coluna alvo para 'y'
        df.rename(columns={target_column: 'y'}, inplace=True)

        # Manter apenas as colunas que o Prophet utiliza ('ds', 'y')
        df_prophet = df[['ds', 'y']]

        # O ID do modelo agora inclui a coluna alvo para garantir que o modelo correto seja carregado
        model_id = f"{app_id}_{target_column}"

        # Carregar o modelo treinado
        try:
            production_model_info = load_production_model_info(model_id)
            if production_model_info is None:
                raise ValueError(f"Nenhum modelo em produção encontrado para '{model_id}'. Treine e promova um modelo primeiro.")
            
            prophet_model = load_model(model_id, production_model_info["version"])
            model_version_used = production_model_info["version"]
        except FileNotFoundError:
            raise ValueError(f"Modelo para '{model_id}' e versão '{production_model_info['version']}' não encontrado.")

        print(f"Previsão iniciada para app_id: {app_id}, target: {target_column}")
        
        # Realizar a previsão
        # O método predict do ProphetModel espera uma série, então vamos converter o dataframe
        series = df_prophet.set_index('ds')['y']
        forecast_series = prophet_model.predict(
            series=series,
            n_predict=n_predict_steps
        )
        
        # Converter o resultado para um dicionário com timestamps como strings
        forecast_dict = {str(k): v for k, v in forecast_series.to_dict().items()}

        return PredictResponse(
            forecast=forecast_dict,
            model_id=model_id,
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
