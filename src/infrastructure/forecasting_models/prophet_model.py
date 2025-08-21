# src/infrastructure/forecasting_models/prophet_model.py

import pandas as pd
from prophet import Prophet

class ProphetModel:
    def __init__(self):
        self.model = None

    def train(self, series: pd.Series):
        """
        Treina o modelo Prophet com a série temporal fornecida.
        Args:
            series (pd.Series): A série temporal histórica (índice de tempo, valores).
                                O índice deve ser de tipo datetime.
        """
        if series.empty:
            raise ValueError("A série temporal para treino não pode ser vazia.")

        df = series.reset_index()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])

        self.model = Prophet()
        self.model.fit(df)

    def predict(self, series: pd.Series, n_predict: int) -> pd.Series:
        """
        Realiza a previsão de uma série temporal usando o modelo Prophet treinado.
        Args:
            series (pd.Series): A série temporal histórica (índice de tempo, valores).
                                O índice deve ser de tipo datetime.
            n_predict (int): O número de passos futuros para prever.
        Returns:
            pd.Series: A série temporal prevista, com índice de tempo.
        """
        if self.model is None:
            raise RuntimeError("O modelo Prophet não foi treinado. Chame .train() primeiro.")
        if n_predict < 0:
            raise ValueError("O número de passos para prever (n_predict) não pode ser negativo.")
        if series.empty:
            raise ValueError("A série de entrada para previsão não pode ser vazia.")

        if len(series) < 2:
            raise ValueError("A série de entrada para previsão deve conter pelo menos 2 pontos de dados para inferir a frequência.")

        # Criar um DataFrame para datas futuras a partir da última data da série de entrada
        # Isso garante que as previsões se estendam a partir do final da 'series' fornecida
        last_date_in_series = series.index.max()

        # Inferir frequência da série de entrada para gerar corretamente as datas futuras
        freq = pd.infer_freq(series.index)
        if freq is None:
            # Se a frequência não puder ser inferida, assumimos diária ('D') como padrão.
            # Para um serviço genérico robusto, pode ser necessário um tratamento mais sofisticado
            # para dados irregulares ou exigir uma frequência consistente.
            print("Aviso: Não foi possível inferir a frequência da série de entrada. Assumindo diária ('D') para previsões futuras.")
            freq = 'D'

        # Gerar apenas as datas futuras, começando *após* a última data na série de entrada
        # periods=n_predict irá gerar exatamente n_predict datas
        future_dates_only = pd.date_range(start=last_date_in_series, periods=n_predict + 1, freq=freq)[1:]

        # Combinar as datas da série de entrada com as datas futuras recém-geradas
        # O método predict do Prophet espera um DataFrame com a coluna 'ds'
        full_future_df = pd.DataFrame({'ds': series.index.union(future_dates_only)})
        full_future_df['ds'] = pd.to_datetime(full_future_df['ds']) # Garantir tipo datetime

        # Gerar previsões para todas as datas em full_future_df
        forecast = self.model.predict(full_future_df)

        # Filtrar para obter apenas as novas previsões futuras
        future_forecast = forecast[forecast['ds'] > last_date_in_series]

        # Retornar como um pandas Series com o timestamp como índice
        return pd.Series(future_forecast['yhat'].values, index=future_forecast['ds'])

if __name__ == "__main__":
    # Exemplo de uso
    print("Testando a classe ProphetModel:")
    data = {
        '2023-01-01': 10.0,
        '2023-01-02': 12.0,
        '2023-01-03': 13.0,
        '2023-01-04': 15.0,
        '2023-01-05': 14.0,
        '2023-01-06': 16.0,
        '2023-01-07': 18.0,
        '2023-01-08': 17.0,
        '2023-01-09': 19.0,
        '2023-01-10': 20.0,
    }
    series = pd.Series(data, index=pd.to_datetime(list(data.keys())))

    model = ProphetModel()
    model.train(series)

    n_forecast_steps = 5
    # Teste com uma série de entrada que é um subconjunto da série de treino
    predict_series_subset = pd.Series(data={
        '2023-01-08': 17.0,
        '2023-01-09': 19.0,
        '2023-01-10': 20.0,
    }, index=pd.to_datetime(['2023-01-08', '2023-01-09', '2023-01-10']))

    forecast_result = model.predict(predict_series_subset, n_forecast_steps)

    print("\nSérie Original (para treino):")
    print(series)
    print("\nSérie de entrada para previsão:")
    print(predict_series_subset)
    print(f"\nPrevisão Prophet ({n_forecast_steps} passos):")
    print(forecast_result)

    # Exemplo com série vazia para treino
    try:
        empty_model = ProphetModel()
        empty_series = pd.Series([], dtype=float)
        empty_model.train(empty_series)
    except ValueError as e:
        print(f"\nErro esperado para treino com série vazia: {e}")

    # Exemplo de previsão sem treino
    try:
        un_trained_model = ProphetModel()
        un_trained_model.predict(series, 3)
    except RuntimeError as e:
        print(f"\nErro esperado para previsão sem treino: {e}")
