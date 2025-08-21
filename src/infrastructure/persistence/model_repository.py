# src/infrastructure/persistence/model_repository.py

import pickle
from pathlib import Path
from typing import Any

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model: Any, app_id: str, version: str = "v1") -> Path:
    """
    Salva um modelo treinado em um arquivo .pkl.

    Args:
        model (Any): O objeto do modelo a ser salvo.
        app_id (str): O ID da aplicação associado ao modelo.
        version (str): A versão do modelo (padrão: 'v1').

    Returns:
        Path: O caminho completo do arquivo onde o modelo foi salvo.
    """
    file_name = f"model-{app_id}-{version}.pkl"
    file_path = MODELS_DIR / file_name
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    return file_path

def load_model(app_id: str, version: str = "v1") -> Any:
    """
    Carrega um modelo de um arquivo .pkl.

    Args:
        app_id (str): O ID da aplicação associado ao modelo.
        version (str): A versão do modelo (padrão: 'v1').

    Returns:
        Any: O objeto do modelo carregado.

    Raises:
        FileNotFoundError: Se o arquivo do modelo não for encontrado.
    """
    file_name = f"model-{app_id}-{version}.pkl"
    file_path = MODELS_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {file_path}")
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    # Exemplo de uso (requer um objeto ProphetModel para testar)
    from src.infrastructure.forecasting_models.prophet_model import ProphetModel
    import pandas as pd

    print("Testando model_repository.py:")

    # Criar um modelo dummy para salvar
    dummy_model = ProphetModel()
    data = {
        '2023-01-01': 10.0,
        '2023-01-02': 12.0,
        '2023-01-03': 13.0,
    }
    series = pd.Series(data, index=pd.to_datetime(list(data.keys())))
    dummy_model.train(series)

    test_app_id = "test_app"
    test_version = "v1"

    # Salvar o modelo
    print(f"Salvando modelo para {test_app_id}...")
    saved_path = save_model(dummy_model, test_app_id, test_version)
    print(f"Modelo salvo em: {saved_path}")

    # Carregar o modelo
    print(f"Carregando modelo para {test_app_id}...")
    loaded_model = load_model(test_app_id, test_version)
    print(f"Modelo carregado: {loaded_model}")

    # Verificar se é o mesmo tipo de objeto
    print(f"Tipo do modelo carregado: {type(loaded_model)}")
    assert isinstance(loaded_model, ProphetModel)

    # Testar FileNotFoundError
    try:
        load_model("non_existent_app")
    except FileNotFoundError as e:
        print(f"\nErro esperado ao carregar modelo inexistente: {e}")
