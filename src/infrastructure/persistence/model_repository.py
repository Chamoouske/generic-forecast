# src/infrastructure/persistence/model_repository.py

import joblib
import os
import json
import logging
from datetime import datetime
from typing import Any

MODELS_DIR = "models"
PRODUCTION_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "production_model.json")

os.makedirs(MODELS_DIR, exist_ok=True)

def _get_next_version(app_id: str) -> str:
    """
    Determina a próxima versão incremental para um dado app_id.
    """
    current_version_num = 0
    if os.path.exists(PRODUCTION_MODEL_INFO_PATH):
        try:
            with open(PRODUCTION_MODEL_INFO_PATH, 'r') as f:
                production_info = json.load(f)
                app_info = production_info.get(app_id)
                if app_info and "version" in app_info:
                    # Extrai o número da versão (ex: "v1" -> 1)
                    try:
                        current_version_num = int(app_info["version"].replace("v", ""))
                    except ValueError:
                        # Se a versão não for numérica (ex: v2023...), reinicia a contagem
                        current_version_num = 0
        except json.JSONDecodeError:
            pass # Arquivo corrompido ou vazio, tratar como se não houvesse versão anterior

    next_version_num = current_version_num + 1
    return f"v{next_version_num}"

def save_model(model: Any, app_id: str, metrics: dict) -> tuple[str, str]:
    """
    Salva um modelo treinado em um arquivo .joblib com versionamento baseado em timestamp.

    Args:
        model (Any): O objeto do modelo a ser salvo.
        app_id (str): O ID da aplicação associado ao modelo.
        metrics (dict): Dicionário de métricas de avaliação do modelo.

    Returns:
        tuple[str, str]: Uma tupla contendo a versão do modelo e o caminho completo do arquivo.
    """
    version = _get_next_version(app_id)
    file_name = f"{app_id}_model_{version}.joblib"
    file_path = os.path.join(MODELS_DIR, file_name)

    update_production_model_info(app_id, version, file_path, metrics)

    joblib.dump(model, file_path)
    return version, file_path

def load_model(app_id: str, version: str) -> Any:
    """
    Carrega um modelo de um arquivo .joblib com base no app_id e versão.

    Args:
        app_id (str): O ID da aplicação associado ao modelo.
        version (str): A versão específica do modelo a ser carregada.

    Returns:
        Any: O objeto do modelo carregado.

    Raises:
        FileNotFoundError: Se o arquivo do modelo não for encontrado.
    """
    file_name = f"{app_id}_model_{version}.joblib"
    file_path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Modelo não encontrado: {file_path}")
    return joblib.load(file_path)

def load_production_model_info(app_id: str) -> dict | None:
    """
    Carrega as informações do modelo em produção para um dado app_id.
    Retorna None se o arquivo não existir ou se o app_id não for encontrado.
    """
    if not os.path.exists(PRODUCTION_MODEL_INFO_PATH):
        return None
    try:
        with open(PRODUCTION_MODEL_INFO_PATH, 'r') as f:
            production_info = json.load(f)
            return production_info.get(app_id)
    except json.JSONDecodeError:
        logging.info(f"Erro ao decodificar JSON em {PRODUCTION_MODEL_INFO_PATH}. O arquivo pode estar corrompido.")
        return None
    except Exception as e:
        logging.info(f"Erro ao carregar informações do modelo de produção: {e}")
        return None

def update_production_model_info(app_id: str, version: str, path: str, metrics: dict):
    """
    Atualiza as informações do modelo em produção no arquivo JSON.
    """
    logging.info(f"DEBUG: update_production_model_info started for app_id: {app_id}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    production_info = {}
    if os.path.exists(PRODUCTION_MODEL_INFO_PATH):
        logging.info(f"DEBUG: {PRODUCTION_MODEL_INFO_PATH} exists. Attempting to read.")
        try:
            with open(PRODUCTION_MODEL_INFO_PATH, 'r') as f:
                production_info = json.load(f)
            logging.info(f"DEBUG: Successfully read {PRODUCTION_MODEL_INFO_PATH}.")
        except json.JSONDecodeError:
            logging.info(f"DEBUG: Warning: {PRODUCTION_MODEL_INFO_PATH} is corrupted or empty. Creating a new one.")
            production_info = {} # Reset if corrupted
        except Exception as e:
            logging.info(f"DEBUG: Error reading {PRODUCTION_MODEL_INFO_PATH}: {e}")
            production_info = {} # Reset on any read error

    production_info[app_id] = {
        "version": version,
        "path": path,
        "metrics": metrics,
        "timestamp_promoted": datetime.now().isoformat()
    }

    logging.info(f"DEBUG: Attempting to write to {PRODUCTION_MODEL_INFO_PATH}.")
    try:
        with open(PRODUCTION_MODEL_INFO_PATH, 'w') as f:
            json.dump(production_info, f, indent=4)
        logging.info(f"DEBUG: Successfully wrote to {PRODUCTION_MODEL_INFO_PATH}.")
    except Exception as e:
        logging.info(f"DEBUG: Error writing to {PRODUCTION_MODEL_INFO_PATH}: {e}")

if __name__ == "__main__":
    # Exemplo de uso (requer um objeto ProphetModel para testar)
    from src.infrastructure.forecasting_models.prophet_model import ProphetModel
    import pandas as pd

    logging.info("Testando model_repository.py:")

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
    logging.info(f"Salvando modelo para {test_app_id}...")
    saved_path = save_model(dummy_model, test_app_id, test_version)
    logging.info(f"Modelo salvo em: {saved_path}")

    # Carregar o modelo
    logging.info(f"Carregando modelo para {test_app_id}...")
    loaded_model = load_model(test_app_id, test_version)
    logging.info(f"Modelo carregado: {loaded_model}")

    # Verificar se é o mesmo tipo de objeto
    logging.info(f"Tipo do modelo carregado: {type(loaded_model)}")
    assert isinstance(loaded_model, ProphetModel)

    # Testar FileNotFoundError
    try:
        load_model("non_existent_app")
    except FileNotFoundError as e:
        logging.info(f"\nErro esperado ao carregar modelo inexistente: {e}")
