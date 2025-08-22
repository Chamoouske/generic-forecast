# src/infrastructure/persistence/data_persistance.py

from pathlib import Path
import pandas as pd

MODELS_DIR = Path("data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(data_filename: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.

    Args:
        data_filename (str): O nome do arquivo CSV a ser carregado (ex: "data_nfae.csv").

    Returns:
        pd.DataFrame: Um DataFrame do pandas contendo os dados do CSV.
    """
    file_path = MODELS_DIR / (data_filename + ".csv")
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {file_path}")
    with open(file_path, "r") as f:
        df = pd.read_csv(f, parse_dates=['timestamp'], date_format='mixed', dayfirst=True, sep=';')
    return df