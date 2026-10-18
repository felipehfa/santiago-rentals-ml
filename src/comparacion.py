"""Comparación de modelos entrenados con datos distintos.

Los datos crecen con el tiempo, así que dos modelos registrados no comparten test. Para compararlos
se evalúan ambos sobre las MISMAS filas: el test del challenger sin los avisos con que se entrenó el
champion (train y val). Así ninguno es evaluado con datos que ya vio.

También mide la degradación: cuánto empeora el MAE del champion en esas filas respecto de su propio
MAE de test cuando se entrenó.
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from .common import ID_COL, cargar_gold, metricas, preparar_datos


def ids_de_run(run_id: str) -> dict | None:
    """Ids de train/val/test de un run, o None si el run es anterior a que se guardaran."""
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/ids.json")
    except (MlflowException, OSError):
        return None


def filas_comunes(ids_challenger: dict, ids_champion: dict) -> set:
    """Test del challenger menos lo que el champion usó para entrenar o para early stopping."""
    vistos = set(ids_champion["train"]) | set(ids_champion["val"])
    return set(ids_challenger["test"]) - vistos


def filas_nuevas(ids_modelo: dict, ids_actuales) -> set:
    """Avisos que el modelo nunca vio, ni siquiera como test: llegaron después de entrenarlo."""
    vistos = set(ids_modelo["train"]) | set(ids_modelo["val"]) | set(ids_modelo["test"])
    return set(ids_actuales) - vistos


def datos_actuales(cfg: dict) -> pd.DataFrame:
    return preparar_datos(cargar_gold(cfg), cfg)


def evaluar(modelo, df: pd.DataFrame, target: str) -> tuple[dict, np.ndarray]:
    pred = modelo.predict(df)["prediccion_clp"].to_numpy()
    return metricas(df[target].to_numpy(), pred), pred


def diferencia_ic(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, semilla: int = 0):
    """MAE(a) - MAE(b) sobre las mismas filas, con intervalo de confianza del 95 % por bootstrap.
    Positivo = b es mejor. Si el intervalo cruza 0, la diferencia puede ser ruido."""
    dif = np.abs(y - pred_a) - np.abs(y - pred_b)
    rng = np.random.default_rng(semilla)
    medias = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(n_boot)]
    return float(dif.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


__all__ = ["ID_COL", "datos_actuales", "diferencia_ic", "evaluar", "filas_comunes", "filas_nuevas", "ids_de_run"]
