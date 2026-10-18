"""Piezas compartidas: configuración, lectura de Gold, split, métricas y MLflow."""

import hashlib
import importlib.metadata
import os
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
ID_COL = "id_aviso"


def cargar_config(ruta: Path | None = None) -> dict:
    with open(ruta or RAIZ / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────────────

def _conexion_bd():
    """Conecta a Supabase. Las credenciales salen del entorno (BD_STRING) o del archivo que
    apunte ENV_FILE; nunca se imprimen."""
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("ENV_FILE") or RAIZ / ".env")
    cadena = os.environ.get("BD_STRING")
    if not cadena:
        raise RuntimeError("Falta BD_STRING (variable de entorno, .env o ENV_FILE)")
    return psycopg2.connect(cadena)


def descargar_gold(cfg: dict) -> pd.DataFrame:
    """Lee gold.aviso_entrenamiento (solo aptos) y lo guarda como snapshot local."""
    con = _conexion_bd()
    try:
        df = pd.read_sql(f"SELECT * FROM {cfg['datos']['tabla']} WHERE apto_entrenamiento", con)
    finally:
        con.close()
    ruta = RAIZ / cfg["datos"]["snapshot"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    return df


def cargar_gold(cfg: dict, refrescar: bool = False) -> pd.DataFrame:
    """Usa el snapshot local si existe (reproducible entre corridas); si no, descarga de Gold."""
    ruta = RAIZ / cfg["datos"]["snapshot"]
    if refrescar or not ruta.exists():
        return descargar_gold(cfg)
    return pd.read_parquet(ruta)


def preparar_datos(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filtros de cordura sobre el target. Silver no marca gastos comunes absurdos (ver config)."""
    target = cfg["target"]
    df = df[df[target].notna() & (df[target] > 0)]
    df = df[df[target] <= cfg["datos"]["costo_total_max_clp"]]
    # Sin gastos comunes registrados (NaN) el aviso se conserva: ya lo marca gastos_comunes_faltante.
    maximo = cfg["datos"].get("gastos_comunes_max_pct_arriendo")
    if maximo is not None:
        df = df[~(df["gastos_comunes"] / df["precio_clp"] > maximo)]
    return df.reset_index(drop=True)


def huella_datos(df: pd.DataFrame, target: str) -> dict:
    """Identifica el dataset exacto de una corrida, para poder reproducirla."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df[[ID_COL, target]], index=False).values.tobytes())
    return {"n_filas": len(df), "huella": h.hexdigest()[:16]}


# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────

def estratos_precio(y: pd.Series, n: int) -> pd.Series:
    return pd.qcut(y, q=n, labels=False, duplicates="drop")


def dividir(df: pd.DataFrame, cfg: dict, semilla: int | None = None):
    """Train/val/test estratificado por quintil de costo total (como en Concepción).

    Se separa primero el test y luego la validación, de modo que las proporciones finales
    sean las pedidas sobre el total."""
    semilla = cfg["semilla"] if semilla is None else semilla
    t, v = cfg["split"]["test"], cfg["split"]["val"]
    estratos = estratos_precio(df[cfg["target"]], cfg["split"]["n_estratos"])
    resto, test = train_test_split(df, test_size=t, random_state=semilla, shuffle=True, stratify=estratos)
    train, val = train_test_split(
        resto, test_size=v / (1 - t), random_state=semilla, shuffle=True, stratify=estratos.loc[resto.index]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def mae_quintil_caro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE del quintil de costo más alto: el segmento donde se concentra el error."""
    q = pd.qcut(y_true, q=5, labels=False, duplicates="drop")
    mascara = q == q.max()
    return float(mean_absolute_error(y_true[mascara], y_pred[mascara]))


def metricas(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    ape = np.abs(y_true - y_pred) / y_true * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(ape.mean()),
        "mdape": float(np.median(ape)),
        "pct_dentro_10": float((ape <= 10).mean() * 100),
        "mae_quintil_caro": mae_quintil_caro(y_true, y_pred),
    }


def baselines(y_train: pd.Series, y_true: np.ndarray, X: pd.DataFrame) -> dict:
    """Referencias que el modelo debe superar: la media de train y el precio por m2 del sector."""
    out = {"media_train": float(mean_absolute_error(y_true, np.full(len(y_true), y_train.mean())))}
    if {"precio_m2_sector", "superficie_util_m2"} <= set(X.columns):
        pred = (X["precio_m2_sector"] * X["superficie_util_m2"]).fillna(y_train.median())
        out["mercado_m2_sector"] = float(mean_absolute_error(y_true, pred))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────────────────────────────────────

def setup_mlflow(cfg: dict) -> None:
    """Sin MLFLOW_TRACKING_URI usa SQLite local. Cambiar a Supabase no toca el código."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri_local"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(RAIZ / uri.removeprefix("sqlite:///")).replace("\\", "/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])


def registrar_reproducibilidad() -> None:
    """Deja en el run activo lo necesario para repetirlo: config.yaml, commit de git (y si el árbol
    tenía cambios sin commitear) y versiones de las librerías que afectan al resultado."""
    mlflow.log_artifact(str(RAIZ / "config.yaml"))
    tags = {}
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True, check=True).stdout  # noqa: E731
        tags["git_commit"] = git("rev-parse", "HEAD").strip()
        tags["git_arbol_sucio"] = str(bool(git("status", "--porcelain").strip())).lower()
    except (OSError, subprocess.CalledProcessError):
        tags["git_commit"] = "desconocido"  # sin commits todavía o sin git disponible
    for paquete in ("xgboost", "lightgbm", "scikit-learn", "pandas", "numpy", "optuna", "mlflow"):
        tags[f"version_{paquete}"] = importlib.metadata.version(paquete)
    mlflow.set_tags(tags)
