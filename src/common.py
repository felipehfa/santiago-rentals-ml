"""Piezas compartidas: configuración, lectura de Gold, split, métricas y MLflow."""

import hashlib
import importlib.metadata
import logging
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
log = logging.getLogger("ml")
GOLD_MAX_HORAS = 36  # más viejo que esto, Silver y Gold dejaron de reconstruirse


def cargar_config(ruta: Path | None = None) -> dict:
    with open(ruta or RAIZ / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────────────

VARIABLES_SECRETAS = ("BD_STRING", "BD_STRING_READ", "BD_STRING_SCORING", "MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_ROOT",
                      "MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")


def limpiar_entorno() -> None:
    """Quita espacios y saltos de línea de los extremos de las credenciales. Al pegar un secret en GitHub es fácil
    arrastrar un salto de línea final: el nombre de la base pasa a incluirlo y la conexión falla sin una pista clara."""
    for nombre in VARIABLES_SECRETAS:
        valor = os.environ.get(nombre)
        if valor is not None and valor != valor.strip():
            os.environ[nombre] = valor.strip()


def _conexion_bd(variable: str = "BD_STRING_READ"):
    """Conecta a Supabase. Las credenciales salen del entorno o del archivo que apunte ENV_FILE; nunca
    se imprimen. BD_STRING_READ es el rol ml_lectura (entrenamiento); BD_STRING_SCORING, el ml_scoring
    (predicciones). Con ENV_FILE apuntando al .env del ELT, cae a BD_STRING (rol dueño)."""
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("ENV_FILE") or RAIZ / ".env")
    limpiar_entorno()
    cadena = os.environ.get(variable) or (os.environ.get("BD_STRING") if variable == "BD_STRING_READ" else None)
    if not cadena:
        raise RuntimeError(f"Falta {variable} (variable de entorno, .env o ENV_FILE)")
    return psycopg2.connect(cadena)


def descargar_gold(cfg: dict) -> pd.DataFrame:
    """Lee gold.aviso_entrenamiento (solo aptos) y lo guarda como snapshot local."""
    con = _conexion_bd()
    try:
        df = pd.read_sql(f"SELECT * FROM {cfg['datos']['tabla']} WHERE apto_entrenamiento", con)
    finally:
        con.close()
    if "construido_en" in df and len(df):
        edad_h = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(df["construido_en"], utc=True).max()).total_seconds() / 3600
        if edad_h > GOLD_MAX_HORAS:
            log.warning("Gold tiene %.0f h de antigüedad (límite %d h): revisar la reconstrucción de Silver y Gold", edad_h, GOLD_MAX_HORAS)
            print(f"::warning::Gold tiene {edad_h:.0f} h de antigüedad; revisar la reconstrucción de Silver y Gold")
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
    excluidas = cfg["datos"].get("comunas_excluidas") or []
    if excluidas:
        df = df[~df["comuna"].isin(excluidas)]
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


def baselines(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    """MAE en test de lo que haría una persona con una planilla, sin entrenar nada. El modelo debe superarlos:
      - mediana_comuna_dormitorios: mediana del costo total de los avisos de la misma comuna y cantidad de dormitorios;
      - m2_comuna: precio por m2 mediano de la comuna por la superficie útil del aviso.
    Los dos se calculan solo con `train` (en la práctica, train completo y validación). `media_train` (siempre la media)
    queda como cota trivial. Ante una combinación sin datos se cae a la comuna y luego a la mediana global."""
    y_tr, y_te = train[target].to_numpy(float), test[target].to_numpy(float)
    mediana_global = float(np.median(y_tr))

    def superficie(d: pd.DataFrame) -> pd.Series:
        return d["superficie_util_m2"].fillna(d["superficie_total_m2"])

    def dormitorios(d: pd.DataFrame) -> pd.Series:
        por_comuna = train.groupby("comuna")["dormitorios"].median()
        return d["dormitorios"].fillna(d["comuna"].map(por_comuna)).fillna(train["dormitorios"].median())

    out = {"media_train": float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())))}

    tr = train.assign(_dorm=dormitorios(train), _sup=superficie(train))
    te = test.assign(_dorm=dormitorios(test), _sup=superficie(test))
    por_comuna_dorm = tr.groupby(["comuna", "_dorm"])[target].median()
    por_comuna = tr.groupby("comuna")[target].median()
    pred = [por_comuna_dorm.get((c, d), por_comuna.get(c, mediana_global))
            for c, d in zip(te["comuna"], te["_dorm"], strict=True)]
    out["mediana_comuna_dormitorios"] = float(mean_absolute_error(y_te, pred))

    m2_comuna = (tr[target] / tr["_sup"]).groupby(tr["comuna"]).median()
    m2_global = float((tr[target] / tr["_sup"]).median())
    superficie_te = te["_sup"].fillna(te["_sup"].median())
    pred = (te["comuna"].map(m2_comuna).fillna(m2_global) * superficie_te).fillna(mediana_global)
    out["m2_comuna"] = float(mean_absolute_error(y_te, pred))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────────────────────────────────────

def setup_mlflow(cfg: dict) -> None:
    """Sin MLFLOW_TRACKING_URI usa SQLite local. Cambiar a Supabase no toca el código."""
    limpiar_entorno()
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri_local"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(RAIZ / uri.removeprefix("sqlite:///")).replace("\\", "/")
    mlflow.set_tracking_uri(uri)
    nombre = cfg["mlflow"]["experimento"]
    raiz = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if raiz and mlflow.get_experiment_by_name(nombre) is None:
        # Con un backend en base de datos, los artefactos van al bucket solo si el experimento nace con esa ruta.
        mlflow.create_experiment(nombre, artifact_location=f"{raiz.rstrip('/')}/{nombre}")
    mlflow.set_experiment(nombre)


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
