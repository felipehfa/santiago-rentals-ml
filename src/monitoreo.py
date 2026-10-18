"""Degradación del @champion con avisos llegados después de entrenarlo.

Evalúa el champion sobre los avisos que nunca vio (ni train, ni val, ni test) y compara su MAE
con el MAE de test de cuando se entrenó. Desglosa por semana de primera aparición, para ver si el
error crece con el tiempo (los datos nuevos se parecen cada vez menos a los de entrenamiento).
Registra todo en un run de MLflow (etapa=monitoreo).

Códigos de salida: 0 normal, 1 sin champion o sin avisos nuevos suficientes, 3 degradado.
Un 3 indica que conviene reentrenar (`make train tune register promote`).
"""

import sys

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from . import comparacion
from .common import ID_COL, cargar_config, setup_mlflow
from .promote import ALIAS_CHAMPION, _version


def error_por_semana(df: pd.DataFrame, pred, target: str) -> pd.DataFrame:
    semana = df["primera_vez_visto"].dt.tz_localize(None).dt.to_period("W").astype(str)
    real = df[target].to_numpy()
    e = pd.DataFrame({"semana": semana.values, "ae": np.abs(real - pred), "ape": np.abs(real - pred) / real * 100})
    return e.groupby("semana").agg(n=("ae", "size"), mae=("ae", "mean"), mape=("ape", "mean")).reset_index()


def main() -> int:
    cfg = cargar_config()
    setup_mlflow(cfg)
    nombre, target = cfg["mlflow"]["modelo_registrado"], cfg["target"]
    champ, run = _version(MlflowClient(), nombre, ALIAS_CHAMPION)
    if champ is None:
        print("Sin @champion que monitorear.")
        return 1
    ids = comparacion.ids_de_run(champ.run_id)
    if ids is None:
        print("El champion no tiene ids.json; ver `python -m src.reconstruir_ids`.")
        return 1

    df = comparacion.datos_actuales(cfg)
    df = df[df[ID_COL].isin(comparacion.filas_nuevas(ids, df[ID_COL]))]
    if len(df) < cfg["monitoreo"]["n_minimo"]:
        print(f"Solo {len(df)} avisos llegaron después de entrenar v{champ.version} "
              f"(mínimo {cfg['monitoreo']['n_minimo']}): aún no se puede medir degradación.")
        return 1

    m, pred = comparacion.evaluar(mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_CHAMPION}"), df, target)
    mae_original = run.data.metrics["test_mae"]
    degradacion = m["mae"] / mae_original - 1
    por_semana = error_por_semana(df, pred, target)
    degradado = degradacion > cfg["gate"]["degradacion_umbral"]

    with mlflow.start_run(run_name=f"monitoreo_v{champ.version}"):
        mlflow.set_tags({"etapa": "monitoreo", "version_champion": champ.version,
                         "degradado": str(degradado).lower()})
        mlflow.log_metrics({"n_nuevos": len(df), "mae_nuevos": m["mae"], "mape_nuevos": m["mape"],
                            "mae_original_test": mae_original, "degradacion": degradacion})
        for i, f in por_semana.iterrows():
            mlflow.log_metrics({"mae_semana": f["mae"], "mape_semana": f["mape"], "n_semana": f["n"]}, step=i)
        mlflow.log_dict(por_semana.to_dict(orient="records"), "error_por_semana.json")

    print(f"v{champ.version} sobre {len(df)} avisos nuevos: MAE {m['mae']:,.0f} vs {mae_original:,.0f} original "
          f"({degradacion:+.1%}), MAPE {m['mape']:.1f}%.")
    print(por_semana.round(1).to_string(index=False))
    if degradado:
        print(f"DEGRADADO (umbral {cfg['gate']['degradacion_umbral']:.0%}): conviene reentrenar.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
