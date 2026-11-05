"""Degradación del @champion con avisos llegados después de entrenarlo.

Evalúa el champion sobre los avisos que nunca vio (ni train, ni val, ni test) y compara su MAPE
con el MAPE de test de cuando se entrenó. Se usa el error porcentual y no el MAE en pesos porque los
avisos nuevos no tienen la mezcla de precios del test (que es estratificado): un día con más avisos
caros sube el MAE aunque el modelo acierte igual (el 3-nov, v16 marcaba +16 % de MAE con el MAPE
casi igual). Desglosa por semana de primera aparición, para ver si el error crece con el tiempo.
Registra todo en un run de MLflow (etapa=monitoreo).

Solo concluye con avisos nuevos suficientes (`n_minimo`) repartidos en al menos `dias_minimos` días
distintos de primera aparición: una sola grilla trae avisos de un solo día y de una mezcla particular.

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


def dias_distintos(df: pd.DataFrame) -> int:
    return int(df["primera_vez_visto"].dt.tz_convert("America/Santiago").dt.date.nunique())


def falta_muestra(n: int, dias: int, mon: dict) -> str | None:
    """Motivo por el que aún no se puede concluir, o None si la muestra alcanza."""
    if n < mon["n_minimo"]:
        return f"solo {n} avisos nuevos (mínimo {mon['n_minimo']})"
    if dias < mon.get("dias_minimos", 1):
        return f"avisos nuevos de solo {dias} días distintos (mínimo {mon['dias_minimos']})"
    return None


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
    dias = dias_distintos(df)
    motivo = falta_muestra(len(df), dias, cfg["monitoreo"])
    if motivo:
        print(f"Después de entrenar v{champ.version} llegaron {motivo}: aún no se puede medir degradación.")
        return 1

    m, pred = comparacion.evaluar(mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_CHAMPION}"), df, target)
    mae_original, mape_original = run.data.metrics["test_mae"], run.data.metrics["test_mape"]
    degradacion = m["mape"] / mape_original - 1
    por_semana = error_por_semana(df, pred, target)
    umbral = cfg["monitoreo"]["degradacion_umbral"]
    degradado = degradacion > umbral

    with mlflow.start_run(run_name=f"monitoreo_v{champ.version}"):
        mlflow.set_tags({"etapa": "monitoreo", "version_champion": champ.version,
                         "degradado": str(degradado).lower(), "metrica_degradacion": "mape"})
        mlflow.log_metrics({"n_nuevos": len(df), "dias_nuevos": dias, "mae_nuevos": m["mae"],
                            "mape_nuevos": m["mape"], "mae_original_test": mae_original,
                            "mape_original_test": mape_original, "degradacion": degradacion,
                            "degradacion_mae": m["mae"] / mae_original - 1})
        for i, f in por_semana.iterrows():
            mlflow.log_metrics({"mae_semana": f["mae"], "mape_semana": f["mape"], "n_semana": f["n"]}, step=i)
        mlflow.log_dict(por_semana.to_dict(orient="records"), "error_por_semana.json")

    print(f"v{champ.version} sobre {len(df)} avisos nuevos de {dias} días: MAPE {m['mape']:.1f}% vs "
          f"{mape_original:.1f}% original ({degradacion:+.1%}); MAE {m['mae']:,.0f} vs {mae_original:,.0f} "
          "(en pesos depende de la mezcla de precios).")
    print(por_semana.round(1).to_string(index=False))
    if degradado:
        print(f"DEGRADADO (MAPE sube más de {umbral:.0%}): conviene reentrenar.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
