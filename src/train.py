"""Paso 1: entrena los modelos base (hiperparámetros por defecto) y registra todo en MLflow.

Es también la base que usa tune.py para el ajuste final: `registrar_modelo_final` entrena el
ensamble de bagging, evalúa en val y test y guarda el modelo en el run activo.
"""

import argparse
import hashlib
import json
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import etiquetas
from .common import (
    ID_COL,
    RAIZ,
    baselines,
    cargar_config,
    cargar_gold,
    dividir,
    huella_datos,
    metricas,
    preparar_datos,
    registrar_reproducibilidad,
    setup_mlflow,
)
from .features import ConstructorVariables
from .models import ALGORITMOS, PARAMS_BASE, Ensamble, ModeloPrecio, entrenar_uno, predecir_uno


def nuevo_constructor(cfg: dict) -> ConstructorVariables:
    v = cfg["variables"]
    return ConstructorVariables(
        radio_sector_m=v["radio_sector_m"], multiplicador_iqr=v["multiplicador_iqr"],
        k_suavizado=v["k_suavizado"], n_niveles_barrio=v["n_niveles_barrio"], target=cfg["target"],
        puntos_referencia=v.get("puntos_referencia"), texto=v.get("texto"),
    )


def aplicar_ventana(train: pd.DataFrame, df: pd.DataFrame, dias: int | None) -> pd.DataFrame:
    """Deja en train solo los avisos publicados hasta `dias` días antes de la última visita registrada en
    los datos (referencia reproducible, no la fecha de hoy). Sin fecha de publicación se conservan."""
    if not dias:
        return train
    ref = pd.to_datetime(df["ultima_vez_visto"], utc=True).max()
    edad = (ref - pd.to_datetime(train["fecha_publicacion_aprox"], utc=True)).dt.days
    return train[edad.isna() | (edad <= dias)].reset_index(drop=True)


def variables_seleccionadas(cfg: dict, disponibles: list, huella: str | None = None) -> list:
    """Lista de `features_seleccionadas.json`; si no existe todavía, todas las variables.

    Si se pasa la huella de los datos y la lista se calculó con otros, falla: una selección vieja
    aplicada a datos nuevos pasa inadvertida (python -m src.seleccion_variables la rehace)."""
    ruta = RAIZ / cfg["variables"]["seleccion"]
    if not ruta.exists():
        return list(disponibles)
    guardada = json.loads(ruta.read_text(encoding="utf-8"))
    if huella and guardada.get("huella_datos") not in (None, huella):
        raise ValueError("features_seleccionadas.json se calculó con otros datos "
                         f"({guardada['huella_datos']} y los actuales son {huella}); "
                         "rehacerla con `python -m src.seleccion_variables` o usar `python -m src.tune`")
    elegidas = guardada["features"]
    faltan = [c for c in elegidas if c not in disponibles]
    if faltan:
        raise ValueError(f"Variables seleccionadas que ya no existen: {faltan}")
    return elegidas


def preparar(cfg: dict, refrescar: bool = False) -> dict:
    """Carga Gold, aplica filtros, divide y construye las variables ajustando SOLO con train."""
    df = preparar_datos(cargar_gold(cfg, refrescar), cfg)
    train_completo, val, test = dividir(df, cfg)
    train = aplicar_ventana(train_completo, df, cfg["datos"].get("ventana_dias"))
    constructor = nuevo_constructor(cfg).fit(train)
    X = {"train": constructor.transform(train), "val": constructor.transform(val),
         "test": constructor.transform(test)}
    t = cfg["target"]
    huella = huella_datos(df, t)
    variables = variables_seleccionadas(cfg, list(X["train"].columns), huella["huella"])
    return {
        "df": df, "train": train, "train_completo": train_completo, "val": val, "test": test,
        "constructor": constructor, "variables": variables,
        "X_train": X["train"][variables], "X_val": X["val"][variables],
        "X_test": X["test"][variables], "X_completo_train": X["train"], "X_completo_test": X["test"],
        "y_train": train[t].astype(float), "y_val": val[t].astype(float), "y_test": test[t].astype(float),
        "huella": huella,
    }


def construir_folds(cfg: dict, train: pd.DataFrame, variables: list) -> list:
    """Folds de CV sobre train. El constructor de variables se reajusta en cada fold, para que ni
    el comparador del sector ni los niveles de barrio vean las filas de validación del fold."""
    kf = KFold(n_splits=cfg["optuna"]["cv_splits"], shuffle=True, random_state=cfg["semilla"])
    t = cfg["target"]
    folds = []
    for tr_idx, va_idx in kf.split(train):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        c = nuevo_constructor(cfg).fit(tr)
        folds.append((c.transform(tr)[variables], tr[t].astype(float).reset_index(drop=True),
                      c.transform(va)[variables], va[t].astype(float).reset_index(drop=True)))
    return folds


def cv_mae(cfg: dict, algoritmo: str, params: dict, folds: list) -> float:
    o = cfg["optuna"]
    maes = []
    for X_tr, y_tr, X_va, y_va in folds:
        m = entrenar_uno(algoritmo, params, X_tr, y_tr, X_va, y_va, cfg["semilla"],
                         o["max_rondas"], o["early_stopping"])
        pred = np.clip(predecir_uno(algoritmo, m, X_va), 0, None)
        maes.append(float(np.mean(np.abs(y_va - pred))))
    return float(np.mean(maes))


def registrar_modelo_final(cfg: dict, algoritmo: str, params: dict, d: dict) -> dict:
    """Entrena el ensamble de bagging con `params`, evalúa y registra en el run activo."""
    b = cfg["bagging"]
    t0 = time.time()
    ens = Ensamble(algoritmo, params).fit(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"], b["n_modelos"], cfg["semilla"],
        b["max_rondas"], b["early_stopping"],
    )
    segundos = time.time() - t0

    pred = {k: ens.predecir(d[f"X_{k}"]) for k in ("val", "test")}
    res = {}
    for k in ("val", "test"):
        for nombre, valor in metricas(d[f"y_{k}"], pred[k]).items():
            res[f"{k}_{nombre}"] = valor
    for nombre, valor in baselines(d["y_train"], d["y_test"].values, d["X_completo_test"]).items():
        res[f"baseline_{nombre}_mae"] = valor
    res["ganancia_vs_mejor_baseline_mae"] = min(
        v for k, v in res.items() if k.startswith("baseline_")) - res["test_mae"]
    res["tiempo_bagging_seg"] = segundos
    mlflow.log_metrics(res)

    mlflow.log_params({f"p_{k}": v for k, v in params.items()})
    mlflow.log_params({"algoritmo": algoritmo, "n_variables": len(d["variables"]),
                       "n_modelos_bagging": b["n_modelos"], "semilla": cfg["semilla"],
                       "n_train": len(d["train"]), "n_train_sin_ventana": len(d["train_completo"]),
                       "ventana_dias": cfg["datos"].get("ventana_dias"),
                       "n_val": len(d["val"]), "n_test": len(d["test"])})
    # Dos runs solo son comparables (cv_mae) si usaron los mismos datos y las mismas variables.
    huella_vars = hashlib.sha256(json.dumps(d["variables"]).encode()).hexdigest()[:8]
    mlflow.set_tags({"algoritmo": algoritmo, "huella_datos": d["huella"]["huella"],
                     "huella_experimento": f"{d['huella']['huella']}-{huella_vars}",
                     "n_filas_datos": str(d["huella"]["n_filas"])})
    registrar_reproducibilidad()
    mlflow.log_dict({"variables": d["variables"]}, "variables.json")
    # Qué avisos vio cada conjunto: permite evaluar después a otro modelo solo con avisos que no usó.
    # "train" lista también los avisos que la ventana dejó fuera del ajuste: el modelo los conoce como
    # parte de su conjunto y no deben contarse como "avisos nuevos" en el gate ni en el monitoreo.
    mlflow.log_dict({"train": d["train_completo"][ID_COL].tolist(), "val": d["val"][ID_COL].tolist(),
                     "test": d["test"][ID_COL].tolist()}, "ids.json")
    mlflow.log_dict({k: float(v) for k, v in ens.shap_nativo(d["X_test"]).items()}, "shap_importancia.json")

    # Referencia del etiquetado: residuos de validación (la validación solo decide el early
    # stopping; el test queda intacto para el gate).
    e = cfg["etiquetas"]
    ref = etiquetas.ajustar_referencia(d["y_val"].values, pred["val"], e["n_deciles"])
    mlflow.log_dict(ref, "referencia_etiquetas.json")
    m_test = ens.matriz(d["X_test"])
    et = etiquetas.etiquetar(d["y_test"].values, m_test.mean(axis=0),
                             m_test.std(axis=0) / np.maximum(m_test.mean(axis=0), 1e-6), ref,
                             e["umbral_oportunidad"], e["umbral_caro"])
    mlflow.log_dict(et["etiqueta"].value_counts(normalize=True).round(4).to_dict(), "etiquetas_test.json")

    modelo = ModeloPrecio(d["constructor"], ens, d["variables"])
    mlflow.pyfunc.log_model(
        name="model", python_model=modelo, code_paths=[str(RAIZ / "src")],
        pip_requirements=["mlflow", "xgboost", "lightgbm", "scikit-learn", "pandas", "numpy"],
    )
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algoritmos", nargs="+", default=list(ALGORITMOS), choices=ALGORITMOS)
    ap.add_argument("--refrescar", action="store_true", help="vuelve a descargar Gold")
    args = ap.parse_args()

    cfg = cargar_config()
    setup_mlflow(cfg)
    d = preparar(cfg, args.refrescar)
    print(f"Datos: {d['huella']['n_filas']} avisos | train {len(d['train'])} / val {len(d['val'])} "
          f"/ test {len(d['test'])} | {len(d['variables'])} variables")
    folds = construir_folds(cfg, d["train"], d["variables"])

    for alg in args.algoritmos:
        with mlflow.start_run(run_name=f"train_{alg}"):
            mlflow.set_tag("etapa", "train")
            params = PARAMS_BASE[alg]
            mlflow.log_metric("cv_mae", cv_mae(cfg, alg, params, folds))
            r = registrar_modelo_final(cfg, alg, params, d)
            print(f"[{alg}] test MAE={r['test_mae']:,.0f} MAPE={r['test_mape']:.1f}% R²={r['test_r2']:.3f} "
                  f"±10%={r['test_pct_dentro_10']:.0f}% | baseline mejor MAE="
                  f"{r['test_mae'] + r['ganancia_vs_mejor_baseline_mae']:,.0f}")


if __name__ == "__main__":
    main()
