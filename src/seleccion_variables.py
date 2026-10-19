"""Selección de variables. Portado de Concepción (02_seleccion_variables.py).

1. Descarta variables constantes.
2. Estabilidad: N modelos XGBoost con KFold aleatorio y semillas distintas; la importancia es el
   SHAP medio absoluto. Pasan las variables con importancia > 0 en al menos `umbral_presencia`
   de los modelos.
3. Curva MAE de validación vs k (variables ordenadas por importancia) y regla de 1 error
   estándar: el k más chico cuyo MAE no supere el mínimo más un SE.
4. Poda por correlación (red de seguridad) quedándose con la de mayor importancia.

Solo usa train y val (el test no se toca). Las variables vienen del constructor ajustado con
train; dentro de la CV de este paso las filas de un fold sí aportan a las variables de las
demás (leave-one-out sobre todo train), una holgura aceptable para elegir variables y que
Optuna no tiene porque reajusta el constructor en cada fold.

Escribe `features_seleccionadas.json`, que usan train y tune, y lo registra en MLflow. `tune` lo corre
antes de la búsqueda de hiperparámetros, así que cada entrenamiento reselecciona con sus datos.
"""

import json
import sys
import time

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from .common import RAIZ, cargar_config, registrar_reproducibilidad, setup_mlflow
from .train import preparar

# Parámetros fijos y simples: aquí interesa la importancia relativa, no el mejor modelo.
PARAMS = {"objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "hist", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0, "nthread": 1}
RONDAS = 200


def _entrenar(X, y, semilla, rondas=RONDAS):
    return xgb.train({**PARAMS, "seed": semilla}, xgb.DMatrix(X, label=y), num_boost_round=rondas)


def _shap_medio(modelo, X) -> np.ndarray:
    return np.abs(modelo.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(axis=0)


def estabilidad(X: pd.DataFrame, y: pd.Series, s: dict, semilla: int) -> pd.DataFrame:
    imp = {c: [] for c in X.columns}
    maes = []
    t0 = time.time()
    for i in range(s["n_modelos"]):
        kf = KFold(n_splits=s["cv_folds"], shuffle=True, random_state=semilla + i)
        por_fold = []
        for tr, va in kf.split(X):
            m = _entrenar(X.iloc[tr], y.iloc[tr], semilla + i)
            maes.append(float(np.mean(np.abs(y.iloc[va] - np.clip(m.predict(xgb.DMatrix(X.iloc[va])), 0, None)))))
            por_fold.append(_shap_medio(m, X.iloc[va]))
        for c, v in zip(X.columns, np.mean(por_fold, axis=0), strict=True):
            imp[c].append(float(v))
        if (i + 1) % 10 == 0:
            print(f"  estabilidad: {i + 1}/{s['n_modelos']} modelos ({time.time() - t0:.0f}s)", flush=True)

    filas = []
    for c, v in imp.items():
        v = np.asarray(v)
        media, std, presencia = v.mean(), v.std(), float((v > 0).mean())
        cv = std / media if media > 0 else np.nan
        score = (1 / (1 + cv)) * presencia if media > 0 else 0.0
        filas.append({"feature": c, "mean_imp": media, "cv_imp": cv, "presence_pct": presencia,
                      "stability_score": score})
    df = pd.DataFrame(filas).sort_values("stability_score", ascending=False).reset_index(drop=True)
    df["pasa_presencia"] = df["presence_pct"] >= s["umbral_presencia"]
    print(f"  MAE medio de los folds internos: {np.mean(maes):,.0f}")
    return df


def curva_k(est: pd.DataFrame, Xtr, ytr, Xva, yva, s: dict, semilla: int):
    # Se ordena por importancia (SHAP medio): el stability_score de Concepción premia la estabilidad y
    # dejaba fuera variables importantes. La estabilidad solo sirve de filtro (presencia).
    candidatas = est.loc[est["pasa_presencia"]].sort_values("mean_imp", ascending=False)["feature"].tolist()
    semillas = [semilla + i for i in range(s["n_semillas_curva"])]
    filas = []
    for k in range(min(s["k_min"], len(candidatas)), len(candidatas) + 1):
        cols = candidatas[:k]
        maes = []
        for sem in semillas:
            m = _entrenar(Xtr[cols], ytr, sem, rondas=300)
            maes.append(float(np.mean(np.abs(yva - np.clip(m.predict(xgb.DMatrix(Xva[cols])), 0, None)))))
        filas.append({"k": k, "mae_val": float(np.mean(maes)), "mae_std_semillas": float(np.std(maes))})
    curva = pd.DataFrame(filas)
    i_min = curva["mae_val"].idxmin()
    umbral = curva.loc[i_min, "mae_val"] + curva.loc[i_min, "mae_std_semillas"] / np.sqrt(len(semillas))
    k_opt = int(curva.loc[curva["mae_val"] <= umbral, "k"].iloc[0])
    print(f"  k con MAE mínimo: {int(curva.loc[i_min, 'k'])} | k óptimo (1-SE): {k_opt} "
          f"(MAE {curva.loc[curva['k'] == k_opt, 'mae_val'].iloc[0]:,.0f})")
    return candidatas[:k_opt], curva


def podar_correlacion(X: pd.DataFrame, elegidas: list, est: pd.DataFrame, umbral: float):
    score = est.set_index("feature")["mean_imp"].to_dict()
    orden = sorted(elegidas, key=lambda c: score.get(c, 0), reverse=True)
    corr = X[orden].fillna(0).corr().abs()
    mantiene, quitadas = [], []
    for c in orden:
        if c in quitadas:
            continue
        mantiene.append(c)
        quitadas += [o for o in orden if o != c and o not in quitadas and corr.loc[c, o] > umbral]
    return mantiene, quitadas


def main(refrescar: bool = False):
    cfg = cargar_config()
    s = cfg["seleccion_variables"]
    setup_mlflow(cfg)
    ruta = RAIZ / cfg["variables"]["seleccion"]
    ruta_previa = ruta.with_suffix(".previa.json")
    if ruta.exists():  # la selección debe partir de todas las variables, no de la anterior
        ruta.rename(ruta_previa)
    try:
        d = preparar(cfg, refrescar)
    finally:
        if ruta_previa.exists():
            ruta_previa.rename(ruta)
    Xtr, ytr, yva = d["X_completo_train"], d["y_train"], d["y_val"]
    Xva = d["constructor"].transform(d["val"])

    constantes = [c for c in Xtr.columns if Xtr[c].var(skipna=True) <= 1e-6 or Xtr[c].isna().all()]
    Xtr, Xva = Xtr.drop(columns=constantes), Xva.drop(columns=constantes)
    print(f"Variables iniciales: {Xtr.shape[1] + len(constantes)} | constantes: {len(constantes)}")

    with mlflow.start_run(run_name="seleccion_variables"):
        mlflow.set_tag("etapa", "seleccion")
        registrar_reproducibilidad()
        est = estabilidad(Xtr, ytr, s, cfg["semilla"])
        print(f"  con presencia >= {s['umbral_presencia']}: {int(est['pasa_presencia'].sum())}/{len(est)}")
        elegidas, curva = curva_k(est, Xtr, ytr, Xva, yva, s, cfg["semilla"])
        finales, quitadas = podar_correlacion(Xtr, elegidas, est, s["umbral_correlacion"])

        ruta.write_text(json.dumps({
            "features": finales, "n_filas": d["huella"]["n_filas"], "huella_datos": d["huella"]["huella"],
            "constantes": constantes, "quitadas_por_correlacion": quitadas,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(ruta))
        mlflow.log_dict(est.to_dict(orient="records"), "estabilidad.json")
        mlflow.log_dict(curva.to_dict(orient="records"), "curva_k.json")
        mlflow.log_metrics({"n_variables_finales": len(finales), "k_optimo": len(elegidas)})

    print(f"Variables finales: {len(finales)} (quitadas por correlación: {quitadas})")
    print("\n".join(f"  {c}" for c in finales))


if __name__ == "__main__":
    main("--refrescar" in sys.argv[1:])
