"""Modelos (XGBoost y LightGBM), bagging de semillas y el objeto que se registra en MLflow."""

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import ConstructorVariables

ALGORITMOS = ("xgboost", "lightgbm")

# Un hilo por modelo: en Windows, usar todos los núcleos dentro de loops de muchos fits genera
# más overhead de hilos que ganancia (lección de Concepción).
N_JOBS = 1

# Objetivo fijo, fuera del espacio de búsqueda: dejarlo dentro hacía que el ganador cambiara por
# ruido de los folds y no por diferencias reales.
OBJETIVO_XGB = "reg:squarederror"

PARAMS_BASE = {
    "xgboost": {
        "learning_rate": 0.05, "max_depth": 4, "max_leaves": 32, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "alpha": 0.1, "lambda": 1.0, "gamma": 0.0,
    },
    "lightgbm": {
        "learning_rate": 0.05, "max_depth": 4, "num_leaves": 16, "min_child_samples": 10,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_split_gain": 0.0,
    },
}


def espacio_busqueda(algoritmo: str, trial) -> dict:
    """Mismo espacio que Concepción (docs/01_decisiones.md, sección 2)."""
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    if algoritmo == "xgboost":
        p.update(
            max_leaves=trial.suggest_int("max_leaves", 8, 64),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            alpha=trial.suggest_float("alpha", 1e-2, 20.0, log=True),
            **{"lambda": trial.suggest_float("lambda", 1e-2, 20.0, log=True)},
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )
    else:
        p.update(
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 20),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 5.0),
        )
    return p


def entrenar_uno(algoritmo, params, X_tr, y_tr, X_va, y_va, semilla, max_rondas, early_stopping):
    """Un modelo con early stopping sobre (X_va, y_va). Devuelve el modelo ya entrenado."""
    if algoritmo == "xgboost":
        # grow_policy lossguide para que max_leaves tenga efecto (con depthwise lo ignora).
        p = {"objective": OBJETIVO_XGB, "eval_metric": "mae", "tree_method": "hist",
             "grow_policy": "lossguide", "verbosity": 0, "seed": semilla, "nthread": N_JOBS, **params}
        dtr, dva = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_va, label=y_va)
        return xgb.train(p, dtr, num_boost_round=max_rondas, evals=[(dva, "val")],
                         early_stopping_rounds=early_stopping, verbose_eval=False)
    p = {"objective": "regression", "metric": "mae", "verbosity": -1, "seed": semilla,
         "n_jobs": N_JOBS, "subsample_freq": 1, **params}
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=max_rondas, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early_stopping, verbose=False)])


def predecir_uno(algoritmo, modelo, X) -> np.ndarray:
    if algoritmo == "xgboost":
        return modelo.predict(xgb.DMatrix(X), iteration_range=(0, modelo.best_iteration + 1))
    return modelo.predict(X, num_iteration=modelo.best_iteration)


class Ensamble:
    """Bagging: los mismos hiperparámetros con N semillas distintas; la predicción es el promedio."""

    def __init__(self, algoritmo: str, params: dict):
        self.algoritmo, self.params, self.modelos = algoritmo, params, []

    def fit(self, X_tr, y_tr, X_va, y_va, n_modelos, semilla_base, max_rondas, early_stopping):
        self.modelos = [
            entrenar_uno(self.algoritmo, self.params, X_tr, y_tr, X_va, y_va, semilla_base + i,
                         max_rondas, early_stopping)
            for i in range(n_modelos)
        ]
        self.columnas = list(X_tr.columns)
        return self

    def matriz(self, X) -> np.ndarray:
        """(n_modelos, n_filas), sin promediar."""
        X = X[self.columnas]
        return np.stack([np.clip(predecir_uno(self.algoritmo, m, X), 0, None) for m in self.modelos])

    def predecir(self, X) -> np.ndarray:
        return self.matriz(X).mean(axis=0)

    def shap_nativo(self, X) -> dict:
        """Importancia SHAP media absoluta con el TreeSHAP nativo (sin la librería `shap`, que
        tira el proceso en este entorno Windows). Promedia sobre el ensamble."""
        X = X[self.columnas]
        acum = np.zeros(len(self.columnas))
        for m in self.modelos:
            if self.algoritmo == "xgboost":
                c = m.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
            else:
                c = m.predict(X, pred_contrib=True)[:, :-1]
            acum += np.abs(c).mean(axis=0)
        return dict(pd.Series(acum / len(self.modelos), index=self.columnas).sort_values(ascending=False))


class ModeloPrecio(mlflow.pyfunc.PythonModel):
    """Lo que se registra: constructor de variables + ensamble en un solo objeto, para que el
    scoring parta de filas de Gold y no pueda desalinearse del entrenamiento."""

    def __init__(self, constructor: ConstructorVariables, ensamble: Ensamble, variables: list):
        self.constructor, self.ensamble, self.variables = constructor, ensamble, variables

    def predict(self, context, model_input, params=None):
        X = self.constructor.transform(model_input)[self.variables]
        return pd.DataFrame({"prediccion_clp": self.ensamble.predecir(X)})
