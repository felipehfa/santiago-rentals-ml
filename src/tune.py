"""Paso 2: Optuna por algoritmo. Un run padre por algoritmo y un run hijo por trial.

Se optimiza el MAE de una CV sobre train (el constructor de variables se reajusta en cada fold).
Con los mejores parámetros se entrena el ensamble final y se registra el modelo en el mismo run.
La selección entre modelos se hace después por `cv_mae`, nunca por el test.
"""

import argparse

import mlflow
import optuna

from .common import cargar_config, setup_mlflow
from .models import ALGORITMOS, espacio_busqueda
from .train import construir_folds, cv_mae, preparar, registrar_modelo_final

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimizar(cfg: dict, algoritmo: str, folds: list, n_trials: int) -> tuple[dict, float]:
    def objetivo(trial):
        params = espacio_busqueda(algoritmo, trial)
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            valor = cv_mae(cfg, algoritmo, params, folds)
            mlflow.log_params({f"p_{k}": v for k, v in params.items()})
            mlflow.log_metric("cv_mae", valor)
        return valor

    estudio = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=cfg["semilla"]))
    estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return estudio.best_params, estudio.best_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algoritmos", nargs="+", default=list(ALGORITMOS), choices=ALGORITMOS)
    ap.add_argument("--n-trials", type=int, help="reemplaza optuna.n_trials de config.yaml")
    ap.add_argument("--refrescar", action="store_true", help="vuelve a descargar Gold")
    args = ap.parse_args()

    cfg = cargar_config()
    n_trials = args.n_trials or cfg["optuna"]["n_trials"]
    setup_mlflow(cfg)
    d = preparar(cfg, args.refrescar)
    print(f"Datos: {d['huella']['n_filas']} avisos | {len(d['variables'])} variables | "
          f"{n_trials} trials por algoritmo")
    folds = construir_folds(cfg, d["train"], d["variables"])

    for alg in args.algoritmos:
        with mlflow.start_run(run_name=f"tune_{alg}"):
            mlflow.set_tag("etapa", "tune")
            params, mejor = optimizar(cfg, alg, folds, n_trials)
            mlflow.log_metrics({"cv_mae": mejor})
            mlflow.log_param("n_trials", n_trials)
            r = registrar_modelo_final(cfg, alg, params, d)
            print(f"[{alg}] CV MAE={mejor:,.0f} | test MAE={r['test_mae']:,.0f} "
                  f"MAPE={r['test_mape']:.1f}% R²={r['test_r2']:.3f} ±10%={r['test_pct_dentro_10']:.0f}%")


if __name__ == "__main__":
    main()
