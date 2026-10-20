"""Reentrenamiento diario con los hiperparámetros del último Optuna (sin volver a buscarlos).

Cada día cambian los datos de Gold, así que se vuelve a elegir el conjunto de variables y se reentrena el ensamble
con los mejores parámetros del último `tune` de cada algoritmo. Es mucho más barato que Optuna (que corre una vez
por semana) y deja un run que `register` y `promote` tratan igual que uno de `tune`.

Si todavía no hay ningún `tune` en MLflow, falla: el primer entrenamiento debe ser un `tune`.
"""

import argparse
import sys

import mlflow

from .common import cargar_config, setup_mlflow
from .models import ALGORITMOS
from .seleccion_variables import main as seleccionar
from .train import construir_folds, cv_mae, preparar, registrar_modelo_final


def convertir(valor: str):
    """MLflow guarda los parámetros como texto: se devuelven a int o float cuando corresponde."""
    for tipo in (int, float):
        try:
            return tipo(valor)
        except ValueError:
            continue
    return valor


def params_del_ultimo_tune(exp_id: str, algoritmo: str) -> dict | None:
    runs = mlflow.search_runs(
        experiment_ids=[exp_id], filter_string=f"tags.etapa = 'tune' AND tags.algoritmo = '{algoritmo}'",
        order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        return None
    fila = runs.iloc[0]
    params = {c.removeprefix("params.p_"): convertir(str(fila[c])) for c in runs.columns
              if c.startswith("params.p_") and fila[c] is not None and str(fila[c]) != "nan"}
    return params or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algoritmos", nargs="+", default=list(ALGORITMOS), choices=ALGORITMOS)
    ap.add_argument("--refrescar", action="store_true", help="vuelve a descargar Gold")
    args = ap.parse_args()

    cfg = cargar_config()
    setup_mlflow(cfg)
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experimento"])
    params = {a: params_del_ultimo_tune(exp.experiment_id, a) for a in args.algoritmos}
    faltan = [a for a, p in params.items() if p is None]
    if faltan:
        print(f"No hay un tune previo de {', '.join(faltan)}: correr primero `python -m src.tune`.")
        return 1

    print("Selección de variables con los datos actuales...", flush=True)
    seleccionar(args.refrescar)
    d = preparar(cfg)
    print(f"Datos: {d['huella']['n_filas']} avisos | {len(d['variables'])} variables", flush=True)
    folds = construir_folds(cfg, d["train"], d["variables"])

    for alg in args.algoritmos:
        with mlflow.start_run(run_name=f"reentrenar_{alg}"):
            # Misma etiqueta que tune: register los considera candidatos.
            mlflow.set_tags({"etapa": "tune", "origen": "reentrenar"})
            mejor = cv_mae(cfg, alg, params[alg], folds)
            mlflow.log_metric("cv_mae", mejor)
            r = registrar_modelo_final(cfg, alg, params[alg], d)
            print(f"[{alg}] CV MAE={mejor:,.0f} | test MAE={r['test_mae']:,.0f} "
                  f"MAPE={r['test_mape']:.1f}% R²={r['test_r2']:.3f} ±10%={r['test_pct_dentro_10']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
