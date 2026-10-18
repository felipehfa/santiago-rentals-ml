"""Paso 3: registra el mejor run como `@challenger`.

El mejor se elige por `cv_mae` (validación cruzada sobre train). El test no interviene: queda para
el quality gate de promote.py.
"""

import argparse
import sys

import mlflow
from mlflow import MlflowClient

from .common import cargar_config, setup_mlflow

ALIAS_CHALLENGER = "challenger"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", help="registrar este run en vez del mejor por cv_mae")
    args = ap.parse_args()
    cfg = cargar_config()
    setup_mlflow(cfg)
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experimento"])
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED' AND metrics.cv_mae > 0", order_by=["metrics.cv_mae ASC"],
    )
    # MLflow no soporta OR entre tags: se descartan acá los trials (sin etiqueta) y la selección.
    runs = runs[runs["tags.etapa"].isin(["train", "tune"])]
    # Solo compiten los runs con los mismos datos que el más reciente: el cv_mae se mide sobre
    # los avisos de train, así que con otro dataset o filtro no es comparable. Las variables sí
    # pueden diferir: probar otra selección sobre los mismos datos es justamente lo que se compara.
    if not runs.empty:
        ultimo = runs.sort_values("start_time", ascending=False).iloc[0]["tags.huella_datos"]
        runs = runs[runs["tags.huella_datos"] == ultimo]
    if runs.empty:
        print("No hay runs de train/tune para registrar.")
        return 1
    if args.run_id:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{args.run_id}'")
        if runs.empty:
            print(f"No existe el run {args.run_id}.")
            return 1
    mejor = runs.iloc[0]
    nombre = cfg["mlflow"]["modelo_registrado"]
    version = mlflow.register_model(f"runs:/{mejor.run_id}/model", nombre)
    cliente = MlflowClient()
    cliente.set_registered_model_alias(nombre, ALIAS_CHALLENGER, version.version)
    cliente.set_model_version_tag(nombre, version.version, "cv_mae", f"{mejor['metrics.cv_mae']:.2f}")
    print(f"Registrado {nombre} v{version.version} ({mejor['tags.algoritmo']}, run {mejor.run_id[:8]}) "
          f"como @{ALIAS_CHALLENGER} | CV MAE={mejor['metrics.cv_mae']:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
