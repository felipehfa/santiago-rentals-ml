"""Reconstruye `ids.json` de un run entrenado antes de que se guardaran los ids de los splits.

Solo es válido si los datos de hoy son exactamente los del run (misma huella): el split es
determinista dada la semilla, pero cambia si cambian las filas. Si no coinciden, se rechaza.
"""

import argparse
import sys

import mlflow
from mlflow import MlflowClient

from .common import ID_COL, cargar_config, dividir, huella_datos, setup_mlflow
from .comparacion import datos_actuales


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    cfg = cargar_config()
    setup_mlflow(cfg)

    run = MlflowClient().get_run(args.run_id)
    df = datos_actuales(cfg)
    huella = huella_datos(df, cfg["target"])["huella"]
    if run.data.tags.get("huella_datos") != huella:
        print(f"Los datos actuales ({huella}) no son los del run ({run.data.tags.get('huella_datos')}); "
              "no se puede reconstruir el split.")
        return 1
    train, val, test = dividir(df, cfg, int(run.data.params["semilla"]))
    ids = {k: d[ID_COL].tolist() for k, d in (("train", train), ("val", val), ("test", test))}
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_dict(ids, "ids.json")
        mlflow.set_tag("ids_reconstruidos", "true")
    print(f"ids.json reconstruido para {args.run_id[:8]}: " + ", ".join(f"{k}={len(v)}" for k, v in ids.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
