"""Limpia del registro y del bucket los runs y versiones que ya no sirven. Por defecto solo SIMULA: lista qué borraría.

Se conserva:
  - el run de cada versión con alias (@champion, @challenger),
  - el último run de cada entrenamiento final por algoritmo (`tune_<algoritmo>`): `reentrenar` toma de ahí los
    hiperparámetros, y el último `seleccion_variables`,
  - todo lo de los últimos `DIAS_MINIMOS` días (no tocar un entrenamiento en curso ni uno recién comparado).
Se borra lo demás: runs viejos (incluidas las pruebas `trial_` de Optuna) y las versiones del registro sin alias cuyo
run se borra (sin su run, la versión ya no se puede cargar).

Con `--aplicar`: borra las versiones, marca los runs como borrados y corre `mlflow gc`, que elimina sus filas y sus
archivos del bucket. Es irreversible. Necesita MLFLOW_TRACKING_URI y las claves del bucket.

Uso:
    python -m src.limpiar_artefactos             # simula
    python -m src.limpiar_artefactos --aplicar   # borra
"""

import argparse
import os
import subprocess
import sys
import time

from mlflow import MlflowClient

from .common import cargar_config, setup_mlflow

DIAS_MINIMOS = 2
ALIAS = ("champion", "challenger")


def elegir(runs: list[dict], versiones: list[dict], ahora_ms: int,
           dias_minimos: int = DIAS_MINIMOS) -> tuple[set, list]:
    """Función pura. runs: {id, nombre, inicio_ms}; versiones: {version, run_id, alias (lista)}.
    Devuelve (ids de runs a borrar, números de versión a borrar)."""
    conservar = {v["run_id"] for v in versiones if v["alias"]}
    ultimos = {}
    for r in runs:
        nombre = r["nombre"] or ""
        if nombre.startswith("tune_") or nombre == "seleccion_variables":
            if nombre not in ultimos or r["inicio_ms"] > ultimos[nombre]["inicio_ms"]:
                ultimos[nombre] = r
    conservar |= {r["id"] for r in ultimos.values()}
    limite = ahora_ms - dias_minimos * 86_400_000
    borrar = {r["id"] for r in runs if r["id"] not in conservar and r["inicio_ms"] < limite}
    versiones_borrar = [v["version"] for v in versiones if not v["alias"] and v["run_id"] in borrar]
    return borrar, sorted(versiones_borrar)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aplicar", action="store_true", help="borra de verdad (sin esto solo simula)")
    args = ap.parse_args()
    cfg = cargar_config()
    nombre_modelo = cfg["mlflow"]["modelo_registrado"]
    try:
        setup_mlflow(cfg)
        cliente = MlflowClient()
        exp = cliente.get_experiment_by_name(cfg["mlflow"]["experimento"])
        runs = [{"id": r.info.run_id, "nombre": r.info.run_name, "inicio_ms": r.info.start_time}
                for r in cliente.search_runs([exp.experiment_id], max_results=50_000)]
        versiones = [{"version": int(v.version), "run_id": v.run_id, "alias": list(v.aliases or [])}
                     for v in cliente.search_model_versions(f"name = '{nombre_modelo}'")]
    except Exception as e:  # noqa: BLE001 - el texto de un error de conexión puede traer la cadena: solo la clase
        print(f"No se pudo leer MLflow ({type(e).__name__})")
        return 1

    borrar, versiones_borrar = elegir(runs, versiones, int(time.time() * 1000))
    print(f"{len(runs)} runs y {len(versiones)} versiones. Se borrarían {len(borrar)} runs y las versiones "
          f"{versiones_borrar or 'ninguna'}; se conservan {len(runs) - len(borrar)} runs.")
    if not args.aplicar:
        print("Simulación: no se borró nada. Para borrar, correr con --aplicar.")
        return 0

    for v in versiones_borrar:
        cliente.delete_model_version(nombre_modelo, str(v))
    for run_id in borrar:
        cliente.delete_run(run_id)
    if borrar:
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        codigo = subprocess.run(["mlflow", "gc", "--backend-store-uri", uri, "--run-ids", ",".join(sorted(borrar))],
                                # la salida puede traer la URI: no se muestra
                                check=False, capture_output=True).returncode
        if codigo != 0:
            print(f"mlflow gc terminó con código {codigo}: los runs quedaron marcados como borrados, sin purgar")
            return 1
    print(f"Borrados {len(borrar)} runs y {len(versiones_borrar)} versiones, con sus archivos del bucket.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
