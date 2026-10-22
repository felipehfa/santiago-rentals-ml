"""Vigilante del entrenamiento: falla (rojo en GitHub, que manda correo) si el último entrenamiento tiene más
de `HORAS_MAX` horas. El entrenamiento corre cada día a las 07:30 UTC; si GitHub no lo lanza o falla antes de
registrar nada en MLflow, nadie se entera (le pasó el 22-oct-2026). Lo corre el workflow `vigilante` una vez al día.

Solo lee MLflow (MLFLOW_TRACKING_URI); no toca la base de Gold ni el bucket.

Uso:
    python -m src.vigilar_entrenamiento
"""

import sys
import time

from mlflow import MlflowClient

from .common import cargar_config, setup_mlflow

HORAS_MAX = 30   # un día y un margen para un entrenamiento lento o un cron atrasado
# Nombres de los runs finales de un entrenamiento (los trial_ son pruebas de Optuna).
PREFIJOS = ("tune_", "reentrenar_")


def edad_horas(inicio_ms: int | None, ahora_s: float | None = None) -> float | None:
    if inicio_ms is None:
        return None
    return ((ahora_s if ahora_s is not None else time.time()) - inicio_ms / 1000) / 3600


def ultimo_entrenamiento(cliente: MlflowClient, experimento_id: str) -> tuple[str, int] | None:
    """(nombre, inicio en ms) del run de entrenamiento más reciente, o None si no hay."""
    mejor = None
    for prefijo in PREFIJOS:
        runs = cliente.search_runs([experimento_id], filter_string=f"attributes.run_name LIKE '{prefijo}%'",
                                   order_by=["attributes.start_time DESC"], max_results=1)
        if runs and (mejor is None or runs[0].info.start_time > mejor[1]):
            mejor = (runs[0].info.run_name, runs[0].info.start_time)
    return mejor


def main() -> int:
    cfg = cargar_config()
    try:
        setup_mlflow(cfg)
        cliente = MlflowClient()
        exp = cliente.get_experiment_by_name(cfg["mlflow"]["experimento"])
        ultimo = ultimo_entrenamiento(cliente, exp.experiment_id) if exp else None
    except Exception as e:  # noqa: BLE001 - el texto de un error de conexión puede traer la cadena: solo la clase
        print(f"No se pudo leer MLflow ({type(e).__name__})")
        return 1
    if ultimo is None:
        print("ALERTA: no hay ningún entrenamiento registrado en MLflow")
        return 1
    nombre, inicio = ultimo
    horas = edad_horas(inicio)
    print(f"Último entrenamiento: {nombre}, hace {horas:.1f} h")
    if horas > HORAS_MAX:
        print(f"ALERTA: no hay un entrenamiento nuevo hace {horas:.0f} h (máximo {HORAS_MAX}): "
              "revisar el workflow entrenar")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
