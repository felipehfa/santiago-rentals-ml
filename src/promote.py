"""Paso 4: quality gate. Si el `@challenger` lo pasa, queda como `@champion`.

Códigos de salida: 0 promueve, 1 no hay challenger, 2 el gate lo rechaza.

Los datos crecen, así que champion y challenger no comparten test. Se comparan sobre las mismas
filas: el test del challenger sin los avisos que el champion usó para entrenar (ver comparacion.py).

Regla de promoción (umbrales en config.yaml):
  1. Se promueve si el challenger mejora el MAE del champion en esas filas (más de `mejora_min`), o
  2. si el champion se degradó (su MAE ahí supera su MAE de test original en más de
     `degradacion_umbral`) y el challenger no es peor que el champion por más de `margen_vs_champion`.
  Si no hay champion, solo rigen los criterios absolutos.
  Criterios absolutos (`null` = no se aplica): MAPE máximo en test y estabilidad entre particiones.
"""

import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from . import comparacion
from .common import cargar_config, setup_mlflow
from .register import ALIAS_CHALLENGER

ALIAS_CHAMPION = "champion"


def evaluar_gate(propias: dict, comunes: dict | None, degradacion: float | None, gate: dict) -> list:
    """Devuelve los motivos de rechazo (lista vacía = pasa). Función pura, para testearla.

    propias:     métricas del challenger sobre su propio test (test_mape, estabilidad_cv_mae).
    comunes:     {"n", "mae_challenger", "mae_champion"} sobre las filas comunes; None sin champion.
    degradacion: MAE del champion en las filas comunes / su MAE de test original - 1.
    """
    motivos = []
    if gate.get("mape_max") is not None and propias["test_mape"] > gate["mape_max"]:
        motivos.append(f"MAPE {propias['test_mape']:.2f}% > {gate['mape_max']}%")
    cv = propias.get("estabilidad_cv_mae")
    if gate.get("estabilidad_cv_max") is not None and cv is not None and cv > gate["estabilidad_cv_max"]:
        motivos.append(f"estabilidad CV {cv:.3f} > {gate['estabilidad_cv_max']}")

    if comunes is not None:
        ch, champ = comunes["mae_challenger"], comunes["mae_champion"]
        mejora = 1 - ch / champ
        degradado = degradacion is not None and degradacion > gate["degradacion_umbral"]
        no_peor = ch <= champ * (1 + gate["margen_vs_champion"])
        if not (mejora > gate["mejora_min"] or (degradado and no_peor)):
            motivos.append(
                f"no mejora al champion (MAE {ch:,.0f} vs {champ:,.0f}, {mejora:+.1%}) y el champion "
                + ("no está degradado" if not degradado else "está degradado pero el challenger es peor")
            )
    return motivos


def _version(cliente: MlflowClient, nombre: str, alias: str):
    try:
        v = cliente.get_model_version_by_alias(nombre, alias)
    except MlflowException:
        return None, None
    return v, cliente.get_run(v.run_id)


def main() -> int:
    cfg = cargar_config()
    setup_mlflow(cfg)
    cliente = MlflowClient()
    nombre, gate, target = cfg["mlflow"]["modelo_registrado"], cfg["gate"], cfg["target"]

    ch, run_ch = _version(cliente, nombre, ALIAS_CHALLENGER)
    if ch is None:
        print("Sin @challenger: nada que promover.")
        return 1
    champ, run_champ = _version(cliente, nombre, ALIAS_CHAMPION)

    comunes, degradacion = None, None
    if champ is not None:
        ids_ch, ids_champ = comparacion.ids_de_run(ch.run_id), comparacion.ids_de_run(champ.run_id)
        if ids_ch is None or ids_champ is None:
            print("Falta ids.json en el challenger o el champion; reconstruirlo con "
                  "`python -m src.reconstruir_ids --run-id <id>` o reentrenar.")
            return 2
        filas = comparacion.filas_comunes(ids_ch, ids_champ)
        df = comparacion.datos_actuales(cfg)
        df = df[df["id_aviso"].isin(filas)]
        if len(df) < gate["n_minimo_comparacion"]:
            print(f"RECHAZADO v{ch.version}: solo {len(df)} avisos comparables "
                  f"(mínimo {gate['n_minimo_comparacion']}).")
            return 2
        m_ch, p_ch = comparacion.evaluar(
            mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_CHALLENGER}"), df, target)
        m_champ, p_champ = comparacion.evaluar(
            mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_CHAMPION}"), df, target)
        comunes = {"n": len(df), "mae_challenger": m_ch["mae"], "mae_champion": m_champ["mae"]}
        mae_original = run_champ.data.metrics["test_mae"]
        degradacion = m_champ["mae"] / mae_original - 1
        dif, lo, hi = comparacion.diferencia_ic(df[target].to_numpy(), p_champ, p_ch)
        info = (f"Sobre {len(df)} avisos comunes: MAE challenger {m_ch['mae']:,.0f} vs champion "
                f"{m_champ['mae']:,.0f} (dif. {dif:+,.0f}, IC95 [{lo:+,.0f}, {hi:+,.0f}]); MAPE "
                f"{m_ch['mape']:.1f}% vs {m_champ['mape']:.1f}%. Degradación del champion: "
                f"{degradacion:+.1%} (MAE original {mae_original:,.0f}).")
        print(info)
        cliente.set_model_version_tag(nombre, ch.version, "comparacion", info[:250])

    motivos = evaluar_gate(run_ch.data.metrics, comunes, degradacion, gate)
    if motivos:
        print(f"RECHAZADO v{ch.version}: " + "; ".join(motivos))
        return 2

    cliente.set_registered_model_alias(nombre, ALIAS_CHAMPION, ch.version)
    cliente.delete_registered_model_alias(nombre, ALIAS_CHALLENGER)
    antes = f" (antes v{champ.version})" if champ else " (primer champion)"
    print(f"PROMOVIDO v{ch.version} a @{ALIAS_CHAMPION}{antes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
