"""Scoring por lotes: predice con el @champion los avisos de Gold que lo necesiten y los guarda en
`gold.predicciones` (solo INSERT; el historial no se pisa).

Un aviso necesita predicción cuando, para la versión actual del champion:
  - nunca fue predicho (aviso nuevo en Gold, o cambió el modelo y hay que repetir todo), o
  - se volvió a visitar después de su última predicción (sus datos pudieron cambiar).
Por eso correrlo dos veces seguidas no escribe nada la segunda vez.

Alcance: los avisos aptos para entrenar y los excluidos solo por motivos de precio (`scoring.motivos_predecibles`),
porque el modelo igual puede estimar su precio esperado, que sirve justamente para detectar los atípicos.
Quedan fuera las comunas de `datos.comunas_excluidas`.

Usa el rol ml_scoring (BD_STRING_SCORING): lee `gold.aviso_entrenamiento` y `gold.predicciones`, e inserta en
esta última. Códigos de salida: 0 normal (incluso sin pendientes), 1 sin champion.
"""

import argparse
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from psycopg2.extras import execute_values

from .common import ID_COL, _conexion_bd, cargar_config, setup_mlflow

ALIAS_CHAMPION = "champion"
TAMANO_LOTE = 2000


def en_alcance(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Avisos que el modelo puede y debe predecir."""
    ok = set(cfg.get("scoring", {}).get("motivos_predecibles") or [])

    def predecible(fila) -> bool:
        if fila["apto_entrenamiento"]:
            return True
        motivos = set(str(fila["motivo_exclusion"] or "").split(","))
        return bool(motivos) and motivos <= ok

    df = df[df.apply(predecible, axis=1)]
    excluidas = cfg["datos"].get("comunas_excluidas") or []
    if excluidas:
        df = df[~df["comuna"].isin(excluidas)]
    return df


def pendientes(df: pd.DataFrame, previas: pd.DataFrame) -> pd.DataFrame:
    """Avisos sin predicción de esta versión, o visitados después de la última que tuvieron.

    `previas`: columnas id_aviso y predicho_en (la más reciente por aviso, solo de la versión vigente).
    """
    ultima = previas.set_index(ID_COL)["predicho_en"]
    visto = pd.to_datetime(df["ultima_vez_visto"], utc=True)
    prev = pd.to_datetime(df[ID_COL].map(ultima), utc=True)
    return df[prev.isna() | (visto > prev)]


def _champion(cliente: MlflowClient, nombre: str):
    try:
        return cliente.get_model_version_by_alias(nombre, ALIAS_CHAMPION)
    except MlflowException:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simular", action="store_true", help="solo cuenta los pendientes; no predice ni escribe")
    args = ap.parse_args()

    cfg = cargar_config()
    setup_mlflow(cfg)
    cliente = MlflowClient()
    nombre = cfg["mlflow"]["modelo_registrado"]
    champ = _champion(cliente, nombre)
    if champ is None:
        print("Sin @champion: no hay con qué predecir.")
        return 1
    version = int(champ.version)

    con = _conexion_bd("BD_STRING_SCORING")
    try:
        gold = pd.read_sql(f"SELECT * FROM {cfg['datos']['tabla']}", con)
        previas = pd.read_sql(
            "SELECT id_aviso, max(predicho_en) AS predicho_en FROM gold.predicciones "
            "WHERE modelo_nombre = %(n)s AND modelo_version = %(v)s GROUP BY id_aviso",
            con, params={"n": nombre, "v": version})
        candidatos = en_alcance(gold, cfg)
        faltan = pendientes(candidatos, previas)
        print(f"Champion v{version}: {len(gold)} avisos en Gold, {len(candidatos)} en alcance, "
              f"{len(previas)} ya predichos, {len(faltan)} por predecir.")
        if args.simular or faltan.empty:
            return 0

        modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_CHAMPION}")
        huella = champ.tags.get("huella_datos") or cliente.get_run(champ.run_id).data.tags.get("huella_datos")
        pred = modelo.predict(faltan.reset_index(drop=True))["prediccion_clp"].to_numpy()
        filas = [(i, nombre, version, huella, float(p)) for i, p in zip(faltan[ID_COL], pred, strict=True)]
        with con, con.cursor() as cur:  # una sola transacción: o entra todo o nada
            for k in range(0, len(filas), TAMANO_LOTE):
                execute_values(
                    cur,
                    "INSERT INTO gold.predicciones (id_aviso, modelo_nombre, modelo_version, huella_datos, "
                    "costo_total_predicho_clp) VALUES %s",
                    filas[k:k + TAMANO_LOTE])
        print(f"Guardadas {len(filas)} predicciones de la v{version} en gold.predicciones.")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
