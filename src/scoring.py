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
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from psycopg2.extras import execute_values

from . import etiquetas
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


def referencia_del_champion(champ, modelo, gold: pd.DataFrame, cfg: dict) -> dict:
    """Referencia del etiquetado guardada con el champion. Los modelos entrenados antes de que se guardaran los cortes
    de confianza no los traen: se recalculan puntuando con ese mismo champion sus avisos de validación (ids.json),
    que es de donde salen en el entrenamiento."""
    ref = mlflow.artifacts.load_dict(f"runs:/{champ.run_id}/referencia_etiquetas.json")
    if "cortes_cv" not in ref:
        val = gold[gold[ID_COL].isin(set(mlflow.artifacts.load_dict(f"runs:/{champ.run_id}/ids.json")["val"]))]
        cv = modelo.predict(val.reset_index(drop=True))["cv_ensamble"].to_numpy()
        n = cfg["etiquetas"].get("n_grupos_confianza", 3)
        ref["cortes_cv"] = np.quantile(cv, np.linspace(0, 1, n + 1)[1:-1]).tolist()
    return ref


def _num(x):
    """NaN -> None, para que Postgres reciba NULL y no un NaN."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def etiquetar_lote(publicado: np.ndarray, pred: np.ndarray, cv: np.ndarray, ref: dict, cfg_etiquetas: dict) -> dict:
    """Etiqueta, z, confianza y banda de mercado de cada aviso. Los avisos sin precio publicado (NaN) no se etiquetan:
    quedan en None, pero igual conservan su confianza y su banda, que dependen solo de la predicción."""
    n = len(pred)
    hay_precio = ~np.isnan(publicado)
    out = {k: [None] * n for k in ("etiqueta", "z_robusto", "nivel_confianza", "banda_inferior", "banda_superior")}
    umbrales = (cfg_etiquetas["umbral_oportunidad"], cfg_etiquetas["umbral_caro"])
    if "cortes_cv" in ref:
        out["nivel_confianza"] = etiquetas.clasificar_confianza(cv, ref).tolist()
    inf, sup = etiquetas.banda_mercado(pred, ref, *umbrales)
    out["banda_inferior"], out["banda_superior"] = inf.tolist(), sup.tolist()
    if hay_precio.any():
        et = etiquetas.etiquetar(publicado[hay_precio], pred[hay_precio], cv[hay_precio], ref, *umbrales)
        for k, pos in enumerate(np.flatnonzero(hay_precio)):
            out["etiqueta"][pos], out["z_robusto"][pos] = et["etiqueta"].iloc[k], float(et["z_robusto"].iloc[k])
    return out


def _champion(cliente: MlflowClient, nombre: str):
    try:
        return cliente.get_model_version_by_alias(nombre, ALIAS_CHAMPION)
    except MlflowException:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simular", action="store_true", help="solo cuenta los pendientes; no predice ni escribe")
    ap.add_argument("--recalcular", action="store_true",
                    help="vuelve a predecir todos los avisos en alcance con la versión vigente (agrega filas nuevas)")
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
        faltan = candidatos if args.recalcular else pendientes(candidatos, previas)
        print(f"Champion v{version}: {len(gold)} avisos en Gold, {len(candidatos)} en alcance, "
              f"{len(previas)} ya predichos, {len(faltan)} por predecir.")
        if args.simular or faltan.empty:
            return 0

        modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_CHAMPION}")
        huella = champ.tags.get("huella_datos") or cliente.get_run(champ.run_id).data.tags.get("huella_datos")
        faltan = faltan.reset_index(drop=True)
        salida = modelo.predict(faltan)
        pred, cv = salida["prediccion_clp"].to_numpy(), salida["cv_ensamble"].to_numpy()

        # Etiqueta de mercado y confianza: el precio publicado contra lo que el modelo espera (ver etiquetas.py).
        ref = referencia_del_champion(champ, modelo, gold, cfg)
        publicado = faltan[cfg["target"]].to_numpy(float)
        et = etiquetar_lote(publicado, pred, cv, ref, cfg["etiquetas"])
        filas = [
            (i, nombre, version, huella, float(p), _num(pub), float(c), et["nivel_confianza"][k],
             _num(et["z_robusto"][k]), et["etiqueta"][k], _num(et["banda_inferior"][k]), _num(et["banda_superior"][k]))
            for k, (i, p, pub, c) in enumerate(zip(faltan[ID_COL], pred, publicado, cv, strict=True))]
        with con, con.cursor() as cur:  # una sola transacción: o entra todo o nada
            for k in range(0, len(filas), TAMANO_LOTE):
                execute_values(
                    cur,
                    "INSERT INTO gold.predicciones (id_aviso, modelo_nombre, modelo_version, huella_datos, "
                    "costo_total_predicho_clp, costo_total_publicado_clp, cv_ensamble, nivel_confianza, z_robusto, "
                    "etiqueta, banda_inferior_clp, banda_superior_clp) VALUES %s",
                    filas[k:k + TAMANO_LOTE])
        print(f"Guardadas {len(filas)} predicciones de la v{version} en gold.predicciones.")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
