"""Publica en `gold.modelo_vigente` las cifras del @champion que muestra la pestaña "Cómo funciona" del mapa.

Las cifras salen del run del champion en MLflow (métricas, tamaños de los conjuntos, baselines, importancia de
variables y referencia del etiquetado). El scoring lo llama en cada corrida y solo escribe cuando la versión del
champion no está publicada todavía: el mapa nunca queda con las cifras de un modelo anterior y no hace falta correr
nada a mano.
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from psycopg2.extras import Json


def cifras_del_champion(cliente, champ) -> dict:
    """Arma el documento (JSON) con las cifras del run del champion."""
    run = cliente.get_run(champ.run_id)
    m, p = run.data.metrics, run.data.params
    with tempfile.TemporaryDirectory() as tmp:
        def artefacto(nombre):
            return json.loads(Path(cliente.download_artifacts(champ.run_id, nombre, tmp)).read_text(encoding="utf-8"))
        shap, ref = artefacto("shap_importancia.json"), artefacto("referencia_etiquetas.json")
        etiquetas, variables = artefacto("etiquetas_test.json"), artefacto("variables.json")["variables"]
    total = sum(shap.values())
    top = sorted(shap.items(), key=lambda kv: -kv[1])[:10]
    return {
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": int(champ.version),
        "algoritmo": p.get("algoritmo"),
        "n_modelos": int(p.get("n_modelos_bagging", 0)),
        "n_variables": len(variables),
        "n_train": int(p["n_train"]), "n_val": int(p["n_val"]), "n_test": int(p["n_test"]),
        "ventana_dias": int(p["ventana_dias"]) if p.get("ventana_dias") else None,
        "metricas": {"mae": m["test_mae"], "mape": m["test_mape"], "mdape": m.get("test_mdape"), "r2": m["test_r2"],
                     "dentro_10": m.get("test_pct_dentro_10"), "mae_quintil_caro": m.get("test_mae_quintil_caro")},
        "baselines": {k: v for k, v in m.items() if k.startswith("baseline_") and k.endswith("_mae")},
        "importancia": [{"variable": k, "pct": round(100 * v / total, 1)} for k, v in top],
        "cortes_costo": ref["cortes_costo"],
        "etiquetas_test": etiquetas,
    }


def ya_publicada(con, nombre: str, version: int) -> bool:
    with con.cursor() as cur:
        cur.execute("SELECT 1 FROM gold.modelo_vigente WHERE modelo_nombre = %s AND modelo_version = %s",
                    (nombre, version))
        return cur.fetchone() is not None


def publicar(con, cliente, champ, nombre: str) -> bool:
    """Escribe las cifras de la versión del champion si aún no están. Devuelve True si escribió.

    Nunca interrumpe el scoring: si falla (por ejemplo, la tabla aún no existe) lo dice y sigue. Imprime solo la clase
    del error, porque el texto de una excepción de conexión puede traer la cadena."""
    version = int(champ.version)
    try:
        if ya_publicada(con, nombre, version):
            return False
        cifras = cifras_del_champion(cliente, champ)
        with con, con.cursor() as cur:
            cur.execute(
                "INSERT INTO gold.modelo_vigente (modelo_nombre, modelo_version, cifras) VALUES (%s, %s, %s) "
                "ON CONFLICT (modelo_nombre, modelo_version) "
                "DO UPDATE SET cifras = EXCLUDED.cifras, publicado_en = now()",
                (nombre, version, Json(cifras)))
        print(f"Cifras de la v{version} publicadas en gold.modelo_vigente.")
        return True
    except Exception as e:  # noqa: BLE001 - el scoring no debe caerse por esto
        con.rollback()
        print(f"No se pudieron publicar las cifras del modelo ({type(e).__name__}); el scoring sigue.")
        return False
