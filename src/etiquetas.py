"""Etiquetado oportunidad / caro / precio de mercado.

Compara el costo total publicado de un aviso con lo que el modelo espera, normalizando el error
por decil de costo total con mediana y MAD (robustos a outliers). Un z muy negativo significa que
el aviso cuesta menos de lo esperado ("oportunidad"); muy positivo, más ("caro").

A diferencia de Concepción, la referencia (mediana y MAD por decil) se calcula una vez con
residuos que el modelo no usó para ajustar sus parámetros y se guarda junto al modelo. Así un
aviso nuevo se etiqueta contra una referencia fija y no contra el lote en que viene.
"""

import numpy as np
import pandas as pd

MAD_ESCALA = 1.4826  # hace el MAD comparable a una desviación estándar bajo normalidad


def ajustar_referencia(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int) -> dict:
    """Referencia fija del etiquetado: cortes de costo por decil y, en cada uno, mediana y MAD del error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    error = y_true - y_pred
    cortes = np.quantile(y_true, np.linspace(0, 1, n_deciles + 1)[1:-1])
    decil = np.searchsorted(cortes, y_true, side="right")
    ref = {"cortes_costo": cortes.tolist(), "deciles": {}}
    for d in np.unique(decil):
        e = error[decil == d]
        mediana = float(np.median(e))
        ref["deciles"][int(d)] = {
            "n": int(len(e)), "mediana": mediana, "mad": float(np.median(np.abs(e - mediana))),
        }
    return ref


def banda_mercado(y_pred, ref: dict, umbral_oportunidad=1.0, umbral_caro=1.0) -> tuple:
    """Rango de costo total que sería "precio de mercado" para lo que el modelo espera: [predicho + mediana ± MAD]."""
    y_pred = np.asarray(y_pred, float)
    decil = np.searchsorted(np.asarray(ref["cortes_costo"]), y_pred, side="right")
    deciles = {int(k): v for k, v in ref["deciles"].items()}
    med = np.array([deciles[d]["mediana"] for d in decil])
    mad = np.array([deciles[d]["mad"] for d in decil]) * MAD_ESCALA
    return y_pred + med - umbral_oportunidad * mad, y_pred + med + umbral_caro * mad


def etiquetar(y_true, y_pred, ref: dict, umbral_oportunidad=1.0, umbral_caro=1.0) -> pd.DataFrame:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    decil = np.searchsorted(np.asarray(ref["cortes_costo"]), y_true, side="right")
    # Las claves vuelven como texto tras un viaje por JSON.
    deciles = {int(k): v for k, v in ref["deciles"].items()}
    med = np.array([deciles[d]["mediana"] for d in decil])
    mad = np.array([deciles[d]["mad"] for d in decil])
    mad = np.where(mad * MAD_ESCALA == 0, 1e-6, mad * MAD_ESCALA)
    z = ((y_true - y_pred) - med) / mad
    etiqueta = np.where(z < -umbral_oportunidad, "oportunidad",
                        np.where(z > umbral_caro, "caro", "precio_de_mercado"))
    out = pd.DataFrame({"z_robusto": z, "etiqueta": etiqueta})
    out["banda_inferior"], out["banda_superior"] = banda_mercado(y_pred, ref, umbral_oportunidad, umbral_caro)
    return out
