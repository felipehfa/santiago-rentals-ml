"""Construcción de variables sin fuga de datos.

Gold ya entrega los avisos limpios y tipados. Lo que queda acá son las variables que dependen
del conjunto de entrenamiento (precio por m2 del sector, nivel del barrio y de la comuna,
imputación de superficie). Todas se ajustan SOLO con train.

Para una fila que pertenece a train, el aviso mismo se excluye del cálculo (leave-one-out).
En Concepción el comparador del sector incluía el propio costo de la fila, lo que filtraba el
target hacia la variable.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .common import ID_COL

RADIO_TIERRA_M = 6_371_000

# Variables numéricas que pasan tal cual desde Gold. NaN se deja como NaN: XGBoost y LightGBM lo
# manejan, y rellenar con 0 una distancia ausente la haría pasar por "justo al lado".
NUMERICAS = [
    "dormitorios", "banos", "estacionamientos", "bodegas", "superficie_total_m2",
    "antiguedad_anos", "piso_unidad", "deptos_por_piso", "latitud", "longitud",
    "cantidad_paraderos", "distancia_min_m_paraderos",
    "cantidad_estaciones_metro", "distancia_min_m_estaciones_metro",
    "cantidad_jardines_infantiles", "distancia_min_m_jardines_infantiles",
    "cantidad_colegios", "distancia_min_m_colegios",
    "cantidad_universidades", "distancia_min_m_universidades",
    "cantidad_plazas", "distancia_min_m_plazas",
    "cantidad_supermercados", "distancia_min_m_supermercados",
    "cantidad_farmacias", "distancia_min_m_farmacias",
    "cantidad_centros_comerciales", "distancia_min_m_centros_comerciales",
    "cantidad_hospitales", "distancia_min_m_hospitales",
    "cantidad_clinicas", "distancia_min_m_clinicas",
    "rank_nac", "c_ig_com", "c_ig_reg", "c_ig_nac", "pob_rsh_uv", "hog_uv", "p_urbano",
]

BOOLEANAS = [
    "tiene_bodega", "amoblado", "admite_mascotas", "condominio_cerrado", "estacionamiento_visitas",
    "piscina", "quincho", "conserjeria", "ascensor", "gastos_comunes_faltante",
]

# Nunca entran como variables: son parte del target (o se derivan de él) o no existen al
# momento de publicar el aviso.
EXCLUIDAS = [
    "gastos_comunes", "precio_clp", "precio_valor", "precio_moneda", "costo_total_clp",
    "costo_total_uf", "tiene_rebaja", "fecha_precio", "dias_en_mercado", "n_cambios_estado",
    "n_cambios_precio", "primera_vez_visto", "ultima_vez_visto", "fecha_publicacion_aprox",
    "fecha_publicacion_precision", "estado_publicacion", "construido_en", "uv_rsh",
]


def _a_float(serie: pd.Series) -> pd.Series:
    def conv(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return np.nan
        if isinstance(v, str):
            return {"true": 1.0, "false": 0.0}.get(v.strip().lower(), np.nan)
        return float(v)

    return serie.map(conv).astype(float)


class ConstructorVariables:
    """Se ajusta con train (`fit`) y transforma cualquier conjunto (`transform`)."""

    def __init__(self, radio_sector_m=300, multiplicador_iqr=3, k_suavizado=20, n_niveles_barrio=5,
                 target="costo_total_clp"):
        self.radio_sector_m = radio_sector_m
        self.multiplicador_iqr = multiplicador_iqr
        self.k_suavizado = k_suavizado
        self.n_niveles_barrio = n_niveles_barrio
        self.target = target

    # ── superficie ────────────────────────────────────────────────────────────
    def _superficie(self, df: pd.DataFrame) -> pd.Series:
        """Útil si existe; si no, total por la razón útil/total de train; si no, mediana por
        número de dormitorios de train."""
        util = df["superficie_util_m2"].astype(float)
        total = df["superficie_total_m2"].astype(float)
        sup = util.fillna(total * self.razon_util_total_)
        por_dorm = df["dormitorios"].map(self.sup_por_dormitorios_)
        return sup.fillna(por_dorm).fillna(self.sup_mediana_)

    # ── promedios suavizados leave-one-out ────────────────────────────────────
    def _stats_grupo(self, clave: pd.Series, valor: pd.Series) -> dict:
        v = valor.notna()
        g = pd.DataFrame({"k": clave[v], "v": valor[v]}).groupby("k")["v"].agg(["sum", "count"])
        return {k: (r["sum"], r["count"]) for k, r in g.iterrows()}

    def _suavizado(self, clave: pd.Series, stats: dict, es_train: pd.Series, propio: pd.Series) -> pd.Series:
        """(suma + k * media_global) / (n + k), restando la propia fila cuando es de train."""
        k = self.k_suavizado
        out = np.empty(len(clave))
        for i, (c, e, p) in enumerate(zip(clave, es_train, propio, strict=False)):
            suma, n = stats.get(c, (0.0, 0))
            suma_g, n_g = self.suma_global_, self.n_global_
            if e and not np.isnan(p):
                suma, n = suma - p, n - 1
                suma_g, n_g = suma_g - p, n_g - 1  # la media global también incluye la fila
            mu = suma_g / n_g if n_g > 0 else 0.0
            out[i] = (suma + k * mu) / (n + k)
        return pd.Series(out, index=clave.index)

    def _niveles(self, suavizado: pd.Series) -> pd.Series:
        return pd.Series(np.searchsorted(self.cortes_nivel_, suavizado.values, side="left"), index=suavizado.index)

    # ── ajuste ────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ConstructorVariables":
        df = df.reset_index(drop=True)
        u, t = df["superficie_util_m2"].astype(float), df["superficie_total_m2"].astype(float)
        ambos = u.notna() & t.notna() & (t > 0)
        self.razon_util_total_ = float((u[ambos] / t[ambos]).median()) if ambos.any() else 0.85
        self.sup_por_dormitorios_ = u.groupby(df["dormitorios"]).median().dropna().to_dict()
        self.sup_mediana_ = float(u.median())

        sup = self._superficie(df)
        pm2 = df[self.target].astype(float) / sup.replace(0, np.nan)
        self.suma_global_ = float(pm2.sum())
        self.n_global_ = int(pm2.notna().sum())

        barrio = df["barrio"].fillna("sin_barrio")
        comuna = df["comuna"].fillna("sin_comuna")
        self.stats_barrio_ = self._stats_grupo(barrio, pm2)
        self.stats_comuna_ = self._stats_grupo(comuna, pm2)

        # Cortes de nivel con cuantiles ponderados por cantidad de avisos (no de barrios), como en
        # Concepción: "muy alto" son ~20 % de los avisos, no ~20 % de los barrios.
        tabla = pd.DataFrame(
            [(b, s / n, n) for b, (s, n) in self.stats_barrio_.items() if n > 0], columns=["b", "media", "n"]
        )
        media_global = self.suma_global_ / self.n_global_
        tabla["ajustado"] = (tabla["n"] * tabla["media"] + self.k_suavizado * media_global) / (
            tabla["n"] + self.k_suavizado
        )
        tabla = tabla.sort_values("ajustado")
        acumulado = tabla["n"].cumsum() / tabla["n"].sum()
        pcts = np.linspace(0, 1, self.n_niveles_barrio + 1)[1:-1]
        self.cortes_nivel_ = np.interp(pcts, acumulado.values, tabla["ajustado"].values)

        # Comparador del sector: solo precios por m2 razonables (filtro IQR) y con coordenadas.
        q1, q3 = pm2.quantile(0.25), pm2.quantile(0.75)
        iqr = q3 - q1
        razonable = pm2.between(q1 - self.multiplicador_iqr * iqr, q3 + self.multiplicador_iqr * iqr)
        valido = razonable & df[["latitud", "longitud"]].notna().all(axis=1)
        self.ids_sector_ = df.loc[valido, ID_COL].to_numpy()
        self.pm2_sector_ = pm2[valido].to_numpy()
        self.arbol_ = BallTree(np.radians(df.loc[valido, ["latitud", "longitud"]].astype(float).values),
                               metric="haversine")

        self.ids_train_ = set(df[ID_COL])
        self.pm2_propio_ = dict(zip(df[ID_COL], pm2, strict=False))
        return self

    # ── transformación ────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        X = pd.DataFrame(index=df.index)
        for c in NUMERICAS:
            X[c] = df[c].astype(float)
        for c in BOOLEANAS:
            X[c] = _a_float(df[c])
        X["superficie_util_m2"] = self._superficie(df)

        es_train = df[ID_COL].isin(self.ids_train_)
        propio = df[ID_COL].map(self.pm2_propio_).astype(float).fillna(np.nan)
        barrio = df["barrio"].fillna("sin_barrio")
        comuna = df["comuna"].fillna("sin_comuna")

        suav_barrio = self._suavizado(barrio, self.stats_barrio_, es_train, propio)
        suav_comuna = self._suavizado(comuna, self.stats_comuna_, es_train, propio)
        X["nivel_barrio"] = self._niveles(suav_barrio).astype(float)
        X["barrio_precio_m2"] = suav_barrio
        X["comuna_precio_m2"] = suav_comuna

        # Precio por m2 mediano de los vecinos de train a menos de `radio_sector_m`, sin contarse
        # a sí mismo. Sin vecinos cae al promedio suavizado de la comuna.
        sector = suav_comuna.copy()
        n_vecinos = np.zeros(len(df))
        tiene = df[["latitud", "longitud"]].notna().all(axis=1).to_numpy()
        if tiene.any():
            coords = np.radians(df.loc[tiene, ["latitud", "longitud"]].astype(float).values)
            vecinos = self.arbol_.query_radius(coords, r=self.radio_sector_m / RADIO_TIERRA_M)
            ids = df.loc[tiene, ID_COL].to_numpy()
            pos = np.flatnonzero(tiene)
            for j, (idx, mi_id) in enumerate(zip(vecinos, ids, strict=False)):
                idx = idx[self.ids_sector_[idx] != mi_id]
                if len(idx):
                    sector.iloc[pos[j]] = np.median(self.pm2_sector_[idx])
                    n_vecinos[pos[j]] = len(idx)
        X["precio_m2_sector"] = sector
        X["n_comparables_sector"] = n_vecinos
        return X
