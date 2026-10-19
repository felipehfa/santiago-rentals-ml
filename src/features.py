"""Construcción de variables sin fuga de datos.

Gold ya entrega los avisos limpios y tipados. Lo que queda acá son las variables que dependen
del conjunto de entrenamiento (precio por m2 del sector, nivel del barrio y de la comuna,
imputación de superficie). Todas se ajustan SOLO con train.

Para una fila que pertenece a train, el aviso mismo se excluye del cálculo (leave-one-out).
En Concepción el comparador del sector incluía el propio costo de la fila, lo que filtraba el
target hacia la variable.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
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
    "fecha_publicacion_precision", "estado_publicacion", "construido_en", "uv_rsh", "titulo", "descripcion",
]


def _a_float(serie: pd.Series) -> pd.Series:
    def conv(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return np.nan
        if isinstance(v, str):
            return {"true": 1.0, "false": 0.0}.get(v.strip().lower(), np.nan)
        return float(v)

    return serie.map(conv).astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Texto del aviso (título y descripción)
# ──────────────────────────────────────────────────────────────────────────────

# Palabras clave que se buscan sobre el texto ya normalizado (sin tildes, sin números). Se evita todo lo que
# dependa de cifras: el texto puede traer el precio del propio aviso y eso no debe llegar al modelo.
PALABRAS_CLAVE = {
    "remodelado": r"remodel|renovad|refaccion",
    "estrenar": r"a estrenar|estrenar|recien entregad|nuevo proyecto",
    "vista_panoramica": r"vista panoramic|vista despejad|vista al (cerro|parque|city|skyline)|gran vista"
                        r"|excelente vista",
    "terraza": r"terraza", "logia": r"logia", "orient_norte": r"orientacion (norte|nor)",
    "aire_acond": r"aire acondicionado", "calef_central": r"calefaccion central|losa radiante",
    "walkin": r"walk.?in", "cocina_equipada": r"cocina (equipada|amoblada|full)",
    "lujo": r"lujo|premium|exclusiv|penthouse|atico", "duplex_loft": r"duplex|loft|triplex",
    "impecable": r"impecable|excelente estado|como nuevo", "luminoso": r"luminos|mucha luz",
    "amoblado_txt": r"amoblad|equipad|full equip", "sin_muebles": r"sin muebl|sin amoblar",
    "gc_incluidos": r"gastos comunes incluid|gc incluid", "mascotas": r"mascota|pet friendly",
    "gimnasio": r"gimnasio|gym", "piscina": r"piscina", "quincho": r"quincho",
    "cowork": r"cowork|sala de estudio", "conserje": r"conserje|porteria", "estudio": r"\bestudio\b|monoambiente",
    "metro_cerca": r"a pasos del metro|cerca del metro|frente al metro",
    # Precio promocional: el arriendo publicado vale solo unos meses y después sube. El aviso parece barato.
    "precio_promocional": r"precio (valido|promocional)|valido por|primeros meses|oferta (de )?lanzamiento"
                          r"|despues (sigue|pagas|pagar|vuelve)|luego (sigue|pagas|pagar)|mes despues",
}


_RE_CLAVE = {k: re.compile(p) for k, p in PALABRAS_CLAVE.items()}


def normalizar_texto(titulo, descripcion) -> str:
    """Minúsculas, sin tildes, sin dígitos ni símbolos de dinero. Los números se quitan a propósito: si el
    aviso escribe su precio, el modelo de texto no debe poder copiarlo."""
    s = f"{titulo or ''} . {descripcion or ''}".lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[\d$€]+|\buf\b|\bclp\b|\bpesos\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _haversine_km(lat, lon, lat0, lon0):
    """Distancia en km entre pares de coordenadas en grados (NaN si falta alguna)."""
    p = np.pi / 180
    a = np.sin((lat0 - lat) * p / 2) ** 2 + np.cos(lat * p) * np.cos(lat0 * p) * np.sin((lon0 - lon) * p / 2) ** 2
    return 2 * (RADIO_TIERRA_M / 1000) * np.arcsin(np.sqrt(a))


class ConstructorVariables:
    """Se ajusta con train (`fit`) y transforma cualquier conjunto (`transform`)."""

    def __init__(self, radio_sector_m=300, multiplicador_iqr=3, k_suavizado=20, n_niveles_barrio=5,
                 target="costo_total_clp", puntos_referencia=None, texto=None):
        self.radio_sector_m = radio_sector_m
        self.multiplicador_iqr = multiplicador_iqr
        self.k_suavizado = k_suavizado
        self.n_niveles_barrio = n_niveles_barrio
        self.target = target
        self.puntos_referencia = puntos_referencia or {}
        self.texto = {"activo": True, "ngram_max": 2, "min_df": 8, "max_features": 30000, "alpha": 3.0, "folds": 5,
                      **(texto or {})}
        # Los constructores pickleados de modelos anteriores no tienen este atributo: transform lo lee
        # con getattr y, sin él, no agrega las variables nuevas (así el champion viejo sigue funcionando).
        self.variables_derivadas = True

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

        # Centro de cada comuna: mediana de las coordenadas de sus avisos de train (solo posiciones, no precios).
        coords = df[["latitud", "longitud"]].astype(float).assign(comuna=comuna)
        self.centros_comuna_ = {
            c: (float(g["latitud"].median()), float(g["longitud"].median()))
            for c, g in coords.dropna().groupby("comuna") if len(g) >= 5
        }

        self.usa_texto = bool(self.texto["activo"]) and "descripcion" in df.columns
        if self.usa_texto:
            self._ajustar_texto(df)
        return self

    # ── modelo de texto ───────────────────────────────────────────────────────
    @staticmethod
    def _textos(df: pd.DataFrame) -> pd.Series:
        titulo = df["titulo"] if "titulo" in df.columns else pd.Series([None] * len(df), index=df.index)
        return pd.Series([normalizar_texto(a, b) for a, b in zip(titulo, df["descripcion"], strict=True)],
                         index=df.index)

    def _modelo_texto(self):
        t = self.texto
        vec = TfidfVectorizer(ngram_range=(1, t["ngram_max"]), min_df=t["min_df"], max_features=t["max_features"],
                              sublinear_tf=True)
        return vec, Ridge(alpha=t["alpha"])

    def _ajustar_texto(self, df: pd.DataFrame) -> None:
        """Ajusta TF-IDF + ridge sobre log(costo por m2 útil) usando SOLO el texto como entrada.

        Para cada aviso de train guarda una predicción fuera de muestra (validación cruzada sobre train): sale de un
        modelo que no vio el precio de ese aviso. Val, test y avisos nuevos usan el modelo ajustado con todo train.
        Sin esto, el modelo principal se apoyaría en una variable que conoce la respuesta de las filas de train."""
        textos = self._textos(df)
        util = df["superficie_util_m2"].astype(float)
        ok = (util > 0) & df[self.target].notna() & (textos.str.len() >= 20)
        y = np.log((df[self.target].astype(float) / util).clip(lower=1))
        pos_ok = np.flatnonzero(ok.to_numpy())
        oof = np.full(len(df), np.nan)
        if len(pos_ok) >= 2 * self.texto["folds"]:
            for a, b in KFold(self.texto["folds"], shuffle=True, random_state=1).split(np.arange(len(df))):
                a = np.intersect1d(a, pos_ok)
                vec, ridge = self._modelo_texto()
                ridge.fit(vec.fit_transform(textos.iloc[a]), y.iloc[a])
                b = b[textos.iloc[b].str.len() >= 20]
                oof[b] = ridge.predict(vec.transform(textos.iloc[b]))
            self.vec_texto_, self.ridge_texto_ = self._modelo_texto()
            self.ridge_texto_.fit(self.vec_texto_.fit_transform(textos.iloc[pos_ok]), y.iloc[pos_ok])
            if hasattr(self.vec_texto_, "stop_words_"):  # lista enorme de términos descartados: no hace falta guardarla
                del self.vec_texto_.stop_words_
        else:
            self.vec_texto_ = self.ridge_texto_ = None
        self.texto_oof_ = dict(zip(df[ID_COL], oof, strict=True))

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
        if getattr(self, "variables_derivadas", False):
            X = self._derivadas(X, df, comuna)
        if getattr(self, "usa_texto", False):
            X = self._variables_texto(X, df)
        return X

    def _variables_texto(self, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Palabras clave, largo del texto y la predicción del modelo de texto. Sin texto (o sin la columna) todo
        queda en NaN: el modelo lo trata como dato ausente."""
        X = X.copy()
        n = len(df)
        nombres = [f"tx_{k}" for k in PALABRAS_CLAVE] + ["tx_largo", "tx_n_pal", "tx_n_excl", "tx_pred_log_m2"]
        if "descripcion" not in df.columns:
            return pd.concat([X, pd.DataFrame(np.nan, index=X.index, columns=nombres)], axis=1)
        textos = self._textos(df).reset_index(drop=True)
        vacio = (textos.str.len() < 20).to_numpy()
        cols = {f"tx_{k}": textos.map(lambda s, rx=rx: 1.0 if rx.search(s) else 0.0).to_numpy()
                for k, rx in _RE_CLAVE.items()}
        cols["tx_largo"] = textos.str.len().astype(float).to_numpy()
        cols["tx_n_pal"] = textos.str.split().map(len).astype(float).to_numpy()
        cols["tx_n_excl"] = df["descripcion"].fillna("").astype(str).str.count("!").astype(float).to_numpy()
        pred = np.full(n, np.nan)
        if self.ridge_texto_ is not None:
            hay = np.flatnonzero(~vacio)
            if len(hay):
                pred[hay] = self.ridge_texto_.predict(self.vec_texto_.transform(textos.iloc[hay]))
            # Filas de train: la predicción fuera de muestra, no la del modelo que las vio
            oof = df[ID_COL].map(self.texto_oof_).astype(float).to_numpy()
            en_train = df[ID_COL].isin(self.ids_train_).to_numpy()
            pred = np.where(en_train, oof, pred)
        cols["tx_pred_log_m2"] = pred
        out = pd.DataFrame(cols, index=X.index)
        out.loc[vacio, [c for c in out.columns if c != "tx_n_excl"]] = np.nan
        return pd.concat([X, out[nombres]], axis=1)

    # ── variables derivadas ───────────────────────────────────────────────────
    def _derivadas(self, X: pd.DataFrame, df: pd.DataFrame, comuna: pd.Series) -> pd.DataFrame:
        """Razones de superficie, equipamiento agregado y distancias a puntos de referencia.
        Solo usan atributos del propio aviso y posiciones de train (nada del precio)."""
        X = X.copy()
        dorm = df["dormitorios"].astype(float).clip(lower=1)
        util = df["superficie_util_m2"].astype(float)
        total = df["superficie_total_m2"].astype(float).replace(0, np.nan)
        X["sup_por_dorm"] = X["superficie_util_m2"] / dorm
        X["banos_por_dorm"] = df["banos"].astype(float) / dorm
        X["ratio_util_total"] = util / total
        X["log_sup_util"] = np.log(X["superficie_util_m2"].clip(lower=1))
        X["est_por_dorm"] = df["estacionamientos"].astype(float) / dorm
        X["extras"] = df["estacionamientos"].astype(float).fillna(0) + df["bodegas"].astype(float).fillna(0)
        X["amenidades"] = X[["piscina", "quincho", "conserjeria", "ascensor", "tiene_bodega", "condominio_cerrado",
                             "estacionamiento_visitas"]].sum(axis=1)
        X["piso_rel"] = X["piso_unidad"] / X["deptos_por_piso"].replace(0, np.nan)
        X["nuevo_con_ascensor"] = (X["antiguedad_anos"].fillna(99) <= 3).astype(float) * X["ascensor"].fillna(0)
        X["servicios_cerca"] = X[[c for c in X.columns if c.startswith("cantidad_")]].sum(axis=1, min_count=1)
        X["dist_media_servicios"] = X[[c for c in X.columns if c.startswith("distancia_min_m_")]].mean(axis=1)

        lat, lon = df["latitud"].astype(float).to_numpy(), df["longitud"].astype(float).to_numpy()
        for nombre, (lat0, lon0) in self.puntos_referencia.items():
            X[f"dist_{nombre}_km"] = _haversine_km(lat, lon, lat0, lon0)
        c_lat = comuna.map(lambda c: self.centros_comuna_.get(c, (np.nan, np.nan))[0]).to_numpy(dtype=float)
        c_lon = comuna.map(lambda c: self.centros_comuna_.get(c, (np.nan, np.nan))[1]).to_numpy(dtype=float)
        X["dist_centro_comuna_km"] = _haversine_km(lat, lon, c_lat, c_lon)
        return X
