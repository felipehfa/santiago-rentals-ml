import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import BOOLEANAS, NUMERICAS  # noqa: E402


@pytest.fixture
def cfg():
    from src.common import cargar_config
    return cargar_config()


@pytest.fixture
def gold_sintetico():
    """Filas con la forma de gold.aviso_entrenamiento, con una relación precio~superficie."""
    rng = np.random.default_rng(0)
    n = 400
    comunas = np.array(["las-condes", "nunoa", "santiago", "maipu"])
    comuna = rng.choice(comunas, n)
    base = pd.Series(comuna).map({"las-condes": 22000, "nunoa": 14000, "santiago": 11000, "maipu": 8000}).values
    sup = rng.uniform(25, 120, n)
    df = pd.DataFrame({c: rng.integers(0, 4, n).astype(float) for c in NUMERICAS})
    for c in BOOLEANAS:
        df[c] = rng.choice([True, False, None], n)
    df["id_aviso"] = [f"MLC-{i}" for i in range(n)]
    df["comuna"] = comuna
    df["barrio"] = [f"{c}-b{i}" for c, i in zip(comuna, rng.integers(0, 3, n), strict=False)]
    df["latitud"] = -33.4 + rng.normal(0, 0.01, n)
    df["longitud"] = -70.6 + rng.normal(0, 0.01, n)
    df["superficie_util_m2"] = sup
    df["superficie_total_m2"] = sup * 1.1
    df["dormitorios"] = rng.integers(1, 4, n).astype(float)
    df["costo_total_clp"] = base * sup + rng.normal(0, 30000, n)
    df["precio_clp"] = df["costo_total_clp"] * 0.8
    df["gastos_comunes"] = df["costo_total_clp"] * 0.2
    return df
