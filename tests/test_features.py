import numpy as np

from src.common import dividir
from src.train import nuevo_constructor

VARIABLES_AJUSTADAS = ["precio_m2_sector", "nivel_barrio", "barrio_precio_m2", "comuna_precio_m2"]


def test_variables_de_train_no_ven_su_propio_target(gold_sintetico, cfg):
    """Cambiar el costo de un aviso de train no debe mover SUS PROPIAS variables ajustadas."""
    train, _, _ = dividir(gold_sintetico, cfg)
    base = nuevo_constructor(cfg).fit(train).transform(train)

    alterado = train.copy()
    alterado.loc[0, "costo_total_clp"] *= 10
    X = nuevo_constructor(cfg).fit(alterado).transform(alterado)

    for col in VARIABLES_AJUSTADAS:
        assert np.isclose(base.loc[0, col], X.loc[0, col]), f"{col} filtra el target de la propia fila"


def test_val_no_influye_en_el_ajuste(gold_sintetico, cfg):
    """Cambiar el target de val/test no debe alterar las variables de train."""
    train, val, _ = dividir(gold_sintetico, cfg)
    c = nuevo_constructor(cfg).fit(train)
    antes = c.transform(train)
    val2 = val.copy()
    val2["costo_total_clp"] *= 5
    c.transform(val2)
    despues = c.transform(train)
    assert antes.equals(despues)


def test_el_sector_no_cuenta_al_propio_aviso(gold_sintetico, cfg):
    train, _, _ = dividir(gold_sintetico, cfg)
    X = nuevo_constructor(cfg).fit(train).transform(train)
    assert (X["n_comparables_sector"] >= 0).all()
    # Un solo aviso aislado no puede ser su propio comparable.
    aislado = train.iloc[[0]].copy()
    aislado["latitud"], aislado["longitud"] = -33.0, -71.0
    c = nuevo_constructor(cfg).fit(aislado)
    assert c.transform(aislado).loc[0, "n_comparables_sector"] == 0


def test_dividir_conserva_proporciones_y_no_repite_avisos(gold_sintetico, cfg):
    train, val, test = dividir(gold_sintetico, cfg)
    n = len(gold_sintetico)
    assert abs(len(test) / n - cfg["split"]["test"]) < 0.02
    assert abs(len(val) / n - cfg["split"]["val"]) < 0.02
    ids = [set(x["id_aviso"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_la_superficie_se_imputa_con_estadisticos_de_train(gold_sintetico, cfg):
    train, val, _ = dividir(gold_sintetico, cfg)
    c = nuevo_constructor(cfg).fit(train)
    sin_util = val.copy()
    sin_util["superficie_util_m2"] = np.nan
    X = c.transform(sin_util)
    assert X["superficie_util_m2"].notna().all()
    assert np.allclose(X["superficie_util_m2"], sin_util["superficie_total_m2"] * c.razon_util_total_)
