import numpy as np
import pandas as pd

from src import etiquetas
from src.common import metricas
from src.models import PARAMS_BASE, Ensamble
from src.promote import evaluar_gate
from src.train import preparar


def test_metricas_de_una_prediccion_perfecta():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    m = metricas(y, y)
    assert m["mae"] == 0 and m["pct_dentro_10"] == 100


def test_ensamble_predice_una_fila_por_aviso():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = 500 + 100 * X["a"] + rng.normal(0, 5, 120)
    for alg in ("xgboost", "lightgbm"):
        e = Ensamble(alg, PARAMS_BASE[alg]).fit(X[:90], y[:90], X[90:], y[90:], 3, 1, 50, 10)
        assert e.matriz(X).shape == (3, 120)
        assert set(e.shap_nativo(X)) == set("abcd")


def test_etiquetas_marcan_los_extremos():
    rng = np.random.default_rng(2)
    real = rng.uniform(300_000, 900_000, 500)
    pred = real + rng.normal(0, 20_000, 500)
    ref = etiquetas.ajustar_referencia(real, pred, 10)
    barato = etiquetas.etiquetar([500_000], [800_000], [0.05], ref)
    caro = etiquetas.etiquetar([800_000], [500_000], [0.05], ref)
    assert barato.loc[0, "etiqueta"] == "oportunidad" and caro.loc[0, "etiqueta"] == "caro"


def test_la_referencia_sobrevive_a_json():
    import json
    real = np.linspace(300_000, 900_000, 200)
    ref = etiquetas.ajustar_referencia(real, real * 1.02, 10)
    ref2 = json.loads(json.dumps(ref))
    a = etiquetas.etiquetar(real, real * 1.02, np.zeros(200), ref)
    b = etiquetas.etiquetar(real, real * 1.02, np.zeros(200), ref2)
    assert a["etiqueta"].equals(b["etiqueta"])


GATE = {"mape_max": None, "estabilidad_cv_max": None, "mejora_min": 0.0, "degradacion_umbral": 0.05,
        "margen_vs_champion": 0.02, "n_minimo_comparacion": 100}
PROPIAS = {"test_mape": 10.0}


def test_gate_sin_champion_solo_aplica_umbrales_absolutos():
    assert evaluar_gate(PROPIAS, None, None, GATE) == []
    assert evaluar_gate(PROPIAS, None, None, {**GATE, "mape_max": 8})
    assert evaluar_gate({**PROPIAS, "estabilidad_cv_mae": 0.2}, None, None, {**GATE, "estabilidad_cv_max": 0.1})


def test_gate_promueve_si_mejora_al_champion():
    assert evaluar_gate(PROPIAS, {"mae_challenger": 90, "mae_champion": 100}, 0.0, GATE) == []


def test_gate_rechaza_si_no_mejora_y_el_champion_no_esta_degradado():
    assert evaluar_gate(PROPIAS, {"mae_challenger": 101, "mae_champion": 100}, 0.01, GATE)


def test_gate_promueve_si_el_champion_se_degrado_y_el_challenger_no_es_peor():
    # Peor por 1 % (dentro del margen del 2 %) pero con el champion 10 % degradado: se promueve.
    assert evaluar_gate(PROPIAS, {"mae_challenger": 101, "mae_champion": 100}, 0.10, GATE) == []


def test_gate_rechaza_si_el_champion_esta_degradado_pero_el_challenger_es_claramente_peor():
    assert evaluar_gate(PROPIAS, {"mae_challenger": 110, "mae_champion": 100}, 0.20, GATE)


def test_filas_comunes_excluyen_lo_que_el_champion_vio():
    from src.comparacion import filas_comunes, filas_nuevas
    champion = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    challenger = {"train": ["x"], "val": ["y"], "test": ["a", "d", "e", "f"]}
    assert filas_comunes(challenger, champion) == {"d", "e", "f"}  # "a" estaba en el train del champion
    assert filas_nuevas(champion, ["a", "d", "z"]) == {"z"}        # el test del champion tampoco es nuevo


def test_diferencia_ic_detecta_un_modelo_mejor():
    from src.comparacion import diferencia_ic
    rng = np.random.default_rng(0)
    y = rng.uniform(300, 900, 400)
    dif, lo, hi = diferencia_ic(y, y + rng.normal(0, 60, 400), y + rng.normal(0, 20, 400))
    assert dif > 0 and lo > 0  # a es peor que b y el intervalo no cruza 0


def test_error_por_semana_agrupa_por_semana_de_primera_vez_visto():
    from src.monitoreo import error_por_semana
    df = pd.DataFrame({"costo_total_clp": [100.0, 200.0, 100.0],
                       "primera_vez_visto": pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-13"], utc=True)})
    out = error_por_semana(df, np.array([110.0, 180.0, 100.0]), "costo_total_clp")
    assert out["n"].tolist() == [2, 1] and out["mae"].tolist() == [15.0, 0.0]


def test_preparar_de_punta_a_punta(gold_sintetico, cfg, tmp_path, monkeypatch):
    ruta = tmp_path / "gold.parquet"
    gold_sintetico.to_parquet(ruta, index=False)
    cfg["datos"]["snapshot"] = str(ruta)
    monkeypatch.setattr("src.common.RAIZ", tmp_path)
    d = preparar(cfg)
    assert len(d["X_train"]) == len(d["train"]) and list(d["X_train"].columns) == d["variables"]
    assert not d["X_train"].columns.duplicated().any()


def test_preparar_descarta_gastos_comunes_absurdos(gold_sintetico, cfg):
    from src.common import preparar_datos
    gold_sintetico.loc[0, "gastos_comunes"] = gold_sintetico.loc[0, "precio_clp"] * 3
    gold_sintetico.loc[1, "gastos_comunes"] = None  # sin dato: se conserva
    out = preparar_datos(gold_sintetico, cfg)
    assert "MLC-0" not in set(out["id_aviso"]) and "MLC-1" in set(out["id_aviso"])
