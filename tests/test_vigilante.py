"""Vigilante del entrenamiento: edad del último run y elección entre tune y reentrenar."""
from types import SimpleNamespace

from src import vigilar_entrenamiento as v


def test_edad_en_horas():
    assert v.edad_horas(None) is None
    assert v.edad_horas(0, ahora_s=7200) == 2.0


def test_elige_el_entrenamiento_mas_reciente_entre_tune_y_reentrenar():
    runs = {"tune_": [SimpleNamespace(info=SimpleNamespace(run_name="tune_lightgbm", start_time=1_000))],
            "reentrenar_": [SimpleNamespace(info=SimpleNamespace(run_name="reentrenar_xgboost", start_time=2_000))]}

    class Cliente:
        def search_runs(self, ids, filter_string, order_by, max_results):
            return next(r for p, r in runs.items() if f"'{p}%'" in filter_string)

    assert v.ultimo_entrenamiento(Cliente(), "1") == ("reentrenar_xgboost", 2_000)


def test_sin_runs_devuelve_none():
    class Cliente:
        def search_runs(self, *a, **k):
            return []

    assert v.ultimo_entrenamiento(Cliente(), "1") is None


def test_limpieza_conserva_alias_ultimo_tune_y_lo_reciente():
    from src.limpiar_artefactos import elegir
    dia = 86_400_000
    ahora = 100 * dia
    runs = [
        {"id": "champ", "nombre": "tune_lightgbm", "inicio_ms": 10 * dia},
        {"id": "tune_viejo", "nombre": "tune_lightgbm", "inicio_ms": 20 * dia},
        {"id": "tune_nuevo", "nombre": "tune_lightgbm", "inicio_ms": 90 * dia},
        {"id": "tune_xgb", "nombre": "tune_xgboost", "inicio_ms": 30 * dia},
        {"id": "trial", "nombre": "trial_003", "inicio_ms": 30 * dia},
        {"id": "reciente", "nombre": "trial_004", "inicio_ms": 99 * dia},
        {"id": "v2", "nombre": "reentrenar_lightgbm", "inicio_ms": 40 * dia},
    ]
    versiones = [{"version": 1, "run_id": "champ", "alias": ["champion"]},
                 {"version": 2, "run_id": "v2", "alias": []}]
    borrar, versiones_borrar = elegir(runs, versiones, ahora)
    assert borrar == {"tune_viejo", "trial", "v2"}
    assert versiones_borrar == [2]
