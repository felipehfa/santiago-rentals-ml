# Plan de la etapa de modelo

Plan acordado el 2026-10-17. Complementa [01_decisiones.md](01_decisiones.md): aquel registra *qué* se decidió; este, *en qué orden* se construye y *qué cambia* respecto de Concepción.

## Qué se rescata de Concepción y cómo cambia

| En Concepción | En este repo |
|---|---|
| `01_ingenieria_variables.py` (todo el dataset junto) | Casi todo lo hace Silver/Gold. Solo quedan en el modelo las variables que usan estadísticos del conjunto (`nivel_barrio`, `precio_m2_sector`, imputación de superficie), dentro de `src/features.py` y **ajustadas solo con train**. |
| `02_seleccion_variables.py` (estabilidad SHAP, curva de k con regla 1-SE, poda por correlación) | `src/seleccion_variables.py`: paso offline que registra su resultado en MLflow y deja la lista en `features_seleccionadas.json`. El CI usa esa lista fija. |
| `01_xgboost.py` / `02_lightgbm.py` (split estratificado, Optuna, bagging de 10 semillas, métricas, baselines, SHAP) | `src/models.py`, `src/train.py` y `src/tune.py`. Cada modelo es un run de MLflow; cada trial de Optuna, un run hijo. |
| Etiquetado oportunidad/caro con z robusto por decil (la confianza por CV del ensamble se quitó el 21-oct-2026) | `src/etiquetas.py`. La referencia (mediana y MAD por decil) se calcula con residuos fuera de entrenamiento y se guarda junto al modelo. |

## Reglas contra la fuga de datos

- Todo lo que usa el target o estadísticos del conjunto se ajusta con **train** y se aplica a val/test.
- `precio_m2_sector`, `nivel_barrio` y los promedios por barrio/comuna usan solo avisos de train. Para una fila de train se **excluye ella misma** (leave-one-out); en Concepción se calculaba con el propio aviso incluido.
- Se excluyen como variables `gastos_comunes`, `precio_clp`, `precio_valor`, `costo_total_uf` y `tiene_rebaja` (son parte del target o se derivan de él), y también `dias_en_mercado`, `n_cambios_*` y las fechas de observación (no existen al momento de publicar).
- La selección de modelo usa la métrica de **validación cruzada** (solo train). El test queda para el gate.
- En la CV de Optuna el constructor de variables se vuelve a ajustar en cada fold.

## Estructura

```
config.yaml · Makefile · requirements.txt · features_seleccionadas.json
src/        common.py, features.py, models.py, etiquetas.py,
            seleccion_variables.py, train.py, tune.py, register.py, promote.py
tests/      con datos sintéticos
```

Reemplaza las carpetas numeradas `01_features/`, `02_entrenamiento/`, `03_evaluacion/`.

## Fases

1. **Esqueleto** (local, MLflow en SQLite): configuración, constructor de variables sin fuga, split, métricas, tests y `ruff`.
2. **Baseline**: XGBoost y LightGBM con parámetros por defecto y bagging de 10 semillas; run en MLflow con métricas, baselines y SHAP.
3. **Selección, tuning y gate**: selección de variables, Optuna (50 trials) por algoritmo, `register.py` (`@challenger`) y `promote.py` (`@champion`, códigos de salida 0/1/2).
4. Acceso a la base con rol de solo lectura, MLflow en Supabase y CI en GitHub Actions. *(pendiente)*
5. Scoring por lotes y `gold.predicciones`. *(pendiente)*

## Limitaciones conocidas

- Con ~4.400 avisos aptos (la carga del detalle sigue en curso) los umbrales del gate y la lista de variables son **provisionales**; se recalculan cuando haya más datos.
- Sin historia real no hay validación temporal útil todavía. Se usa split estratificado por quintil de precio.
- Los finalizados entran solo si son de los últimos 90 días (sesgo de supervivencia conocido).
- Hasta tener el rol de solo lectura, los datos se leen con el connection string del ELT, vía la variable `ENV_FILE`, y se guardan como snapshot local (`data/`, fuera de git).
