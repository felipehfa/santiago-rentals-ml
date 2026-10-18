# santiago-rentals-ml

Etapa de desarrollo del modelo de ML para estimar si un arriendo de departamento en la Región Metropolitana está caro, barato o a precio de mercado.

## Contrato con `santiago-rentals-elt`

- La única entrada es `gold.aviso_entrenamiento` (filtrar `apto_entrenamiento`). Este repo nunca lee Bronce ni Silver.
- Los scrapers y la construcción de Silver/Gold viven en `santiago-rentals-elt`; aquí no se escribe en esas capas.
- Las variables que usan estadísticos del conjunto (`nivel_barrio`, `precio_m2_sector`, imputación de superficie) se ajustan **solo con train**, para no repetir la fuga de datos de Concepción (ver `docs/01_antecedentes_concepcion_y_databricks.md` en el repo ELT).

## Estructura

```
config.yaml · Makefile · requirements.txt · features_seleccionadas.json
src/        common, features (sin fuga), models, etiquetas,
            seleccion_variables, train, tune, register, promote
tests/      datos sintéticos
docs/       01_decisiones.md (qué se decidió), 02_plan.md (en qué orden se construye)
```

## Cómo ejecutar

Con el entorno conda `data-science` y, mientras no exista el rol de solo lectura,
`ENV_FILE` apuntando al `.env` del ELT (el connection string no se copia a este repo):

```bash
python -m src.seleccion_variables   # una vez, o al cambiar mucho los datos
python -m src.train                 # modelos base
python -m src.tune                  # Optuna (50 trials) por algoritmo
python -m src.register              # mejor por cv_mae -> @challenger
python -m src.promote               # quality gate -> @champion (0 promueve, 1 sin challenger, 2 rechaza)
python -m src.monitoreo            # degradación del champion con avisos nuevos (3 = degradado)
python -m pytest && python -m ruff check src tests
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La primera corrida descarga Gold a `data/` (fuera de git) y las siguientes la reutilizan.

## Decisiones tomadas

Detalle en `docs/01_decisiones.md`. En resumen:

- Dos repos separados (`elt` local, `modelo` en GitHub Actions); el contacto es solo Gold.
- Modelos: XGBoost y LightGBM de Concepción, cada uno con Optuna (50 trials, KFold 5 solo sobre train).
- Target: `costo_total_clp`.
- Arquitectura del webinar `jcbf08/Webninar_SDC_mlflow`: `train` → `tune` → `register` → `promote` (quality gate) → `predict`, con MLflow y alias `@challenger` / `@champion`.
- MLflow local al inicio; Postgres y Storage de Supabase al automatizar.
- Sin API pública: scoring por lotes que escribe en `gold.predicciones`.

## Decisiones pendientes

- Umbrales del quality gate (se fijan tras la primera corrida base).
- Estrategia de validación definitiva (estratificada por precio, temporal o ambas) y manejo del sesgo de supervivencia entre avisos activos y finalizados.
- Si el CI y el scoring usan un `Dockerfile` o el runner directo.
