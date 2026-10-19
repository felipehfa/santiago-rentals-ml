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
            arquitectura_ml.html y promocion_modelos.html (diagramas interactivos; fuente en los .json)
```

## Cómo ejecutar

Con el entorno conda `data-science` y un `.env` propio con `BD_STRING_READ`, el connection string del rol
`ml_lectura` (solo `SELECT` sobre `gold.aviso_entrenamiento`; ver
`santiago-rentals-elt/scripts/roles_minimo_privilegio.sql`). `BD_STRING_SCORING` es el del rol `ml_scoring`,
para el scoring por lotes que aún no existe. El `.env` no se versiona.

```bash
python -m src.tune                  # reselecciona variables y luego Optuna (50 trials) por algoritmo
python -m src.seleccion_variables   # solo la selección (tune ya la ejecuta; --sin-seleccion la omite)
python -m src.train                 # modelos base
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
- La selección de variables se rehace en cada entrenamiento (`tune`), con los mismos datos; una lista calculada
  con otros datos hace fallar la corrida.
- Train usa solo avisos publicados hasta 60 días antes de la última visita (`datos.ventana_dias`): los precios de
  oferta viejos suelen estar desactualizados. Validación y test quedan completos, para comparar con el champion.
- Variables derivadas: razones de superficie, equipamiento agregado y distancias en km al centro de la comuna,
  Plaza de Armas, El Golf, Las Condes y Providencia (puntos en `config.yaml`).
- Target: `costo_total_clp`.
- Arquitectura del webinar `jcbf08/Webninar_SDC_mlflow`: `train` → `tune` → `register` → `promote` (quality gate) → `predict`, con MLflow y alias `@challenger` / `@champion`.
- MLflow local al inicio; Postgres y Storage de Supabase al automatizar.
- Sin API pública: scoring por lotes que escribe en `gold.predicciones`.

## Decisiones pendientes

- Umbrales del quality gate (se fijan tras la primera corrida base).
- Estrategia de validación definitiva (estratificada por precio, temporal o ambas) y manejo del sesgo de supervivencia entre avisos activos y finalizados.
- Si el CI y el scoring usan un `Dockerfile` o el runner directo.
