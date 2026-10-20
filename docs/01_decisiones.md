# Decisiones del proyecto de modelo

Registro de las decisiones tomadas el 2026-10-17 para la etapa de modelo. Se actualiza cuando cambie alguna.

## 1. Repos y contrato con el ELT

- Son **dos repos separados**: `elt` (scrapers, Bronce, Silver y Gold) y `modelo` (este).
- El ELT corre **local** (Portal Inmobiliario bloquea las IPs de GitHub Actions). El modelo corre en **GitHub Actions**, aunque al inicio se ejecuta local con los mismos comandos.
- El único punto de contacto es `gold.aviso_entrenamiento` en Supabase. El modelo no lee Bronce ni Silver.

## 2. Modelos y optimización

Se reutilizan los dos modelos de Concepción (`modelamiento/02_modelos/01_xgboost.py` y `02_lightgbm.py`), cada uno con su propia optimización:

- Optuna, 50 trials, KFold de 5 **solo sobre train**, early stopping de 50 rondas y hasta 500 rondas.
- Objetivo fijo (`reg:squarederror`): dejarlo dentro del espacio de búsqueda hacía que el ganador cambiara por ruido de los folds.
- Espacio de búsqueda: `learning_rate` 0,01–0,10 (log), `max_depth` 2–6, hojas 8–64, `min_child_weight` / `min_child_samples` 1–20, `subsample` y `colsample_bytree` 0,5–1, regularización L1 y L2 0,01–20 (log), `gamma` / `min_split_gain` 0–5.
- Bagging de 10 semillas con los mejores parámetros, y validación de estabilidad con varias particiones (8 en Concepción; se puede bajar a 5 si pesa).
- Split base: estratificado por quintiles de precio. Se agrega una validación temporal por fecha de publicación.

## 3. Target

**`costo_total_clp`** (pesos). `costo_total_uf` sigue existiendo en Gold pero no se usa para entrenar.

## 4. Arquitectura tomada del webinar

Referencia: `github.com/jcbf08/Webninar_SDC_mlflow`. Se copia la estructura de pasos, con `Makefile`, `config.yaml`, tests, `ruff` y CI.

| Paso | Archivo | Qué hace |
|---|---|---|
| Config | `config.yaml` | Semilla, splits, trials y umbrales del gate |
| Común | `src/common.py` | Lee Gold, pipeline (preprocesamiento + modelo en un objeto), métricas |
| 1 | `src/train.py` | Entrena los modelos base y registra todo en MLflow |
| 2 | `src/tune.py` | Optuna con runs hijos por trial |
| 3 | `src/register.py` | Registra el mejor como `@challenger` |
| 4 | `src/promote.py` | Quality gate; si pasa, `@champion`. Códigos de salida: 0 promueve, 1 sin challenger, 2 rechaza |
| 5 | `src/predict.py` | Scoring por lotes con el `@champion` |

### Qué NO se copia del webinar

- Elegir el mejor run por `test_auc`: usa el test para seleccionar. Aquí se selecciona por la métrica de validación cruzada y el test queda solo para el gate final.
- Optimizar un único modelo: aquí son dos, cada uno con su Optuna.
- Métricas de clasificación (AUC, recall) y la utilidad de retención: aquí es regresión, con MAE, MAPE, R² y MAE del quintil más caro. Métrica de negocio candidata: porcentaje de avisos con error de ±10 %.
- `api.py`, `.devcontainer/` y `GUIA.md`.

### Quality gate

Los datos crecen con el tiempo, así que champion y challenger no comparten test. Se comparan sobre **las mismas filas**: el test del challenger sin los avisos con que se entrenó el champion (train y val). Cada run guarda `ids.json` con sus splits para poder hacerlo.

- **Mejora:** se promueve si el challenger baja el MAE del champion en esas filas.
- **Degradación:** si el MAE del champion en esas filas supera su MAE de test original en más de `degradacion_umbral` (5 %), se promueve al challenger siempre que no sea peor que `margen_vs_champion` (2 %). Sin ese margen se promovería un modelo claramente peor solo porque el anterior envejeció.
- Se informa también la diferencia de MAE con su intervalo de confianza del 95 % (bootstrap). Con pocos avisos comunes el intervalo suele cruzar 0: la diferencia puede ser ruido.
- Criterios absolutos opcionales (MAPE máximo, estabilidad entre particiones); `null` mientras no haya datos para calibrarlos.
- Con menos de `n_minimo_comparacion` avisos comunes no se compara y se rechaza.

`src/monitoreo.py` mide el champion sobre avisos que nunca vio (llegados después de entrenarlo), desglosa el error por semana y registra la degradación en MLflow. Sale con código 3 si supera el umbral, señal de que conviene reentrenar.

`register.py` elige entre algoritmos por `cv_mae` solo entre runs con los mismos datos (el `cv_mae` se mide sobre train); la comparación entre tiempos distintos es la del gate.

## 5. Sin fuga de datos

Las variables que dependen del conjunto (`nivel_barrio`, `precio_m2_sector`, imputación de superficie) se ajustan **solo con train**, dentro del pipeline, para que MLflow las guarde junto al modelo.

**Texto del aviso (2026-10-19).** El título y la descripción aportan calidad que las demás variables no ven (remodelación, vista, precio promocional). Se usan de dos formas, ambas dentro de `ConstructorVariables`:

- Palabras clave (`PALABRAS_CLAVE` en `features.py`): columnas de 1 o 0 sobre el texto normalizado.
- Un modelo de texto (TF-IDF + ridge) que predice `log(costo por m2)` solo con el texto. Su salida entra como la variable `tx_pred_log_m2`. Para las filas de train se calcula **fuera de muestra** (validación cruzada de 5 grupos sobre train), porque con el modelo que las vio la variable conocería su propia respuesta; val, test y avisos nuevos usan el modelo ajustado con todo train. En las pruebas con datos reales la correlación con el precio por m2 es 0,784 en train (fuera de muestra) y 0,778 en test.
- Antes de usar el texto se **quitan todos los números y símbolos de dinero**: casi la mitad de las descripciones escribe el precio y el modelo no debe poder copiarlo (se usa para detectar avisos caros o baratos).
- Un hallazgo: el texto delata el **precio promocional** ("precio válido por X meses, después sigue pagando"), avisos cuyo arriendo publicado es menor que el real.
- En pruebas sobre 3 particiones, las palabras clave bajaron el MAPE cerca de 2 % y, junto al modelo de texto, 6 % (10,86 % a 10,18 %). El texto es en buena parte ubicación (nombres de comunas y barrios), que ya conocen otras variables.

## 6. MLflow

- **Backend** (metadatos: runs, parámetros, métricas, registry y alias): al inicio SQLite local (`mlflow.db`); al automatizar, **Postgres de Supabase**, en un schema aparte.
- **Artefactos** (modelos, gráficos): al inicio carpeta local; al automatizar, **Supabase Storage** (compatible con S3) en un bucket privado, con `boto3`.
- Variables: `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_ROOT`, `MLFLOW_S3_ENDPOINT_URL` y las claves de acceso. En GitHub van como secrets, nunca en el repo.
- **Entrenamiento diario con Optuna completo:** en el runner de Actions tarda unos 22 minutos (50 trials por algoritmo), así que se repite cada día y no depende de los hiperparámetros de la semana anterior. `src.reentrenar` queda para pruebas rápidas.
- **Baselines (lo que haría una planilla, sin entrenar nada):** mediana del costo total por comuna y dormitorios, y precio por m2 mediano de la comuna por la superficie útil. Se registran en cada run como `baseline_*_mae`, junto con `ganancia_vs_mejor_baseline_mae` y `_pct`. Sobre el test del champion v13 (2026-10-20): 86.484 y 87.227 de MAE contra 52.188 del champion (40 % menos). Los modelos lineales o de boosting con pocas variables NO son baselines: son modelos de ML más simples.
- **Retención:** solo se conservan los artefactos del `@champion` vigente y el último run de cada algoritmo (guarda los hiperparámetros del reentrenamiento diario). Las corridas de pruebas y las versiones viejas del registro se eliminan con `mlflow gc` tras cada promoción (el historial de métricas se exporta antes solo si se quiere guardar).
- El cambio de local a Supabase no toca el código, porque `setup_mlflow` lee `MLFLOW_TRACKING_URI`.

## 7. Entrenamiento, CI y CD

- **CI con datos reales** de la base, leídos con un rol de Postgres de **solo lectura**. Los PR desde forks no reciben secrets, así que el CI con datos solo corre en ramas del propio repo. Los tests unitarios con datos sintéticos pueden quedar además.
- **Local primero**, automático después: el `Makefile` ejecuta lo mismo en el PC y en Actions.
- **Workflows de GitHub Actions:**
  - CI (en cada push): `ruff`, `pytest` y un entrenamiento corto contra Gold.
  - Entrenar y promover (a mano o programado): Optuna, registro y gate.
  - Predecir (programado y tras cada promoción): ver la sección 8.
- **Roles de Postgres:** uno de solo lectura para el CI y el entrenamiento, y otro con `SELECT` en Gold e `INSERT` solo en `gold.predicciones` para el scoring.

## 8. Sin API

- **No hay API pública de datos** ni schema `api`: Bronce, Silver y Gold son privados.
- **No hay API de predicción por ahora.** Las predicciones se hacen por lotes: un workflow programado carga el `@champion`, lee los avisos de Gold sin predicción de esa versión, predice y escribe en `gold.predicciones` (aviso, versión del modelo, precio estimado, fecha).
- Es idempotente (clave: aviso + versión) y conserva las predicciones de versiones anteriores. Cuando se promueve un modelo nuevo, se recalcula con esa versión.
- Para saber si hay datos nuevos usa un `cron` diario, ya que el ELT corre local y no puede avisarle a Actions solo. Si no hay nada pendiente, termina en segundos.
- Si más adelante aparece un consumidor externo, una API privada (por ejemplo un Hugging Face Space privado, con la misma imagen de Docker) queda como opción.

## 9. Docker

Sin contenedores al inicio: `requirements.txt` con versiones fijadas y el CI en el runner de GitHub. Se agrega un `Dockerfile` si aparece la API o si hay diferencias entre el PC y Actions. GitHub guarda imágenes en `ghcr.io` y Actions puede ejecutar jobs dentro de ellas, pero no aloja servicios permanentes.

## Fases

1. Esqueleto: `config.yaml`, `Makefile`, `common.py`, `features.py` y tests.
2. Entrenamiento base de ambos modelos sobre Gold.
3. Optuna, registro y gate con umbrales calibrados.
4. CI en GitHub Actions.
5. Scoring por lotes y `gold.predicciones`.
6. Opcional: API privada y Docker.
