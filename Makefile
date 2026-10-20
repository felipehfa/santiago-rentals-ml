# Los mismos comandos corren en el PC y en GitHub Actions.
# En Windows sin make, ejecutar la línea de cada objetivo a mano con el python del entorno.
PY ?= python

.PHONY: datos seleccion train tune register promote scoring monitoreo test lint

datos:       ## vuelve a descargar Gold (necesita BD_STRING)
	$(PY) -c "from src.common import cargar_config, descargar_gold; print(len(descargar_gold(cargar_config())))"

seleccion:   ## selección de variables -> features_seleccionadas.json
	$(PY) -m src.seleccion_variables

train:       ## modelos base
	$(PY) -m src.train

tune:        ## Optuna + ensamble final por algoritmo
	$(PY) -m src.tune

register:    ## mejor run por cv_mae -> @challenger
	$(PY) -m src.register

promote:     ## quality gate -> @champion (0 promueve, 1 sin challenger, 2 rechaza)
	$(PY) -m src.promote

scoring:     ## predice con el @champion los avisos pendientes -> gold.predicciones
	$(PY) -m src.scoring

monitoreo:   ## degradación del champion con avisos nuevos (3 = degradado)
	$(PY) -m src.monitoreo

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
