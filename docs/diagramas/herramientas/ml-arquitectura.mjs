import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

await construir({
  archivo: "ml-arquitectura",
  destino: DESTINO,
  pagina: "Arquitectura del modelo",
  nodos: [
    { key: "gold", shape: "cylinder", label: "Gold", color: "Datos", x: 40, y: 290, w: 200, h: 100 },
    { key: "datos", shape: "rect", label: "Datos y split", color: "Servicio", x: 380, y: 305, w: 200, h: 70 },
    { key: "vars", shape: "rect", label: "Variables", color: "Servicio", x: 720, y: 305, w: 200, h: 70 },
    { key: "entrenar", shape: "rect", label: "Entrenamiento", color: "IA", x: 1060, y: 305, w: 200, h: 70 },
    { key: "mlflow", shape: "cylinder", label: "MLflow", color: "Almacén", x: 1400, y: 290, w: 200, h: 100 },
    { key: "gate", shape: "diamond", label: "Gate", color: "Alerta", x: 1740, y: 285, w: 200, h: 110 },
    { key: "champ", shape: "rect", label: "@champion", color: "Éxito", x: 2080, y: 305, w: 200, h: 70 },
    { key: "monit", shape: "rect", label: "Monitoreo", color: "Info", x: 2380, y: 305, w: 160, h: 70 },
    { key: "scoring", shape: "rect", label: "Scoring por lotes", color: "Servicio", x: 1740, y: 640, w: 200, h: 70 },
    { key: "pred", shape: "cylinder", label: "gold.predicciones", color: "Éxito", x: 1400, y: 625, w: 200, h: 100 },
    nota("n_datos", 330, 130, 320, 98, "Aptos de Gold de 12 comunas, costo hasta\n10 M y gastos hasta 50 %. Split 70/15/15;\ntrain solo con avisos de hasta 60 días."),
    nota("n_vars", 670, 130, 320, 98, "Sin fuga: precio_m2_sector, barrio, comuna y\nmodelo de texto solo con train. Derivadas,\ndistancias y 26 palabras clave del texto."),
    nota("n_entrenar", 1010, 130, 320, 98, "Selección de variables y Optuna: 50 pruebas\npor algoritmo (XGBoost, LightGBM), KFold 5,\nbagging de 10 semillas. Elige por CV MAE."),
    nota("n_mlflow", 1350, 130, 320, 98, "Postgres + bucket. Cada run guarda métricas,\nids de los splits, config, commit, versiones\ny el modelo con su constructor de variables."),
    nota("n_gate", 1690, 120, 330, 118, "Promueve si el MAE baja ≥ 1 % sobre las\nmismas filas y el IC95 de la mejora es > 0,\no si el champion se degradó > 5 % y el\nchallenger no es peor."),
    nota("n_baselines", 40, 520, 340, 98, "Baselines de planilla (sin entrenar): mediana\npor comuna y dormitorios, y precio/m² de la\ncomuna × superficie. El modelo debe superarlos."),
    nota("n_scoring", 1700, 760, 350, 98, "Predice avisos sin predicción de la versión\nvigente o vistos de nuevo. Guarda etiqueta de\nmercado, confianza y banda de precio."),
    nota("n_monit", 2250, 130, 300, 78, "Mide la degradación del champion con\navisos nuevos (código 3 si empeora)."),
  ],
  aristas: [
    { from: "gold", to: "datos", label: "aptos" },
    { from: "datos", to: "vars", label: "train/val/test" },
    { from: "vars", to: "entrenar", label: "matrices" },
    { from: "entrenar", to: "mlflow", label: "runs y modelo" },
    { from: "mlflow", to: "gate", label: "@challenger" },
    { from: "gate", to: "champ", label: "promueve" },
    { from: "champ", to: "monit", label: "mide" },
    { from: "champ", to: "scoring", label: "versión vigente", fromSide: "s", toSide: "e" },
    { from: "scoring", to: "pred", label: "INSERT" },
  ],
  eventos: [
    { ref: "promueve", name: "Promueve", primitive: "OCCURRENCE", sentence: "{target} promueve al challenger" },
  ],
  historias: [
    {
      nombre: "Entrenamiento y promoción de un modelo",
      pasos: [
        { ev: "envia", de: "gold", a: "datos", espera: 0 },
        { ev: "envia", de: "datos", a: "vars", tras: [0] },
        { ev: "envia", de: "vars", a: "entrenar", tras: [1] },
        { ev: "procesa", nodo: "entrenar", tras: [2] },
        { ev: "envia", de: "entrenar", a: "mlflow", espera: 700 },
        { ev: "envia", de: "mlflow", a: "gate", tras: [4] },
        { ev: "promueve", nodo: "gate", tras: [5] },
        { ev: "envia", de: "gate", a: "champ", tras: [5] },
        { ev: "envia", de: "champ", a: "monit", tras: [7] },
      ],
    },
    {
      nombre: "Scoring de los avisos con el champion",
      pasos: [
        { ev: "envia", de: "champ", a: "scoring", espera: 0 },
        { ev: "procesa", nodo: "scoring", tras: [0] },
        { ev: "envia", de: "scoring", a: "pred", espera: 600, tras: [0] },
        { ev: "procesa", nodo: "pred", tras: [2] },
      ],
    },
  ],
});
