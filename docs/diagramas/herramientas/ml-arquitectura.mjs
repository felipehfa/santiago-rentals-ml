import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

// Cuadrícula de 4 columnas (x = 30, 360, 690, 1020) y 3 filas (y = 130, 400, 670): la primera fila va de izquierda a derecha,
// la segunda de derecha a izquierda y la tercera cierra con el scoring.
const C = [30, 360, 690, 1020];
const F = [130, 400, 670];

await construir({
  archivo: "ml-arquitectura",
  destino: DESTINO,
  pagina: "Arquitectura del modelo",
  nodos: [
    { key: "gold", shape: "cylinder", label: "Gold", color: "Datos", x: C[0], y: F[0] - 10, w: 230, h: 110 },
    { key: "datos", shape: "rect", label: "Datos y split", color: "Servicio", x: C[1], y: F[0], w: 230, h: 90 },
    { key: "vars", shape: "rect", label: "Variables", color: "Servicio", x: C[2], y: F[0], w: 230, h: 90 },
    { key: "entrenar", shape: "rect", label: "Entrenamiento", color: "IA", x: C[3], y: F[0], w: 230, h: 90 },
    { key: "mlflow", shape: "cylinder", label: "MLflow", color: "Almacén", x: C[3], y: F[1] - 10, w: 230, h: 110 },
    { key: "gate", shape: "diamond", label: "Gate", color: "Alerta", x: C[2], y: F[1] - 10, w: 230, h: 110 },
    { key: "champ", shape: "rect", label: "@champion", color: "Éxito", x: C[1], y: F[1], w: 230, h: 90 },
    { key: "monit", shape: "rect", label: "Monitoreo", color: "Info", x: C[0], y: F[1], w: 230, h: 90 },
    { key: "scoring", shape: "rect", label: "Scoring", color: "Servicio", x: C[1], y: F[2], w: 230, h: 90 },
    { key: "pred", shape: "cylinder", label: "gold.predicciones", color: "Éxito", x: C[0] - 10, y: F[2] - 10, w: 250, h: 110 },
    nota("n_datos", C[1] - 10, 10, 250, 70, "70/15/15; entrena con\navisos de hasta 60 días"),
    nota("n_vars", C[2] - 10, 10, 250, 70, "Sin fuga: ajustadas\nsolo con train"),
    nota("n_entrenar", C[3] - 10, 10, 250, 70, "Optuna: 50 pruebas\npor algoritmo"),
    nota("n_gate", C[2] - 20, 530, 270, 70, "Promueve si el MAE baja\n≥ 1 % con IC95 > 0"),
    nota("n_scoring", C[2] - 20, 690, 330, 90, "Etiqueta de mercado,\nconfianza y banda de precio"),
  ],
  aristas: [
    { from: "gold", to: "datos", label: "aptos" },
    { from: "datos", to: "vars", label: "splits" },
    { from: "vars", to: "entrenar", label: "matrices" },
    { from: "entrenar", to: "mlflow", label: "runs", fromSide: "s", toSide: "n" },
    { from: "mlflow", to: "gate", label: "@challenger" },
    { from: "gate", to: "champ", label: "promueve" },
    { from: "champ", to: "monit", label: "mide" },
    { from: "champ", to: "scoring", label: "versión", fromSide: "s", toSide: "n" },
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
