import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

// Fila 1: el entrenamiento diario (cron -> entrenar -> MLflow -> gate -> champion -> monitoreo).
// Fila 3: el scoring (cron cada 30 min -> scoring -> predicciones), que lee Gold y el champion.
const C = [20, 330, 640, 950, 1260, 1570];

await construir({
  archivo: "ml-automatizacion",
  destino: DESTINO,
  pagina: "ML automático en GitHub",
  nodos: [
    { key: "cron_e", shape: "circle", label: "Cron diario", color: "Externo", x: C[0], y: 135, w: 200, h: 110 },
    { key: "entrenar", shape: "rect", label: "1. Entrenar", color: "IA", x: C[1], y: 145, w: 200, h: 90 },
    { key: "mlflow", shape: "cylinder", label: "MLflow", color: "Almacén", x: C[2], y: 130, w: 200, h: 110 },
    { key: "gate", shape: "diamond", label: "2. Gate", color: "Alerta", x: C[3], y: 130, w: 200, h: 120 },
    { key: "champ", shape: "rect", label: "@champion", color: "Éxito", x: C[4], y: 145, w: 200, h: 90 },
    { key: "monit", shape: "rect", label: "4. Monitoreo", color: "Info", x: C[5], y: 145, w: 200, h: 90 },
    { key: "gold", shape: "cylinder", label: "Gold", color: "Datos", x: C[1], y: 395, w: 230, h: 110 },
    { key: "cron_s", shape: "circle", label: "Cron cada 30 min", color: "Externo", x: C[2], y: 645, w: 230, h: 110 },
    { key: "scoring", shape: "rect", label: "3. Scoring", color: "Servicio", x: C[4], y: 655, w: 200, h: 90 },
    { key: "pred", shape: "cylinder", label: "gold.predicciones", color: "Éxito", x: C[5] + 30, y: 640, w: 250, h: 110 },
    nota("n_cron", C[0] - 10, 270, 250, 70, "07:30 UTC, o a mano.\nEl primero debe ser tune"),
    nota("n_entrenar", C[1] - 10, 10, 260, 70, "Optuna completo,\nunos 22 minutos"),
    nota("n_gate", C[3] - 30, 10, 290, 70, "MAE baja ≥ 1 % con\nIC95 > 0, o champion degradado"),
    nota("n_monit", C[5] - 10, 270, 250, 70, "Código 3 si el champion\nse degrada"),
    nota("n_scoring", C[3] - 60, 400, 380, 110, "Predice avisos sin predicción\nde la versión, o vistos de nuevo;\ntodos si hay champion nuevo"),
    nota("n_cron_s", C[2] - 10, 790, 250, 70, "Y al terminar el\nentrenamiento diario"),
    nota("n_ci", C[0] - 10, 640, 270, 70, "CI en cada push:\nruff y pytest"),
  ],
  aristas: [
    { from: "cron_e", to: "entrenar", label: "dispara" },
    { from: "gold", to: "entrenar", label: "aptos" },
    { from: "entrenar", to: "mlflow", label: "registra" },
    { from: "mlflow", to: "gate", label: "challenger" },
    { from: "gate", to: "champ", label: "promueve" },
    { from: "champ", to: "monit", label: "mide" },
    { from: "cron_s", to: "scoring", label: "dispara" },
    { from: "gold", to: "scoring", label: "lee avisos", fromSide: "s", toSide: "s" },
    { from: "champ", to: "scoring", label: "versión vigente" },
    { from: "scoring", to: "pred", label: "INSERT" },
  ],
  eventos: [
    { ref: "promueve", name: "Promueve", primitive: "OCCURRENCE", sentence: "{target} promueve al challenger" },
    { ref: "rechaza", name: "Rechaza", primitive: "OCCURRENCE", sentence: "{target} rechaza al challenger" },
    { ref: "mantiene", name: "Se mantiene", primitive: "OCCURRENCE", sentence: "{target} sigue igual" },
  ],
  historias: [
    {
      nombre: "Entrenamiento diario: el challenger gana",
      pasos: [
        { ev: "envia", de: "cron_e", a: "entrenar", espera: 0 },
        { ev: "envia", de: "gold", a: "entrenar", tras: [0] },
        { ev: "procesa", nodo: "entrenar", tras: [0, 1] },
        { ev: "envia", de: "entrenar", a: "mlflow", espera: 800 },
        { ev: "envia", de: "mlflow", a: "gate", tras: [3] },
        { ev: "promueve", nodo: "gate", tras: [4] },
        { ev: "envia", de: "gate", a: "champ", tras: [4] },
        { ev: "envia", de: "champ", a: "monit", tras: [6] },
        { ev: "procesa", nodo: "monit", tras: [7] },
        { ev: "envia", de: "champ", a: "scoring", espera: 600, tras: [6] },
        { ev: "procesa", nodo: "scoring", tras: [9] },
        { ev: "envia", de: "scoring", a: "pred", tras: [9] },
      ],
    },
    {
      nombre: "El gate rechaza al challenger",
      pasos: [
        { ev: "envia", de: "cron_e", a: "entrenar", espera: 0 },
        { ev: "envia", de: "gold", a: "entrenar", tras: [0] },
        { ev: "procesa", nodo: "entrenar", tras: [0, 1] },
        { ev: "envia", de: "entrenar", a: "mlflow", espera: 800 },
        { ev: "envia", de: "mlflow", a: "gate", tras: [3] },
        { ev: "rechaza", nodo: "gate", tras: [4] },
        { ev: "mantiene", nodo: "champ", espera: 600, tras: [4] },
        { ev: "envia", de: "champ", a: "monit", tras: [4] },
        { ev: "procesa", nodo: "monit", tras: [7] },
      ],
    },
    {
      nombre: "Scoring cada 30 minutos",
      pasos: [
        { ev: "envia", de: "cron_s", a: "scoring", espera: 0 },
        { ev: "envia", de: "gold", a: "scoring", tras: [0] },
        { ev: "envia", de: "champ", a: "scoring", tras: [0] },
        { ev: "procesa", nodo: "scoring", tras: [1, 2] },
        { ev: "envia", de: "scoring", a: "pred", espera: 600, tras: [1, 2] },
        { ev: "procesa", nodo: "pred", tras: [4] },
      ],
    },
  ],
});
