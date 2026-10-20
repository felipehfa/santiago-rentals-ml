import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

await construir({
  archivo: "ml-automatizacion",
  destino: DESTINO,
  pagina: "ML automático en GitHub",
  nodos: [
    { key: "cron_e", shape: "circle", label: "Cron diario 07:30 UTC", color: "Externo", x: 40, y: 200, w: 190, h: 100 },
    { key: "entrenar", shape: "rect", label: "1. Entrenar (Optuna)", color: "IA", x: 380, y: 215, w: 200, h: 70 },
    { key: "mlflow", shape: "cylinder", label: "MLflow (Postgres + bucket)", color: "Almacén", x: 800, y: 195, w: 210, h: 110 },
    { key: "gate", shape: "diamond", label: "2. Gate", color: "Alerta", x: 1250, y: 190, w: 220, h: 120 },
    { key: "champ", shape: "rect", label: "@champion", color: "Éxito", x: 1700, y: 215, w: 190, h: 70 },
    { key: "monit", shape: "rect", label: "4. Monitoreo", color: "Info", x: 2100, y: 215, w: 190, h: 70 },
    { key: "gold", shape: "cylinder", label: "Gold (Supabase)", color: "Datos", x: 380, y: 520, w: 200, h: 100 },
    { key: "cron_s", shape: "circle", label: "Cron cada 3 h", color: "Externo", x: 800, y: 760, w: 170, h: 100 },
    { key: "scoring", shape: "rect", label: "3. Scoring", color: "Servicio", x: 1700, y: 775, w: 190, h: 70 },
    { key: "pred", shape: "cylinder", label: "gold.predicciones", color: "Éxito", x: 2100, y: 755, w: 200, h: 110 },
    nota("n_cron", 40, 330, 250, 60, "Manual: Run workflow.\nEl primero debe ser tune."),
    nota("n_entrenar", 330, 10, 380, 115, "Datos: avisos aptos de 12 comunas.\nTrain: publicados en los últimos 60 días.\nVariables + Optuna: 50 pruebas por\nalgoritmo (XGBoost, LightGBM);\nelige por CV MAE."),
    nota("n_gate", 1100, 10, 520, 100, "Promueve si:\n• el MAE baja ≥ 1 % y el IC95 de la mejora es > 0, o\n• el champion se degradó > 5 % y el challenger no es peor.\nSin champion: promueve."),
    nota("n_monit", 2060, 330, 290, 75, "Código 3 si el champion se degrada\ncon avisos nuevos.\nNo detiene el flujo."),
    nota("n_scoring", 1000, 540, 580, 140, "Predice los avisos que:\n• no tienen predicción de esta versión, o\n• se visitaron después de su última predicción.\nSi hay champion nuevo, repite todos.\nGuarda etiqueta de mercado, confianza y banda\nde precio. Sin las comunas fuera del modelo."),
    nota("n_cron_s", 770, 690, 280, 50, "También al terminar el\nentrenamiento diario."),
    nota("n_ci", 40, 700, 250, 60, "CI en cada push:\nruff y pytest (sin secrets)."),
  ],
  aristas: [
    { from: "cron_e", to: "entrenar", label: "dispara" },
    { from: "gold", to: "entrenar", label: "lee aptos" },
    { from: "entrenar", to: "mlflow", label: "registra @challenger" },
    { from: "mlflow", to: "gate", label: "challenger vs champion" },
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
      nombre: "Scoring cada 3 horas",
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
