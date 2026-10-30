import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

// El ciclo MLOps en una sola fila, pensado para compartir: datos -> variables -> entrenar -> registrar -> gate -> servir,
// con el monitoreo y el reloj debajo. Es la versión corta de ml-automatizacion (sin CI ni limpieza).
const C = [20, 290, 560, 830, 1100, 1370, 1640];

await construir({
  archivo: "ml-recorrido-mlops",
  destino: DESTINO,
  pagina: "Ciclo MLOps del modelo",
  nodos: [
    { key: "gold", shape: "cylinder", label: "1. Datos (Gold)", color: "Datos", x: C[0], y: 205, w: 210, h: 110 },
    { key: "sel", shape: "rect", label: "2. Variables", color: "Servicio", x: C[1], y: 215, w: 195, h: 90 },
    { key: "entrenar", shape: "rect", label: "3. Entrenar", color: "IA", x: C[2], y: 215, w: 195, h: 90 },
    { key: "mlflow", shape: "cylinder", label: "4. MLflow", color: "Almacén", x: C[3], y: 200, w: 195, h: 110 },
    { key: "gate", shape: "diamond", label: "5. Gate", color: "Alerta", x: C[4], y: 200, w: 195, h: 120 },
    { key: "champ", shape: "rect", label: "@champion", color: "Éxito", x: C[5], y: 215, w: 195, h: 90 },
    { key: "scoring", shape: "rect", label: "6. Scoring", color: "Servicio", x: C[6], y: 215, w: 195, h: 90 },
    { key: "cron", shape: "circle", label: "Cron diario", color: "Externo", x: C[1], y: 480, w: 195, h: 110 },
    { key: "monit", shape: "rect", label: "7. Monitoreo", color: "Info", x: C[5], y: 490, w: 195, h: 90 },
    { key: "pred", shape: "cylinder", label: "gold.predicciones", color: "Éxito", x: C[6] - 25, y: 475, w: 245, h: 110 },
    nota("n_gold", C[0] - 10, 80, 215, 66, "Solo avisos aptos,\nsin leer Silver"),
    nota("n_sel", C[1] - 20, 80, 215, 66, "SHAP en 30 rondas\ny poda por correlación"),
    nota("n_entrenar", C[2] - 20, 80, 215, 66, "Optuna: 50 pruebas\npor algoritmo"),
    nota("n_mlflow", C[3] - 20, 80, 215, 66, "Cada run: métricas,\nsplits y modelo"),
    nota("n_gate", C[4] - 20, 80, 215, 66, "Mismas filas, MAE ≥ 1 %\ny mejora sobre ruido"),
    nota("n_champ", C[5] - 20, 80, 215, 66, "El único modelo\nque predice"),
    nota("n_scoring", C[6] - 20, 80, 215, 66, "Cada 30 minutos,\nlo que haya llegado"),
    nota("n_cron", C[1] - 50, 620, 290, 66, "07:30 UTC en GitHub\nActions, todos los días"),
    nota("n_monit", C[5] - 50, 620, 290, 66, "Error con avisos que\nnunca vio; alerta si sube"),
  ],
  aristas: [
    { from: "cron", to: "sel", label: "dispara" },
    { from: "gold", to: "sel", label: "datos" },
    { from: "sel", to: "entrenar", label: "lista" },
    { from: "entrenar", to: "mlflow", label: "registra" },
    { from: "mlflow", to: "gate", label: "nuevo" },
    { from: "gate", to: "champ", label: "sí" },
    { from: "champ", to: "scoring", label: "versión" },
    { from: "champ", to: "monit", label: "mide" },
    { from: "scoring", to: "pred", label: "INSERT" },
  ],
  eventos: [
    { ref: "promueve", name: "Promueve", primitive: "OCCURRENCE", sentence: "{target} promueve al challenger" },
    { ref: "rechaza", name: "Rechaza", primitive: "OCCURRENCE", sentence: "{target} rechaza al challenger" },
    { ref: "mantiene", name: "Se mantiene", primitive: "OCCURRENCE", sentence: "{target} sigue igual" },
  ],
  historias: [
    {
      nombre: "Un día de MLOps: el challenger gana",
      pasos: [
        { ev: "envia", de: "cron", a: "sel", espera: 0 },
        { ev: "envia", de: "gold", a: "sel", tras: [0] },
        { ev: "procesa", nodo: "sel", tras: [0, 1] },
        { ev: "envia", de: "sel", a: "entrenar", espera: 600, tras: [1] },
        { ev: "procesa", nodo: "entrenar", tras: [3] },
        { ev: "envia", de: "entrenar", a: "mlflow", espera: 800, tras: [3] },
        { ev: "envia", de: "mlflow", a: "gate", tras: [5] },
        { ev: "promueve", nodo: "gate", tras: [6] },
        { ev: "envia", de: "gate", a: "champ", tras: [6] },
        { ev: "envia", de: "champ", a: "monit", tras: [8] },
        { ev: "procesa", nodo: "monit", tras: [9] },
        { ev: "envia", de: "champ", a: "scoring", espera: 600, tras: [8] },
        { ev: "procesa", nodo: "scoring", tras: [11] },
        { ev: "envia", de: "scoring", a: "pred", tras: [11] },
      ],
    },
    {
      nombre: "El gate rechaza al challenger",
      pasos: [
        { ev: "envia", de: "cron", a: "sel", espera: 0 },
        { ev: "envia", de: "gold", a: "sel", tras: [0] },
        { ev: "procesa", nodo: "sel", tras: [0, 1] },
        { ev: "envia", de: "sel", a: "entrenar", espera: 600, tras: [1] },
        { ev: "procesa", nodo: "entrenar", tras: [3] },
        { ev: "envia", de: "entrenar", a: "mlflow", espera: 800, tras: [3] },
        { ev: "envia", de: "mlflow", a: "gate", tras: [5] },
        { ev: "rechaza", nodo: "gate", tras: [6] },
        { ev: "mantiene", nodo: "champ", espera: 600, tras: [6] },
        { ev: "envia", de: "champ", a: "scoring", tras: [6] },
        { ev: "procesa", nodo: "scoring", tras: [9] },
        { ev: "envia", de: "scoring", a: "pred", tras: [9] },
      ],
    },
  ],
});
