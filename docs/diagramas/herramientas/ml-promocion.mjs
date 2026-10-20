import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

await construir({
  archivo: "ml-promocion",
  destino: DESTINO,
  pagina: "Promoción de modelos",
  nodos: [
    { key: "challenger", shape: "rect", label: "@challenger", color: "IA", x: 20, y: 100, w: 230, h: 90 },
    { key: "champion", shape: "rect", label: "@champion", color: "Éxito", x: 20, y: 390, w: 230, h: 90 },
    { key: "filas", shape: "rect", label: "Mismas filas", color: "Servicio", x: 360, y: 245, w: 230, h: 90 },
    { key: "metricas", shape: "rect", label: "MAE, MAPE, IC95", color: "Info", x: 700, y: 245, w: 250, h: 90 },
    { key: "gate", shape: "diamond", label: "¿Promueve?", color: "Alerta", x: 1060, y: 220, w: 250, h: 140 },
    { key: "promovido", shape: "rect", label: "@champion nuevo", color: "Éxito", x: 1430, y: 100, w: 250, h: 90 },
    { key: "rechazado", shape: "rect", label: "Sigue el champion", color: "Externo", x: 1430, y: 390, w: 250, h: 90 },
    { key: "scoring", shape: "rect", label: "Scoring repite todo", color: "Servicio", x: 1430, y: -80, w: 250, h: 90 },
    nota("n_filas", 330, 420, 300, 88, "Test del challenger sin\nlo que vio el champion\n(mínimo 100 avisos)"),
    nota("n_metricas", 690, 100, 280, 66, "IC95 de la mejora de MAE\npor bootstrap"),
    nota("n_criterios", 1010, 420, 360, 110, "1. MAE baja ≥ 1 % y IC95 > 0\n2. o el champion se degradó\n   más de 5 % y el challenger\n   no es peor"),
  ],
  aristas: [
    { from: "challenger", to: "filas", label: "evalúa" },
    { from: "champion", to: "filas", label: "evalúa" },
    { from: "filas", to: "metricas", label: "predice" },
    { from: "metricas", to: "gate", label: "compara" },
    { from: "gate", to: "promovido", label: "sí" },
    { from: "gate", to: "rechazado", label: "no" },
    { from: "promovido", to: "scoring", label: "versión nueva", fromSide: "n", toSide: "s" },
  ],
  eventos: [
    { ref: "promueve", name: "Promueve", primitive: "OCCURRENCE", sentence: "{target} promueve al challenger" },
    { ref: "rechaza", name: "Rechaza", primitive: "OCCURRENCE", sentence: "{target} rechaza al challenger" },
    { ref: "degradado", name: "Se degradó", primitive: "OCCURRENCE", sentence: "{target} se degradó más de 5 %" },
  ],
  historias: [
    {
      nombre: "Mejora significativa: se promueve",
      pasos: [
        { ev: "envia", de: "challenger", a: "filas", espera: 0 },
        { ev: "envia", de: "champion", a: "filas", espera: 200 },
        { ev: "procesa", nodo: "filas", tras: [0, 1] },
        { ev: "envia", de: "filas", a: "metricas", espera: 700, tras: [0, 1] },
        { ev: "envia", de: "metricas", a: "gate", tras: [3] },
        { ev: "promueve", nodo: "gate", tras: [4] },
        { ev: "envia", de: "gate", a: "promovido", tras: [4] },
        { ev: "envia", de: "promovido", a: "scoring", tras: [6] },
      ],
    },
    {
      nombre: "Empate: el gate rechaza al challenger",
      pasos: [
        { ev: "envia", de: "challenger", a: "filas", espera: 0 },
        { ev: "envia", de: "champion", a: "filas", espera: 200 },
        { ev: "procesa", nodo: "filas", tras: [0, 1] },
        { ev: "envia", de: "filas", a: "metricas", espera: 700, tras: [0, 1] },
        { ev: "envia", de: "metricas", a: "gate", tras: [3] },
        { ev: "rechaza", nodo: "gate", tras: [4] },
        { ev: "envia", de: "gate", a: "rechazado", tras: [4] },
      ],
    },
    {
      nombre: "El champion se degradó: entra el challenger",
      pasos: [
        { ev: "degradado", nodo: "champion", espera: 0 },
        { ev: "envia", de: "challenger", a: "filas", espera: 400 },
        { ev: "envia", de: "champion", a: "filas", espera: 200 },
        { ev: "envia", de: "filas", a: "metricas", espera: 700, tras: [1, 2] },
        { ev: "envia", de: "metricas", a: "gate", tras: [3] },
        { ev: "promueve", nodo: "gate", tras: [4] },
        { ev: "envia", de: "gate", a: "promovido", tras: [4] },
      ],
    },
  ],
});
