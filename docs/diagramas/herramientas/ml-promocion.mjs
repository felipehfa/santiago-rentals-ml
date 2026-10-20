import { fileURLToPath } from "node:url";

import { construir, nota } from "./lib.mjs";

const DESTINO = fileURLToPath(new URL("..", import.meta.url));  // docs/diagramas/

await construir({
  archivo: "ml-promocion",
  destino: DESTINO,
  pagina: "Promoción de modelos",
  nodos: [
    { key: "challenger", shape: "rect", label: "@challenger", color: "IA", x: 40, y: 200, w: 200, h: 70 },
    { key: "champion", shape: "rect", label: "@champion actual", color: "Éxito", x: 40, y: 500, w: 200, h: 70 },
    { key: "filas", shape: "rect", label: "Mismas filas", color: "Servicio", x: 400, y: 350, w: 200, h: 70 },
    { key: "metricas", shape: "rect", label: "MAE, MAPE e IC95", color: "Info", x: 760, y: 350, w: 200, h: 70 },
    { key: "gate", shape: "diamond", label: "¿Promueve?", color: "Alerta", x: 1120, y: 330, w: 220, h: 110 },
    { key: "promovido", shape: "rect", label: "@champion nuevo", color: "Éxito", x: 1520, y: 200, w: 200, h: 70 },
    { key: "rechazado", shape: "rect", label: "Sigue el champion", color: "Externo", x: 1520, y: 500, w: 200, h: 70 },
    { key: "scoring", shape: "rect", label: "Scoring repite todo", color: "Servicio", x: 1880, y: 200, w: 200, h: 70 },
    nota("n_filas", 350, 480, 330, 98, "Test del challenger sin los avisos que el\nchampion usó para entrenar o validar: ninguno\nlos vio. Se exigen al menos 100 avisos."),
    nota("n_metricas", 710, 190, 330, 98, "MAE y MAPE de cada modelo y la diferencia\nde MAE con su intervalo de confianza al 95 %\n(bootstrap sobre los mismos avisos)."),
    nota("n_criterios", 1060, 520, 380, 138, "Promueve si:\n1. el MAE baja ≥ 1 % y el IC95 de la mejora\n    es > 0, o\n2. el champion se degradó > 5 % y el\n    challenger no es peor. Sin champion: promueve."),
    nota("n_scoring", 1840, 330, 310, 78, "Con una versión nueva, todos los avisos\nse vuelven a predecir con ella."),
  ],
  aristas: [
    { from: "challenger", to: "filas", label: "evalúa" },
    { from: "champion", to: "filas", label: "evalúa" },
    { from: "filas", to: "metricas", label: "predicciones" },
    { from: "metricas", to: "gate", label: "compara" },
    { from: "gate", to: "promovido", label: "sí" },
    { from: "gate", to: "rechazado", label: "no" },
    { from: "promovido", to: "scoring", label: "versión nueva" },
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
