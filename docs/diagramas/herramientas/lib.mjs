// Constructor declarativo de diagramas Fluyo con Historias (usa el servidor MCP local por stdio).
// Un diagrama = nodos + aristas + eventos + Historias; sale como .fluyo.json (ábrelo en fluyo.space) y .svg estático.
// Solo nombres de componentes: nunca hosts, cadenas de conexión ni claves.
import fs from "node:fs";
import path from "node:path";
import { abrir } from "./cliente.mjs";

const json = (r) => {
  const t = r.textos.find((x) => x.trim().startsWith("{"));
  return t ? JSON.parse(t) : null;
};

/** Nota: recuadro punteado sin relleno, con saltos de línea manuales (el texto no se ajusta solo). */
export const nota = (key, x, y, w, h, label) => ({
  key, shape: "rect", label, x, y, w: Math.round(w * 1.15), h: Math.round(h * 1.1), fs: 17, fill: "none", border: "dashed",
  color: "Externo",
});

const EVENTOS_BASE = [
  { ref: "envia", name: "Envía", primitive: "FLOW", sentence: "{source} envía a {target}", motion: "normal" },
  { ref: "procesa", name: "Procesa", primitive: "OCCURRENCE", sentence: "{target} procesa" },
];

/** El SVG de Fluyo sale con fondo transparente: sobre el modo oscuro de GitHub el texto oscuro desaparece. Se le agrega
 *  un rectángulo blanco del tamaño de su viewBox, justo después de la etiqueta de apertura. */
export function conFondo(svg, color = "#ffffff") {
  const abre = svg.match(/<svg[^>]*>/);
  const vb = abre?.[0].match(/viewBox="([^"]+)"/)?.[1].trim().split(/\s+/).map(Number);
  if (!abre || !vb || vb.length !== 4) return svg;
  const [x, y, w, h] = vb;
  const rect = `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${color}"/>`;
  return svg.replace(abre[0], abre[0] + "\n" + rect);
}

export async function construir(spec) {
  const { llamar, cerrar } = await abrir();
  const falla = async (msg, extra = "") => {
    console.log(`[${spec.archivo}] ERROR: ${msg}\n${extra}`.slice(0, 2500));
    await cerrar();
    process.exit(1);
  };

  const base = await llamar("create_diagram", {
    pageName: spec.pagina,
    theme: "claro",
    customBg: "#ffffff",
    // Letras grandes: el README muestra el SVG a menos de la mitad de su tamaño, así que 14 px no se leen.
    nodes: spec.nodos.map((n) => ({ fs: 22, ...n })),
    edges: spec.aristas.map((a) => ({ route: "ortho", fs: 18, ...a })),
  });
  if (base.isError) await falla("create_diagram", base.textos.join("\n"));
  const doc0 = json(base) ?? JSON.parse(base.textos[1]);
  const d0 = json(await llamar("describe_document", { document: doc0 }));
  const pag = d0.pages[0];

  const etiqueta = Object.fromEntries(spec.nodos.map((n) => [n.key, n.label]));
  const idNodo = (key) => {
    const norm = (t) => String(t).replace(/\s+/g, " ").trim();  // las etiquetas con saltos de línea vuelven con espacios
    // Por etiqueta; si el servidor la devuelve distinta (por ejemplo, con saltos de línea), por orden de creación.
    const id = pag.nodes.find((n) => norm(n.label) === norm(etiqueta[key]))?.id
      ?? pag.nodes[spec.nodos.findIndex((n) => n.key === key)]?.id;
    if (id === undefined) throw new Error(`nodo desconocido: ${key}`);
    return id;
  };
  const idArista = (de, a) => {
    const id = pag.connections.find((e) => e.from === idNodo(de) && e.to === idNodo(a))?.id;
    if (id === undefined) throw new Error(`no hay arista ${de} -> ${a}`);
    return id;
  };

  const eventos = [...EVENTOS_BASE, ...(spec.eventos ?? [])];
  const ops = eventos.map((e) => ({ op: "create_event_type", scope: "eventType", ...e }));
  (spec.historias ?? []).forEach((h, si) => {
    const sref = `h${si}`;
    ops.push({ op: "create_story", scope: "story", pageIndex: 0, ref: sref, name: h.nombre });
    h.pasos.forEach((p, i) => {
      const target = p.a ? { edgeId: idArista(p.de, p.a) } : { nodeId: idNodo(p.nodo) };
      ops.push({
        op: "add_step", scope: "story", pageIndex: 0, storyId: { ref: sref }, ref: `${sref}_${i}`,
        eventTypeId: { ref: p.ev }, target, waitMs: p.espera ?? 300,
        ...(p.tras?.length ? { waitsForArrivalOf: p.tras.map((k) => ({ ref: `${sref}_${k}` })) } : {}),
      });
    });
  });

  const r = await llamar("author_document", { document: doc0, baseRevision: d0.revision, operations: ops });
  const res = json(r);
  if (r.isError || !res?.document) await falla("author_document", r.textos.join("\n"));

  fs.mkdirSync(spec.destino, { recursive: true });
  fs.writeFileSync(path.join(spec.destino, `${spec.archivo}.fluyo.json`), JSON.stringify(res.document, null, 1));

  const d1 = json(await llamar("describe_document", { document: res.document }));
  for (const s of d1.pages[0].stories ?? []) {
    const run = await llamar("run_story", { document: res.document, pageIndex: 0, storyId: s.storyId });
    console.log(`[${spec.archivo}] ${run.isError ? "ERROR" : "ok"} | ${s.name} | ${(run.textos[0] ?? "").slice(0, 90)}`);
  }
  const ex = await llamar("export_diagram", { document: res.document, pageIndex: 0, crop: true });
  const svg = ex.textos.find((t) => t.trim().startsWith("<svg")) ?? ex.textos[ex.textos.length - 1];
  fs.writeFileSync(path.join(spec.destino, `${spec.archivo}.svg`), conFondo(svg));
  await cerrar();
}
