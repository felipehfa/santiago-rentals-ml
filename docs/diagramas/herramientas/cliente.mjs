// Cliente mínimo del servidor MCP de Fluyo (stdio). El servidor se instala aparte (ver LEEME.md) y se indica con la
// variable FLUYO_MCP; por defecto se busca en prototipos/fluyo/fluyo-mcp, junto a los repos.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const AQUI = path.dirname(fileURLToPath(import.meta.url));
const RAIZ_REPO = path.resolve(AQUI, "..", "..", "..");
const FLUYO_MCP = process.env.FLUYO_MCP ?? path.resolve(RAIZ_REPO, "..", "prototipos", "fluyo", "fluyo-mcp");

if (!fs.existsSync(path.join(FLUYO_MCP, "dist", "index.js"))) {
  console.error(`No encuentro el servidor de Fluyo en ${FLUYO_MCP}. Instálalo como dice LEEME.md o define FLUYO_MCP.`);
  process.exit(1);
}
const sdk = (ruta) => import(pathToFileURL(path.join(FLUYO_MCP, "node_modules", "@modelcontextprotocol", "sdk", "dist", "esm", ruta)).href);
const { Client } = await sdk("client/index.js");
const { StdioClientTransport } = await sdk("client/stdio.js");

export async function abrir() {
  const transport = new StdioClientTransport({ command: process.execPath, args: [path.join(FLUYO_MCP, "dist", "index.js")] });
  const client = new Client({ name: "santiago-rentals-diagramas", version: "1.0.0" });
  await client.connect(transport);
  const llamar = async (name, args = {}) => {
    const r = await client.callTool({ name, arguments: args });
    const textos = (r.content ?? []).filter((c) => c.type === "text").map((c) => c.text);
    return { isError: !!r.isError, textos };
  };
  return { client, llamar, cerrar: () => client.close() };
}
