# Herramientas de los diagramas (Fluyo)

Los diagramas de `docs/diagramas/` se generan con estos scripts y el servidor MCP de [Fluyo](https://github.com/itsnect/fluyo-mcp)
(MIT). Cada script deja en `docs/diagramas/` un `.fluyo.json` (ábrelo con el botón **Abrir** de https://fluyo.space para ver las
Historias animadas y exportar el GIF) y un `.svg` (vista estática que usa el README).

## Instalar el servidor (una vez)

```bash
git clone https://github.com/itsnect/fluyo-mcp
cd fluyo-mcp
git checkout efb316f          # el commit con que se generaron estos diagramas
npm ci --ignore-scripts
npm run build
```

Requiere Node 22.18 o superior. Por defecto los scripts buscan el servidor en `../prototipos/fluyo/fluyo-mcp` (junto a los repos);
en otro lugar, define `FLUYO_MCP=<ruta a fluyo-mcp>`.

## Regenerar

```bash
node docs/diagramas/herramientas/<nombre>.mjs
```

`lib.mjs` es el constructor (nodos, aristas, eventos e Historias); cada `<nombre>.mjs` describe un diagrama. Para revisar el
resultado: `captura.ps1 -svg <archivo.svg> -png <salida.png>` (Windows, usa Edge). Los diagramas solo llevan nombres de
componentes, nunca hosts, cadenas de conexión ni claves. Si cambia la arquitectura, edita el script, regenera y sube el
`.fluyo.json` y el `.svg` en el mismo commit que el README.

## Legibilidad

GitHub muestra el SVG a menos de la mitad de su tamaño, así que los diagramas se arman compactos: letra de 22 en los nodos, 18 en
las flechas y 17 en las notas (`lib.mjs`), un ancho total de unos 1.300 a 1.800 px, cajas grandes y juntas, y notas de dos o tres
líneas cortas. El detalle largo va en el README, no en el diagrama. Al editar un script, revisa el resultado a un ancho de unos
1.000 px (`captura.ps1`) y comprueba que ninguna nota ni etiqueta se pise con una caja.
