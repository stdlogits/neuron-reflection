// Render every Mermaid file (*.mmd) in a folder with pele (https://pele.run) to <name>.svg beside it, in the colours and
// type of the graphics; scripts/graphics.py then frames each one as a PNG for the README.
//
//     node docs/figures/render.mjs [FIGURES_DIR=docs/figures] [OUT_DIR=FIGURES_DIR]
//
// pele must be resolvable from this script's folder: run `npm install pele` from the repository root. Exits 1 and
// names the file and line on a Mermaid syntax error.
import { render } from 'pele';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const THEME = {  // the graphics' page: white, blue for the AI, orange for the cat
  bg: 'transparent', fg: '#0E1117', muted: '#636B7D', line: '#A3A8B4', surface: '#F4F5F7', 'surface-alt': '#EBEDF1',
  border: '#D9DDE3', accent: '#2F6FD6', 'series-1': '#2F6FD6', 'series-2': '#E8781E', 'series-3': '#A3A8B4',
  'series-4': '#1E9EAA', 'series-5': '#E0709E', 'series-6': '#E34958',
};
const FONT = "'Wix Madefor Text', system-ui, sans-serif";
const dir = process.argv[2] || 'docs/figures', out = process.argv[3] || dir;
mkdirSync(out, { recursive: true });
let failed = 0;
for (const f of readdirSync(dir).filter((f) => f.endsWith('.mmd')).sort()) {
  const name = f.replace(/\.mmd$/, '');
  try {
    const { svg } = render(readFileSync(join(dir, f), 'utf8'), { fontFamily: FONT, idPrefix: `${name}-` });
    // The colours go on the root element itself, so the SVG carries them wherever it is drawn.
    const css = Object.entries(THEME).map(([k, v]) => `--pele-${k}:${v};`).join('') + `--pele-font:${FONT};`;
    writeFileSync(join(out, `${name}.svg`), svg.replace('style="', `style="${css}`));
    console.log(name);
  } catch (e) {
    failed++;
    console.error(`${f}: ${e.code || 'error'}${e.line ? ` at line ${e.line}, column ${e.column}` : ''}: ${String(e.message).split('\n')[0]}`);
    if (e.snippet) console.error(e.snippet);
  }
}
process.exit(failed ? 1 : 0);
