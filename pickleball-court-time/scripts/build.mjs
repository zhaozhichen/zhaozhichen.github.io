import {build} from 'esbuild';
import {fileURLToPath} from 'node:url';
import {dirname,resolve} from 'node:path';
process.chdir(resolve(dirname(fileURLToPath(import.meta.url)),'..'));
await build({absWorkingDir:process.cwd(),entryPoints:['src/main.tsx'],bundle:true,minify:true,format:'esm',target:['es2022'],outfile:'assets/app.js',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},legalComments:'linked',sourcemap:false});
console.log('Built static GitHub Pages assets.');
