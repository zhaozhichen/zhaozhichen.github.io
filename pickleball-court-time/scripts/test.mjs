import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dir = await mkdtemp(join(tmpdir(), 'court-time-tests-'));
try {
  const outfile = join(dir, 'gotham.test.mjs');
  await build({ absWorkingDir: root, entryPoints: ['tests/gotham.test.ts'], outfile, bundle: true, platform: 'node', format: 'esm' });
  execFileSync(process.execPath, ['--test', outfile], { stdio: 'inherit' });
} finally {
  await rm(dir, { recursive: true, force: true });
}
