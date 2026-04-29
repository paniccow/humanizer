/**
 * Run held-out detector eval on one or more eval.json files. Spins up a
 * temporary 4090 pod (or A4000 — flag the cheapest available), uploads
 * eval.json + heldout_eval.py, runs, downloads .holdout.json files,
 * terminates.
 *
 *   bun run run_heldout.ts ./eval.json ./eval-r3.json
 *
 * Cost: ~$0.10 for two evals (only ~3-5 minutes of pod time on a 4090).
 *
 * Env vars: same as launch.ts (RUNPOD_API_KEY, RUNPOD_PUBLIC_KEY,
 * RUNPOD_PRIVATE_KEY).
 */
import { readFile, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';

import { provision, terminate } from './provision.ts';
import { PodConnection } from './ssh.ts';

const HELDOUT_PY = resolve(import.meta.dir, 'heldout_eval.py');

// torch>=2.6 needed for some detector .bin files (Hello-SimpleAI).
// transformers>=4.45 + tokenizers>=0.20 for modern AutoModel resolution.
const PIP_HELDOUT =
  'python -m pip install --quiet --upgrade pip wheel && ' +
  'python -m pip install --quiet --upgrade ' +
  '"torch>=2.6" "transformers>=4.45" "tokenizers>=0.20" "numpy<2"';

async function main(): Promise<void> {
  const inputs = process.argv.slice(2);
  if (inputs.length === 0) {
    console.error('usage: bun run run_heldout.ts <eval.json> [<eval2.json> ...]');
    process.exit(1);
  }
  const privateKeyPath = process.env.RUNPOD_PRIVATE_KEY;
  if (!privateKeyPath) throw new Error('RUNPOD_PRIVATE_KEY not set');

  // Sanity: every input must exist locally before we spin up a pod.
  for (const p of inputs) {
    await readFile(p); // throws if missing
  }

  console.log(`> ${inputs.length} eval file(s) to process`);
  const { pod } = await provision();
  let success = false;
  const conn = new PodConnection(pod, privateKeyPath);
  try {
    let lastErr: unknown;
    for (let i = 0; i < 10; i++) {
      try {
        await conn.connect();
        lastErr = undefined;
        break;
      } catch (e) {
        lastErr = e;
        await new Promise((r) => setTimeout(r, 5_000));
      }
    }
    if (lastErr) throw lastErr;
    console.log('  connected.');

    await conn.run('mkdir -p /workspace/code /workspace/output');
    await conn.uploadFile(HELDOUT_PY, '/workspace/code/heldout_eval.py');

    console.log('> installing minimal deps (just transformers + torch already on the image)...');
    const install = await conn.run(PIP_HELDOUT);
    if (install.code !== 0) throw new Error(`pip install failed: ${install.stderr.slice(-1500)}`);

    for (const localPath of inputs) {
      const baseName = basename(localPath);
      const remotePath = `/workspace/output/${baseName}`;
      console.log(`\n=== held-out for ${localPath} ===`);
      await conn.uploadFile(localPath, remotePath);
      const out = await conn.run(`python /workspace/code/heldout_eval.py ${remotePath}`);
      if (out.code !== 0) {
        console.error(`  FAILED on ${localPath} (code ${out.code}):\n${out.stderr.slice(-1000)}`);
        continue;
      }
      const holdoutRemote = remotePath.replace(/\.json$/, '.holdout.json');
      const holdoutContent = await conn.readFile(holdoutRemote);
      const holdoutLocal = localPath.replace(/\.json$/, '.holdout.json');
      await writeFile(holdoutLocal, holdoutContent);
      console.log(`  -> ${holdoutLocal}`);
    }
    success = true;
  } finally {
    conn.dispose();
    console.log(`\n> terminating pod ${pod.id}${success ? '' : ' (errors above)'}`);
    await terminate(pod.id);
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    console.error('\nrun_heldout failed:', e instanceof Error ? e.message : e);
    process.exit(1);
  }
}
