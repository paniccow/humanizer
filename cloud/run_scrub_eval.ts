/**
 * Spin up a tiny 4090, upload all 4 eval JSONs + scrub.py + scrub_score_eval.py,
 * score raw + scrubbed outputs with the same RoBERTa detectors used in training,
 * download .scrub-eval.json files, terminate.
 *
 * Cost target: ~$0.20-0.30 for processing 4 files (5-10 min pod time).
 *
 *   bun run run_scrub_eval.ts
 *
 * Env: same RUNPOD_* as launch.ts.
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { provision, terminate } from './provision.ts';
import { PodConnection } from './ssh.ts';

const SCRUB_PY = resolve(import.meta.dir, '..', 'humanizer', 'pipeline', 'scrub.py');
const EVAL_PY = resolve(import.meta.dir, 'scrub_score_eval.py');

const EVAL_FILES_LOCAL = [
  resolve(import.meta.dir, '..', 'experiments', 'run-002-pre', 'run1_eval.json'),
  resolve(import.meta.dir, 'eval.json'),
  resolve(import.meta.dir, 'eval-r3.json'),
  resolve(import.meta.dir, 'eval-r4.json'),
];

const EVAL_FILES_REMOTE = [
  '/workspace/eval_files/run1_eval.json',
  '/workspace/eval_files/run2_eval.json',
  '/workspace/eval_files/run3_eval.json',
  '/workspace/eval_files/run4_eval.json',
];

const PIP_INSTALL =
  'python -m pip install --quiet --upgrade pip wheel && ' +
  'python -m pip install --quiet --upgrade ' +
  '"transformers>=4.45,<5.0" "tokenizers>=0.20" "numpy<2"';

async function main(): Promise<void> {
  const privateKeyPath = process.env.RUNPOD_PRIVATE_KEY;
  if (!privateKeyPath) throw new Error('RUNPOD_PRIVATE_KEY not set');

  // Sanity: verify all eval files exist locally before paying for a pod.
  for (const f of EVAL_FILES_LOCAL) {
    await readFile(f);
  }

  const t0 = Date.now();
  const { pod, hourlyRate } = await provision();
  const conn = new PodConnection(pod, privateKeyPath);
  let success = false;
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

    await conn.run(
      'mkdir -p /workspace/code/scrub_pkg /workspace/eval_files /workspace/output',
    );

    console.log('> uploading scrub.py + eval script');
    // Upload scrub.py as a tiny installable package
    await conn.uploadFile(SCRUB_PY, '/workspace/code/scrub_pkg/scrub.py');
    // Make scrub_pkg a Python package
    await conn.run('touch /workspace/code/scrub_pkg/__init__.py');
    await conn.uploadFile(EVAL_PY, '/workspace/code/scrub_score_eval.py');

    console.log('> uploading 4 eval JSONs');
    for (let i = 0; i < EVAL_FILES_LOCAL.length; i++) {
      await conn.uploadFile(EVAL_FILES_LOCAL[i]!, EVAL_FILES_REMOTE[i]!);
      console.log(`    [${i + 1}/${EVAL_FILES_LOCAL.length}] ${EVAL_FILES_REMOTE[i]}`);
    }

    console.log('> installing minimal Python deps...');
    const install = await conn.run(PIP_INSTALL);
    if (install.code !== 0) throw new Error(`pip install failed:\n${install.stderr.slice(-1500)}`);

    console.log('> running scrub-score eval (this is the GPU work — ~3-5 min)');
    const run = await conn.run('cd /workspace && python /workspace/code/scrub_score_eval.py 2>&1');
    if (run.code !== 0) throw new Error(`eval failed:\n${run.stderr.slice(-1500)}`);

    console.log('> downloading .scrub-eval.json files');
    for (let i = 0; i < EVAL_FILES_REMOTE.length; i++) {
      const remoteOut = EVAL_FILES_REMOTE[i]!.replace(/\.json$/, '.scrub-eval.json');
      const localOut = EVAL_FILES_LOCAL[i]!.replace(/\.json$/, '.scrub-eval.json');
      try {
        const content = await conn.readFile(remoteOut);
        await writeFile(localOut, content);
        console.log(`    -> ${localOut}`);
      } catch (e) {
        console.error(`    failed ${remoteOut}: ${e}`);
      }
    }
    success = true;

    const podSec = (Date.now() - t0) / 1000;
    const cost = hourlyRate ? (hourlyRate * podSec) / 3600 : 0;
    console.log(`\npod uptime ${(podSec / 60).toFixed(1)} min, est cost $${cost.toFixed(2)}`);
  } finally {
    conn.dispose();
    console.log(`> terminating pod ${pod.id}${success ? '' : ' (errors above)'}`);
    await terminate(pod.id);
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    console.error('\nrun_scrub_eval failed:', e instanceof Error ? e.message : e);
    process.exit(1);
  }
}
