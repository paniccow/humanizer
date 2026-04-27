/**
 * End-to-end orchestrator: provision a 4090 pod, upload the training script,
 * run GRPO, download the trained adapter + eval, terminate.
 *
 *   bun run launch.ts
 *
 * Env vars (required):
 *   RUNPOD_API_KEY      — from https://www.runpod.io/console/user/settings
 *   RUNPOD_PUBLIC_KEY   — contents of ~/.ssh/id_ed25519.pub
 *   RUNPOD_PRIVATE_KEY  — path to ~/.ssh/id_ed25519
 *
 * Env vars (optional):
 *   RUNPOD_GPU             — default "NVIDIA GeForce RTX 4090"
 *   HUMANIZER_TRAIN_N      — default 600 prompts
 *   HUMANIZER_G            — default 8 generations per prompt
 *   HUMANIZER_LR           — default 5e-6
 *   HUMANIZER_BETA         — default 0.001 (KL penalty)
 *   ADAPTER_OUT            — local download dir (default ./adapter)
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import { provision, terminate } from './provision.ts';
import { PodConnection } from './ssh.ts';
import type { TrainResult } from './types.ts';

const TRAIN_PY = resolve(import.meta.dir, 'train.py');
const ADAPTER_OUT = resolve(process.cwd(), process.env.ADAPTER_OUT ?? 'adapter');
const EVAL_OUT = resolve(process.cwd(), 'eval.json');

async function main(): Promise<TrainResult> {
  const privateKeyPath = process.env.RUNPOD_PRIVATE_KEY;
  if (!privateKeyPath) {
    throw new Error('RUNPOD_PRIVATE_KEY not set. Should be the path to ~/.ssh/id_ed25519 (no .pub).');
  }

  const t0 = Date.now();
  const { pod, hourlyRate } = await provision();

  const conn = new PodConnection(pod, privateKeyPath);
  let trained = false;
  try {
    console.log('> connecting via SSH...');
    // Pod SSHD takes a moment after the runtime is "ready"; retry a few times.
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

    console.log('> uploading train.py');
    await conn.run('mkdir -p /workspace/code /workspace/output');
    await conn.uploadFile(TRAIN_PY, '/workspace/code/train.py');

    console.log('> installing python deps (one-time)...');
    // Force-upgrade — runpod/pytorch image ships with older trl/transformers
    // that don't have GRPOConfig.
    const PIP =
      'python -m pip install --quiet --upgrade pip wheel && ' +
      // No TRL — train.py uses hand-rolled REINFORCE+KL (avoids version hell).
      'python -m pip install --quiet --upgrade ' +
      '"transformers>=4.45" "peft>=0.13" "accelerate>=1.0" ' +
      '"datasets>=3.0" "sentence-transformers>=3.0" "bitsandbytes>=0.44" "numpy<2"';
    const install = await conn.run(PIP);
    if (install.code !== 0) throw new Error(`pip install failed (code ${install.code})`);
    // Quick sanity: confirm the bare essentials import + GPU is visible.
    const sanity = await conn.run(
      `python -c "
import torch, transformers, peft, datasets
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('transformers', transformers.__version__, 'peft', peft.__version__, 'datasets', datasets.__version__)
"`,
    );
    if (sanity.code !== 0) throw new Error(`sanity import failed:\n${sanity.stderr}`);

    console.log('> launching training (this is the long part — sit back)...');
    const train = await conn.run('cd /workspace && python -u code/train.py 2>&1 | tee output/run.log');
    if (train.code !== 0) {
      throw new Error(`train.py exited ${train.code}. Tail:\n${train.stderr.slice(-1500)}`);
    }
    trained = true;

    console.log(`> downloading adapter -> ${ADAPTER_OUT}`);
    await conn.downloadDir('/workspace/output/adapter', ADAPTER_OUT);

    console.log(`> downloading eval -> ${EVAL_OUT}`);
    const evalJson = await conn.readFile('/workspace/output/eval.json');
    await mkdir(dirname(EVAL_OUT), { recursive: true });
    await writeFile(EVAL_OUT, evalJson);

    const podSeconds = (Date.now() - t0) / 1000;
    const cost = hourlyRate ? (hourlyRate * podSeconds) / 3600 : 0;

    const evalParsed = JSON.parse(evalJson) as { summary: TrainResult['eval'] };
    console.log('\n=== summary ===');
    console.log(JSON.stringify(evalParsed.summary, null, 2));
    console.log(`\npod uptime: ${(podSeconds / 60).toFixed(1)} min`);
    console.log(`estimated cost: $${cost.toFixed(2)}`);

    return {
      adapterPath: ADAPTER_OUT,
      eval: evalParsed.summary,
      podSeconds,
      estimatedCostUsd: cost,
    };
  } finally {
    conn.dispose();
    console.log(`> terminating pod ${pod.id}${trained ? '' : ' (training failed — kept logs locally if downloaded)'}`);
    await terminate(pod.id);
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    console.error('\nlaunch failed:', e instanceof Error ? e.message : e);
    process.exit(1);
  }
}
