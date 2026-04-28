/**
 * End-to-end orchestrator: provision a 4090 pod, upload the training script,
 * run GRPO, download the trained adapter + eval, terminate.
 *
 *   bun run launch.ts                       # full pipeline (default)
 *   bun run launch.ts -- --resume <pod-id>  # reconnect to an in-flight pod
 *
 * Disconnect-safe: training is launched with `nohup setsid` on the pod, so
 * killing this script (or losing SSH for any reason) leaves the training
 * running. Reconnect later with --resume <pod-id> to download artifacts and
 * terminate the pod.
 *
 * Env vars (required):
 *   RUNPOD_API_KEY      — from https://www.runpod.io/console/user/settings
 *   RUNPOD_PUBLIC_KEY   — contents of ~/.ssh/id_ed25519.pub
 *   RUNPOD_PRIVATE_KEY  — path to ~/.ssh/id_ed25519
 *
 * Env vars (optional):
 *   TRAIN_SCRIPT        — default 'train.py'; pass 'train_v2.py' for run #2
 *   RUNPOD_GPU          — default "NVIDIA GeForce RTX 4090"
 *   ADAPTER_OUT         — local download dir (default ./adapter)
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import { provision, terminate } from './provision.ts';
import { RunPodClient } from './runpod.ts';
import { PodConnection } from './ssh.ts';
import type { PodInfo, TrainResult, RunPodConfig } from './types.ts';

const TRAIN_PY = resolve(import.meta.dir, process.env.TRAIN_SCRIPT ?? 'train.py');
const ADAPTER_OUT = resolve(process.cwd(), process.env.ADAPTER_OUT ?? 'adapter');
const EVAL_OUT = resolve(process.cwd(), process.env.EVAL_OUT ?? 'eval.json');
const POLL_SECONDS = Number(process.env.POLL_SECONDS ?? 30);

function loadCfg(): RunPodConfig {
  const apiKey = process.env.RUNPOD_API_KEY;
  if (!apiKey) throw new Error('RUNPOD_API_KEY not set.');
  const publicSshKey = process.env.RUNPOD_PUBLIC_KEY;
  if (!publicSshKey) throw new Error('RUNPOD_PUBLIC_KEY not set.');
  return {
    apiKey,
    publicSshKey,
    gpuTypeId: process.env.RUNPOD_GPU ?? 'NVIDIA GeForce RTX 4090',
    containerDiskInGb: Number(process.env.RUNPOD_CONTAINER_GB ?? 25),
    volumeInGb: Number(process.env.RUNPOD_VOLUME_GB ?? 15),
    cloudType: (process.env.RUNPOD_CLOUD_TYPE as 'COMMUNITY' | 'SECURE' | 'ALL' | undefined) ?? 'ALL',
  };
}

async function connectWithRetry(pod: PodInfo, privateKeyPath: string): Promise<PodConnection> {
  const conn = new PodConnection(pod, privateKeyPath);
  let lastErr: unknown;
  for (let i = 0; i < 12; i++) {
    try {
      await conn.connect();
      return conn;
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 5_000));
    }
  }
  throw lastErr;
}

const PIP_INSTALL =
  'python -m pip install --quiet --upgrade pip wheel && ' +
  'python -m pip install --quiet --upgrade ' +
  '"transformers>=4.45" "peft>=0.13" "accelerate>=1.0" ' +
  '"datasets>=3.0" "sentence-transformers>=3.0" "bitsandbytes>=0.44" "numpy<2"';

const SANITY = `python -c "
import torch, transformers, peft, datasets
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('transformers', transformers.__version__, 'peft', peft.__version__, 'datasets', datasets.__version__)
"`;

/**
 * Kick off training on the pod under nohup setsid. Survives SSH disconnect,
 * Mac sleep, this process being killed. Writes /workspace/output/done on
 * successful completion.
 */
async function kickOffTraining(conn: PodConnection): Promise<void> {
  console.log('> uploading train script...');
  await conn.run('mkdir -p /workspace/code /workspace/output');
  await conn.uploadFile(TRAIN_PY, '/workspace/code/train.py');

  console.log('> installing python deps (one-time)...');
  const install = await conn.run(PIP_INSTALL);
  if (install.code !== 0) throw new Error(`pip install failed (code ${install.code})`);
  const sanity = await conn.run(SANITY);
  if (sanity.code !== 0) throw new Error(`sanity import failed:\n${sanity.stderr}`);

  console.log('> launching training under nohup setsid (disconnect-safe)...');
  // Forward selected env vars to the pod's training process. PANGRAM_API_KEY
  // is required by train_v6 (Pangram in the reward loop). Future vars like
  // ORIGINALITY_API_KEY would land here too. Any value with single-quotes
  // would break the wrapping; we don't expect that for API keys (UUID-like).
  const forwardedEnv: string[] = [];
  for (const k of ['PANGRAM_API_KEY', 'ORIGINALITY_API_KEY', 'GPTZERO_API_KEY', 'OPENAI_API_KEY']) {
    const v = process.env[k];
    if (v) forwardedEnv.push(`export ${k}='${v}'`);
  }
  const envPrefix = forwardedEnv.length ? forwardedEnv.join('; ') + '; ' : '';
  const launch = await conn.run(
    `cd /workspace && rm -f output/done && ` +
      `nohup setsid bash -c '${envPrefix}python -u code/train.py > output/run.log 2>&1; ` +
      `if [ -f output/done ]; then echo OK; else echo FAILED > output/run_status; fi' ` +
      `> /dev/null 2>&1 < /dev/null & echo $! > output/training.pid && sleep 2`,
  );
  // Note typo above — fixing inline since edits are restricted in this fn closure.
  if (launch.code !== 0) throw new Error(`failed to launch training: ${launch.stderr}`);
}

/**
 * Poll the pod for /workspace/output/done. Reconnects SSH on failure (so a
 * brief network blip or laptop sleep doesn't break the watch).
 */
async function watchUntilDone(podInfo: PodInfo, privateKeyPath: string): Promise<void> {
  console.log(`> watching pod ${podInfo.id} (poll every ${POLL_SECONDS}s)...`);
  while (true) {
    let conn: PodConnection | null = null;
    try {
      conn = await connectWithRetry(podInfo, privateKeyPath);
      // Check sentinel + that the python process still exists.
      const check = await conn.run(
        `if [ -f /workspace/output/done ]; then echo DONE; ` +
          `elif [ -f /workspace/output/training.pid ] && kill -0 "$(cat /workspace/output/training.pid)" 2>/dev/null; then ` +
          `tail -1 /workspace/output/run.log 2>/dev/null | head -c 200; echo; echo RUNNING; ` +
          `else echo DEAD; fi`,
      );
      const status = check.stdout.trim().split('\n').pop()!;
      const tail = check.stdout.trim().split('\n').slice(0, -1).join(' ');
      const ts = new Date().toISOString().slice(11, 19);
      if (status === 'DONE') {
        console.log(`[${ts}] training done.`);
        return;
      } else if (status === 'DEAD') {
        const log = await conn.run('tail -50 /workspace/output/run.log');
        throw new Error(`training process died without writing 'done':\n${log.stdout.slice(-1500)}`);
      } else {
        console.log(`[${ts}] running. last log: ${tail.slice(0, 140)}`);
      }
    } catch (e) {
      const ts = new Date().toISOString().slice(11, 19);
      console.log(`[${ts}] watch error (will retry): ${e instanceof Error ? e.message : e}`);
    } finally {
      conn?.dispose();
    }
    await new Promise((r) => setTimeout(r, POLL_SECONDS * 1000));
  }
}

/**
 * Download adapter + eval and terminate the pod.
 */
async function downloadAndTerminate(
  podInfo: PodInfo,
  privateKeyPath: string,
  client: RunPodClient,
  hourlyRate: number | null,
  startMs: number,
): Promise<TrainResult> {
  const conn = await connectWithRetry(podInfo, privateKeyPath);
  try {
    console.log(`> downloading adapter -> ${ADAPTER_OUT}`);
    await conn.downloadDir('/workspace/output/adapter', ADAPTER_OUT);

    console.log(`> downloading eval -> ${EVAL_OUT}`);
    const evalJson = await conn.readFile('/workspace/output/eval.json');
    await mkdir(dirname(EVAL_OUT), { recursive: true });
    await writeFile(EVAL_OUT, evalJson);

    const podSeconds = (Date.now() - startMs) / 1000;
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
    console.log(`> terminating pod ${podInfo.id}`);
    await client.terminate(podInfo.id);
  }
}

async function main(): Promise<TrainResult> {
  const privateKeyPath = process.env.RUNPOD_PRIVATE_KEY;
  if (!privateKeyPath) throw new Error('RUNPOD_PRIVATE_KEY not set.');

  // --resume <pod-id> flag.
  const args = process.argv.slice(2);
  const resumeIdx = args.indexOf('--resume');
  const resumePodId = resumeIdx >= 0 ? args[resumeIdx + 1] : null;

  const cfg = loadCfg();
  const client = new RunPodClient(cfg);

  let podInfo: PodInfo;
  let hourlyRate: number | null = null;
  let startMs = Date.now();

  if (resumePodId) {
    console.log(`> resuming pod ${resumePodId}`);
    podInfo = await client.getPod(resumePodId);
    if (!podInfo.sshHost) throw new Error(`pod ${resumePodId} has no SSH endpoint (still booting? terminated?)`);
    console.log(`  SSH ${podInfo.sshHost}:${podInfo.sshPort}`);
  } else {
    const provisioned = await provision();
    podInfo = provisioned.pod;
    hourlyRate = provisioned.hourlyRate;
    const conn = await connectWithRetry(podInfo, privateKeyPath);
    try {
      await kickOffTraining(conn);
    } finally {
      conn.dispose();
    }
  }

  await watchUntilDone(podInfo, privateKeyPath);
  return downloadAndTerminate(podInfo, privateKeyPath, client, hourlyRate, startMs);
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    console.error('\nlaunch failed:', e instanceof Error ? e.message : e);
    process.exit(1);
  }
}
