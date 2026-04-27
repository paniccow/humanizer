/**
 * Provision an RTX 4090 RunPod pod ready to accept SSH + a Python training run.
 *
 * Standalone CLI:
 *   bun run provision.ts                  → create + wait + print SSH details
 *   bun run provision.ts terminate <id>   → kill a running pod
 *
 * Or imported from launch.ts as part of the full pipeline.
 */
import { RunPodClient, waitForPod } from './runpod.ts';
import type { PodInfo, RunPodConfig } from './types.ts';

const POD_IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04';

function loadConfig(): RunPodConfig {
  const apiKey = process.env.RUNPOD_API_KEY;
  if (!apiKey) {
    throw new Error('RUNPOD_API_KEY not set. Get one at https://www.runpod.io/console/user/settings');
  }
  const publicSshKey = process.env.RUNPOD_PUBLIC_KEY;
  if (!publicSshKey) {
    throw new Error(
      'RUNPOD_PUBLIC_KEY not set. Run:\n' +
        '  cat ~/.ssh/id_ed25519.pub  (or your key)\n' +
        '  export RUNPOD_PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)"',
    );
  }
  return {
    apiKey,
    publicSshKey,
    gpuTypeId: process.env.RUNPOD_GPU ?? 'NVIDIA GeForce RTX 4090',
    containerDiskInGb: Number(process.env.RUNPOD_CONTAINER_GB ?? 50),
    volumeInGb: Number(process.env.RUNPOD_VOLUME_GB ?? 30),
    cloudType: (process.env.RUNPOD_CLOUD_TYPE as 'COMMUNITY' | 'SECURE' | 'ALL' | undefined) ?? 'COMMUNITY',
    podName: process.env.RUNPOD_POD_NAME ?? 'humanizer-grpo',
  };
}

export interface ProvisionResult {
  pod: PodInfo;
  client: RunPodClient;
  /** ssh command-line ready to use: `ssh -p 12345 root@1.2.3.4` */
  sshCommand: string;
  hourlyRate: number | null;
}

/** Spin up a pod and wait until SSH is reachable. */
export async function provision(): Promise<ProvisionResult> {
  const cfg = loadConfig();
  const client = new RunPodClient(cfg);

  console.log(`> Verifying GPU type "${cfg.gpuTypeId}"...`);
  const gpu = await client.findGpuType(cfg.gpuTypeId);
  const rate = gpu.communityPrice ?? gpu.secureCloud;
  console.log(`  found. lowest community $/hr = ${gpu.communityPrice ?? 'n/a'}`);

  console.log(`> Creating pod (image=${POD_IMAGE}, gpu=${cfg.gpuTypeId})...`);
  const created = await client.createPod({
    image: POD_IMAGE,
    name: cfg.podName ?? 'humanizer-grpo',
    publicKey: cfg.publicSshKey,
    cloudType: cfg.cloudType,
  });
  console.log(`  pod id: ${created.id}`);

  console.log(`> Waiting for SSH to come up (typically 60-90s)...`);
  const pod = await waitForPod(
    client,
    created.id,
    (p) => Boolean(p.sshHost && p.sshPort && p.desiredStatus === 'RUNNING'),
    {
      intervalMs: 5_000,
      timeoutMs: 6 * 60_000,
      onTick: (p) =>
        console.log(`    status=${p.desiredStatus} ssh=${p.sshHost ?? '?'}:${p.sshPort ?? '?'}`),
    },
  );

  const sshCommand = `ssh -p ${pod.sshPort} root@${pod.sshHost}`;
  console.log(`> Pod ready. SSH:  ${sshCommand}`);
  return { pod, client, sshCommand, hourlyRate: rate };
}

/** Terminate a pod by id. */
export async function terminate(podId: string): Promise<void> {
  const cfg = loadConfig();
  const client = new RunPodClient(cfg);
  console.log(`> Terminating pod ${podId}`);
  await client.terminate(podId);
  console.log(`  done.`);
}

/** CLI entry. */
if (import.meta.main) {
  const [cmd, arg] = Bun.argv.slice(2);
  if (cmd === 'terminate') {
    if (!arg) {
      console.error('usage: bun run provision.ts terminate <pod-id>');
      process.exit(1);
    }
    await terminate(arg);
  } else {
    const r = await provision();
    console.log(`\nNext: ssh in (${r.sshCommand}) or run 'bun run launch.ts' for end-to-end.`);
    console.log(`Estimated cost: $${(r.hourlyRate ?? 0).toFixed(3)}/hr (community 4090).`);
  }
}
