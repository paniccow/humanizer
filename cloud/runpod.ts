/**
 * Minimal RunPod GraphQL client.
 *
 * Only the four operations we need:
 *   - findGpuType  : verify our GPU is available + price
 *   - createPod    : spin up an on-demand pod with our image + SSH key
 *   - getPod       : poll for runtime ready, extract SSH endpoint
 *   - terminatePod : nuke the pod when done
 */
import type { PodInfo, RunPodConfig } from './types.ts';

const ENDPOINT = 'https://api.runpod.io/graphql';

export class RunPodClient {
  constructor(private readonly cfg: RunPodConfig) {}

  private async gql<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
    const res = await fetch(`${ENDPOINT}?api_key=${this.cfg.apiKey}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query, variables }),
    });
    if (!res.ok) {
      throw new Error(`RunPod GraphQL ${res.status}: ${await res.text()}`);
    }
    const json = (await res.json()) as { data?: T; errors?: Array<{ message: string }> };
    if (json.errors?.length) {
      throw new Error(`RunPod GraphQL errors: ${json.errors.map((e) => e.message).join(', ')}`);
    }
    if (!json.data) throw new Error('RunPod GraphQL returned no data');
    return json.data;
  }

  /** Find the GPU type by id; returns the lowest community price/hr if available. */
  async findGpuType(gpuTypeId: string): Promise<{ id: string; communityPrice: number | null; secureCloud: number | null }> {
    const data = await this.gql<{
      gpuTypes: Array<{
        id: string;
        displayName: string;
        lowestPrice: { uninterruptablePrice: number | null; minimumBidPrice: number | null };
        secureCloud: number | null;
        communityCloud: number | null;
      }>;
    }>(`
      query GpuTypes {
        gpuTypes {
          id
          displayName
          lowestPrice(input: { gpuCount: 1 }) {
            uninterruptablePrice
            minimumBidPrice
          }
          secureCloud
          communityCloud
        }
      }
    `);
    const match = data.gpuTypes.find((g) => g.id === gpuTypeId);
    if (!match) {
      const available = data.gpuTypes.map((g) => g.id).join(', ');
      throw new Error(`GPU type "${gpuTypeId}" not found. Available: ${available}`);
    }
    return {
      id: match.id,
      communityPrice: match.lowestPrice.uninterruptablePrice,
      secureCloud: match.secureCloud ?? null,
    };
  }

  /**
   * Create an on-demand pod. We use the official runpod/pytorch image which
   * comes with SSHD + CUDA + a base PyTorch install. We pass our public SSH
   * key via the PUBLIC_KEY env var (the runpod/pytorch image installs it
   * into authorized_keys at boot).
   */
  async createPod(opts: {
    image: string;
    name: string;
    publicKey: string;
    cloudType?: 'COMMUNITY' | 'SECURE' | 'ALL';
  }): Promise<PodInfo> {
    const cloudType = opts.cloudType ?? this.cfg.cloudType ?? 'COMMUNITY';
    const data = await this.gql<{ podFindAndDeployOnDemand: { id: string; desiredStatus: string } }>(
      `
      mutation Create($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
          id
          desiredStatus
        }
      }
      `,
      {
        input: {
          cloudType,
          gpuCount: 1,
          gpuTypeId: this.cfg.gpuTypeId,
          name: opts.name,
          imageName: opts.image,
          containerDiskInGb: this.cfg.containerDiskInGb,
          volumeInGb: this.cfg.volumeInGb,
          volumeMountPath: '/workspace',
          ports: '22/tcp,8000/http',
          minVcpuCount: 2,
          minMemoryInGb: 8,
          env: [
            { key: 'PUBLIC_KEY', value: opts.publicKey },
            { key: 'JUPYTER_PASSWORD', value: '' }, // disable jupyter
          ],
        },
      },
    );
    return {
      id: data.podFindAndDeployOnDemand.id,
      sshHost: '',
      sshPort: 0,
      desiredStatus: data.podFindAndDeployOnDemand.desiredStatus,
    };
  }

  /** Get pod details — used for polling until SSH is reachable. */
  async getPod(podId: string): Promise<PodInfo> {
    const data = await this.gql<{
      pod: {
        id: string;
        desiredStatus: string;
        runtime: {
          uptimeInSeconds: number;
          ports: Array<{ ip: string; isIpPublic: boolean; publicPort: number; privatePort: number; type: string }>;
        } | null;
      };
    }>(
      `
      query Pod($input: PodFilter!) {
        pod(input: $input) {
          id
          desiredStatus
          runtime {
            uptimeInSeconds
            ports {
              ip
              isIpPublic
              publicPort
              privatePort
              type
            }
          }
        }
      }
      `,
      { input: { podId } },
    );
    const sshPort = data.pod.runtime?.ports?.find((p) => p.privatePort === 22 && p.isIpPublic);
    return {
      id: data.pod.id,
      desiredStatus: data.pod.desiredStatus,
      sshHost: sshPort?.ip ?? '',
      sshPort: sshPort?.publicPort ?? 0,
    };
  }

  /** Terminate. Releases the GPU and the volume. Use with care. */
  async terminate(podId: string): Promise<void> {
    await this.gql(
      `
      mutation Terminate($input: PodTerminateInput!) {
        podTerminate(input: $input)
      }
      `,
      { input: { podId } },
    );
  }
}

/** Block until `predicate(pod)` returns true; throws after `timeoutMs`. */
export async function waitForPod(
  client: RunPodClient,
  podId: string,
  predicate: (pod: PodInfo) => boolean,
  opts: { intervalMs?: number; timeoutMs?: number; onTick?: (pod: PodInfo) => void } = {},
): Promise<PodInfo> {
  const interval = opts.intervalMs ?? 5_000;
  const timeout = opts.timeoutMs ?? 10 * 60_000;
  const start = Date.now();
  while (true) {
    const pod = await client.getPod(podId);
    opts.onTick?.(pod);
    if (predicate(pod)) return pod;
    if (Date.now() - start > timeout) {
      throw new Error(`Timeout waiting for pod ${podId} (last status: ${pod.desiredStatus})`);
    }
    await new Promise((r) => setTimeout(r, interval));
  }
}
