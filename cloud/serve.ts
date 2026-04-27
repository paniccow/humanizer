/**
 * Bun + Hono HTTP server that exposes the trained humanizer.
 *
 *   bun run serve.ts                    # serves on :8000
 *   curl localhost:8000/health
 *   curl -XPOST localhost:8000/humanize -d '{"text":"...","n":4}'
 *
 * Architecture: TypeScript handles routing, request validation, batching,
 * and the public API; the actual model inference is a long-lived Python
 * subprocess that holds the LoRA adapter loaded in GPU/MPS RAM. We talk to
 * it over stdin/stdout JSON-lines so we never pay model-load latency per
 * request.
 *
 * On a Mac the subprocess uses MPS; on a real GPU machine it uses CUDA.
 */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { Hono } from 'hono';

import type { HumanizeRequest, HumanizeResponse } from './types.ts';

const PORT = Number(process.env.PORT ?? 8000);
const ADAPTER_PATH = resolve(process.cwd(), process.env.ADAPTER_PATH ?? 'adapter');
const BASE_MODEL = process.env.BASE_MODEL ?? 'Qwen/Qwen2.5-1.5B-Instruct';
const PYTHON = process.env.PYTHON ?? 'python3';
const WORKER = resolve(import.meta.dir, 'worker.py');

// ---- One persistent Python worker ------------------------------------------

interface PendingRequest {
  resolve: (r: HumanizeResponse) => void;
  reject: (e: Error) => void;
}

class HumanizerWorker {
  private proc: ReturnType<typeof spawn> | null = null;
  private buffer = '';
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private ready: Promise<void> | null = null;

  start(): Promise<void> {
    if (this.ready) return this.ready;
    this.ready = new Promise((readyResolve, readyReject) => {
      const proc = spawn(
        PYTHON,
        ['-u', WORKER, '--base', BASE_MODEL, '--adapter', ADAPTER_PATH],
        { stdio: ['pipe', 'pipe', 'inherit'] },
      );
      this.proc = proc;

      proc.on('error', (e) => readyReject(e));
      proc.on('exit', (code) => {
        for (const [, p] of this.pending) p.reject(new Error(`worker exited ${code}`));
        this.pending.clear();
        this.proc = null;
        this.ready = null;
      });

      proc.stdout!.on('data', (chunk: Buffer) => {
        this.buffer += chunk.toString('utf8');
        let nl: number;
        while ((nl = this.buffer.indexOf('\n')) >= 0) {
          const line = this.buffer.slice(0, nl);
          this.buffer = this.buffer.slice(nl + 1);
          if (!line.trim()) continue;
          let msg: { type: string; id?: number; ok?: boolean; data?: HumanizeResponse; error?: string };
          try {
            msg = JSON.parse(line);
          } catch {
            console.error('worker emitted non-JSON line:', line);
            continue;
          }
          if (msg.type === 'ready') {
            readyResolve();
            continue;
          }
          if (msg.type === 'response' && msg.id !== undefined) {
            const p = this.pending.get(msg.id);
            if (!p) continue;
            this.pending.delete(msg.id);
            if (msg.ok && msg.data) p.resolve(msg.data);
            else p.reject(new Error(msg.error ?? 'unknown worker error'));
          }
        }
      });
    });
    return this.ready;
  }

  async humanize(req: HumanizeRequest): Promise<HumanizeResponse> {
    if (!this.proc) await this.start();
    const id = this.nextId++;
    const promise = new Promise<HumanizeResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.proc!.stdin!.write(JSON.stringify({ type: 'humanize', id, ...req }) + '\n');
    return promise;
  }

  shutdown(): void {
    this.proc?.kill('SIGTERM');
  }
}

// ---- HTTP API ---------------------------------------------------------------

const worker = new HumanizerWorker();
const app = new Hono();

app.get('/health', (c) =>
  c.json({ ok: true, baseModel: BASE_MODEL, adapter: ADAPTER_PATH, workerAlive: !!(worker as any).proc }),
);

app.post('/humanize', async (c) => {
  const body = await c.req.json<HumanizeRequest>();
  if (typeof body.text !== 'string' || !body.text.trim()) {
    return c.json({ error: 'text (non-empty string) is required' }, 400);
  }
  const response = await worker.humanize({
    text: body.text,
    n: body.n ?? 1,
    temperature: body.temperature ?? 0.85,
    burstiness: body.burstiness ?? false,
  });
  return c.json(response);
});

console.log(`[serve] starting worker (model=${BASE_MODEL}, adapter=${ADAPTER_PATH})`);
await worker.start();
console.log(`[serve] worker ready; listening on :${PORT}`);

const server = Bun.serve({ port: PORT, fetch: app.fetch });

const shutdown = () => {
  console.log('\n[serve] shutting down');
  worker.shutdown();
  server.stop();
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
