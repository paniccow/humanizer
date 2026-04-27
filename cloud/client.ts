/**
 * Typed SDK for the humanizer HTTP server.
 *
 *   import { HumanizerClient } from './client.ts';
 *
 *   const client = new HumanizerClient({ baseUrl: 'http://localhost:8000' });
 *   const { text, pAi } = await client.humanize({ text: 'Furthermore, ...', n: 4 });
 */
import type { HumanizeRequest, HumanizeResponse } from './types.ts';

export interface HumanizerClientOptions {
  baseUrl: string;
  /** Optional bearer token for an auth layer you put in front of the server. */
  authToken?: string;
  /** Per-request timeout (ms). Defaults to 60s. */
  timeoutMs?: number;
}

export class HumanizerClient {
  constructor(private readonly opts: HumanizerClientOptions) {}

  /** Calls /humanize. */
  async humanize(req: HumanizeRequest): Promise<HumanizeResponse> {
    return this.post<HumanizeResponse>('/humanize', req);
  }

  /** Calls /health. Throws if the server is down. */
  async health(): Promise<{ ok: true; baseModel: string; adapter: string; workerAlive: boolean }> {
    return this.get('/health');
  }

  // ---- internals ---------------------------------------------------------

  private async get<T>(path: string): Promise<T> {
    return this.request('GET', path);
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>('POST', path, body);
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.opts.timeoutMs ?? 60_000);
    try {
      const headers: Record<string, string> = { 'content-type': 'application/json' };
      if (this.opts.authToken) headers.authorization = `Bearer ${this.opts.authToken}`;
      const res = await fetch(`${this.opts.baseUrl}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`humanizer ${method} ${path} → ${res.status}: ${text || res.statusText}`);
      }
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }
}
