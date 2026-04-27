/**
 * Thin wrappers around node-ssh for the operations we need:
 *   - copy a file or directory to the pod
 *   - run a command, streaming stdout/stderr
 *   - download a file or directory
 *
 * We intentionally keep this minimal — the launch script orchestrates a small
 * fixed sequence (upload → run → download → terminate), no general-purpose
 * remote execution.
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { NodeSSH } from 'node-ssh';

import type { PodInfo } from './types.ts';

export class PodConnection {
  private ssh = new NodeSSH();

  constructor(private readonly pod: PodInfo, private readonly privateKeyPath: string) {}

  async connect(): Promise<void> {
    await this.ssh.connect({
      host: this.pod.sshHost,
      port: this.pod.sshPort,
      username: 'root',
      privateKeyPath: this.privateKeyPath,
      // RunPod pods boot with fresh host keys every time; skip strict checking.
      tryKeyboard: false,
    });
  }

  /** Run a command on the pod, streaming stdout/stderr to local console. */
  async run(cmd: string): Promise<{ code: number; stdout: string; stderr: string }> {
    let stdout = '';
    let stderr = '';
    const result = await this.ssh.execCommand(cmd, {
      cwd: '/workspace',
      onStdout: (chunk) => {
        const s = chunk.toString('utf8');
        stdout += s;
        process.stdout.write(s);
      },
      onStderr: (chunk) => {
        const s = chunk.toString('utf8');
        stderr += s;
        process.stderr.write(s);
      },
    });
    return { code: result.code ?? -1, stdout, stderr };
  }

  /** Upload a single local file to a remote path. */
  async uploadFile(localPath: string, remotePath: string): Promise<void> {
    await this.ssh.putFile(localPath, remotePath);
  }

  /** Upload an entire directory. */
  async uploadDir(localDir: string, remoteDir: string): Promise<void> {
    const result = await this.ssh.putDirectory(localDir, remoteDir, {
      recursive: true,
      concurrency: 5,
      validate: (item) => !item.includes('node_modules') && !item.includes('__pycache__') && !item.includes('.git'),
    });
    if (!result) throw new Error(`uploadDir failed: ${localDir} -> ${remoteDir}`);
  }

  /** Read one remote file into a string. */
  async readFile(remotePath: string): Promise<string> {
    const buf = await this.ssh.execCommand(`cat ${shellEscape(remotePath)}`);
    if (buf.code !== 0) throw new Error(`cat ${remotePath}: ${buf.stderr}`);
    return buf.stdout;
  }

  /** Download a directory recursively. */
  async downloadDir(remoteDir: string, localDir: string): Promise<void> {
    await mkdir(localDir, { recursive: true });
    const ok = await this.ssh.getDirectory(localDir, remoteDir, {
      recursive: true,
      concurrency: 5,
    });
    if (!ok) throw new Error(`downloadDir failed: ${remoteDir} -> ${localDir}`);
  }

  /** Download a single file. */
  async downloadFile(remotePath: string, localPath: string): Promise<void> {
    await mkdir(dirname(localPath), { recursive: true });
    await this.ssh.getFile(localPath, remotePath);
  }

  dispose(): void {
    this.ssh.dispose();
  }
}

function shellEscape(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
}
