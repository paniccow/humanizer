/**
 * Live training monitor — tails training.log and shows rolling stats.
 *
 *   bun run monitor.ts                   # ~/humanizer/cloud/training.log by default
 *   bun run monitor.ts ./run.log         # explicit path
 *
 * Shows: current step / total, last reward, last p_ai, KL drift, projected
 * ETA based on the per-step time in the log, and a sparkline of mean reward
 * over the last 30 steps so you can eyeball the training curve.
 *
 * Pure stdout, no TTY tricks — works over SSH, in tmux, in a CI tail.
 */
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const PATH = resolve(process.argv[2] ?? `${process.env.HOME}/humanizer/cloud/training.log`);
const POLL_MS = 5_000;

interface Step {
  step: number;
  total: number;
  meanReward: number;
  maxReward: number;
  pAi: number;
  kl: number;
  loss: number;
  tSec: number;
}

const STEP_RE =
  /^step\s+(\d+)\/(\d+)\s+R̄=([-\d.]+)\s+R_max=([-\d.]+)\s+p_ai=([-\d.]+)\s+KL=([-+\d.]+)\s+loss=([-+\d.]+)\s+t=([\d.]+)s/;

function parseSteps(content: string): Step[] {
  const out: Step[] = [];
  for (const line of content.split('\n')) {
    const m = STEP_RE.exec(line);
    if (!m) continue;
    out.push({
      step: Number(m[1]),
      total: Number(m[2]),
      meanReward: Number(m[3]),
      maxReward: Number(m[4]),
      pAi: Number(m[5]),
      kl: Number(m[6]),
      loss: Number(m[7]),
      tSec: Number(m[8]),
    });
  }
  return out;
}

const SPARK = '▁▂▃▄▅▆▇█';
function sparkline(values: number[], min = 0, max = 1): string {
  if (!values.length) return '';
  const range = max - min || 1e-9;
  return values
    .map((v) => {
      const idx = Math.min(SPARK.length - 1, Math.max(0, Math.floor(((v - min) / range) * SPARK.length)));
      return SPARK[idx];
    })
    .join('');
}

function fmtDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${(sec % 60).toFixed(0)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

async function readSafe(path: string): Promise<string> {
  try {
    return await readFile(path, 'utf8');
  } catch (e) {
    return '';
  }
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

let lastStep = 0;
let lastSize = 0;
let firstStepT = 0;

async function tick(): Promise<void> {
  const content = await readSafe(PATH);
  if (!content) {
    process.stdout.write(`\r[monitor] waiting for ${PATH}...`);
    return;
  }
  const steps = parseSteps(content);
  if (!steps.length) {
    process.stdout.write(`\r[monitor] log opened, no step lines yet (${content.length} bytes)...`);
    return;
  }
  const last = steps[steps.length - 1]!;
  if (firstStepT === 0) firstStepT = steps[0]!.tSec;

  if (last.step === lastStep) return; // no new step since last tick
  lastStep = last.step;

  const recent = steps.slice(-30);
  const recentR = recent.map((s) => s.meanReward);
  const recentP = recent.map((s) => s.pAi);
  const stepsLeft = last.total - last.step;
  const avgPerStep = (last.tSec - firstStepT) / Math.max(last.step - steps[0]!.step, 1);
  const eta = stepsLeft * avgPerStep;
  const elapsed = last.tSec;
  const cost4090 = (elapsed / 3600) * 0.34;
  const projCost = ((elapsed + eta) / 3600) * 0.34;

  process.stdout.write('\x1b[2K\r'); // clear current line
  console.log(
    [
      `[step ${String(last.step).padStart(4)}/${last.total}]`,
      `R̄=${last.meanReward.toFixed(3)} (avg30=${mean(recentR).toFixed(3)})`,
      `R_max=${last.maxReward.toFixed(3)}`,
      `p_ai=${last.pAi.toFixed(3)} (avg30=${mean(recentP).toFixed(3)})`,
      `KL=${last.kl.toFixed(3)}`,
      `loss=${last.loss.toFixed(3)}`,
      `t=${fmtDuration(elapsed)}  ETA ${fmtDuration(eta)}`,
      `cost ~$${cost4090.toFixed(2)} (proj $${projCost.toFixed(2)})`,
    ].join('  '),
  );
  console.log(`  R̄ last30:    ${sparkline(recentR, 0, 1)}`);
  console.log(`  p_ai last30: ${sparkline(recentP, 0, 1)}  (lower = more human)`);

  // Stop if we've reached the end.
  if (last.step >= last.total) {
    console.log(`\n[monitor] training reached ${last.step}/${last.total}, exiting.`);
    process.exit(0);
  }
  // Also stop if launch.ts wrote a 'terminating pod' or 'launch failed' line.
  if (/launch failed|terminating pod/.test(content.split('\n').slice(-25).join('\n'))) {
    console.log(`\n[monitor] launch.ts reports termination/failure, exiting.`);
    process.exit(0);
  }
}

console.log(`[monitor] watching ${PATH} (poll every ${POLL_MS / 1000}s)`);
if (!(await exists(PATH))) {
  console.log(`[monitor] file does not exist yet — will wait for it`);
}
await tick();
setInterval(tick, POLL_MS);
