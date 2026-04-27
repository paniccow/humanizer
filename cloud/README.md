# humanizer cloud — TypeScript orchestration + Python GRPO inner loop

Hybrid stack: **everything in TypeScript** except the ~150-line GRPO inner loop, which has to be Python because no TS library does GRPO updates on a 1.5B-param LLM. You will only ever touch TS in normal use; `train.py` is opaque infrastructure pushed to the rented pod.

```
cloud/
├── package.json          # bun + hono + node-ssh
├── tsconfig.json         # strict, no emit
├── types.ts              # all shared interfaces (PodInfo, TrainConfig, ...)
├── runpod.ts             # GraphQL client (createPod, getPod, terminate)
├── ssh.ts                # node-ssh wrapper (upload/run/download)
├── provision.ts          # spin up + wait for SSH (standalone CLI too)
├── launch.ts             # END-TO-END: provision → train → eval → download → terminate
├── serve.ts              # Bun + Hono HTTP server with persistent Python worker
├── client.ts             # typed SDK
├── train.py              # GRPO inner loop (Python, runs on the pod)
└── worker.py             # long-lived inference worker for serve.ts
```

## How much training do we actually need?

| Steps | Wall (4090) | $ on RunPod | What you get |
|------:|------------:|------------:|--------------|
|   100 |   ~1 hr     |     ~$0.40  | smoke test only — policy barely shifts |
|   300 |   ~3 hr     |     ~$1.10  | sees real gain over best-of-N           |
| **600** | **~6 hr** | **~$2.20** | **default — close to AuthorMist's 714** |
|  1000 |   ~9 hr     |     ~$3.30  | diminishing returns; risk of overfit    |

**Default is 600.** AuthorMist (the SOTA recipe we're copying) used 714 steps with G=8 and reported 78-96% ASR; we're within 15% of that with our 600-step budget. Override via `HUMANIZER_TRAIN_N=300 bun run launch.ts` if you want a smoke test first.

Budget **$5** on your RunPod balance: covers the default run plus one retry if anything goes sideways.

## One-time setup (5 minutes)

```bash
# 1. Bun (or use Node — you'll need to install ts-node if so)
curl -fsSL https://bun.sh/install | bash

# 2. install JS deps
cd cloud
bun install

# 3. RunPod account + API key
#    https://www.runpod.io/console/user/settings  → +Create API Key (read+write)
#    add ~$5 of credit if you don't have $30 trial credit

# 4. SSH key (you almost certainly already have one)
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
```

## Launch a real training run

```bash
export RUNPOD_API_KEY="pa_yourkeyhere..."
export RUNPOD_PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)"
export RUNPOD_PRIVATE_KEY="$HOME/.ssh/id_ed25519"

cd cloud
bun run launch.ts
```

What happens:
1. Verifies a 4090 is available + price; spins up the pod.
2. Waits for SSH (60-90s).
3. Uploads `train.py`.
4. `pip install`s the training stack on the pod (~2 min, one time).
5. Runs GRPO. Logs stream to your terminal.
6. Downloads the LoRA adapter (~30 MB) to `./adapter/`.
7. Downloads `eval.json` showing base vs trained on 30 held-out prompts.
8. **Terminates the pod** (always — even on error, in the `finally` block).
9. Prints summary + estimated cost.

## Serve the trained model

```bash
# Mac/Linux — Mac uses MPS, anything with CUDA uses the GPU
export ADAPTER_PATH=./adapter
bun run serve.ts                # listens on :8000
```

```bash
curl -XPOST http://localhost:8000/humanize \
  -H 'content-type: application/json' \
  -d '{"text":"Furthermore, organizations leverage AI to navigate intricate complexities.","n":4,"burstiness":true}'
```

```ts
import { HumanizerClient } from './client.ts';

const client = new HumanizerClient({ baseUrl: 'http://localhost:8000' });
const { text, pAi, similarity } = await client.humanize({ text: aiText, n: 4 });
```

The server keeps one Python subprocess alive that holds the adapter resident in GPU/MPS RAM, so requests don't pay model-load latency.

## Tearing down a stuck pod manually

```bash
bun run provision.ts terminate <pod-id>     # if launch.ts crashed before cleanup
```

You can also cancel from the [RunPod dashboard](https://www.runpod.io/console/pods).

## Tuning

Override any of these as env vars to `launch.ts`:

| var                     | default                            | effect                                      |
|-------------------------|------------------------------------|---------------------------------------------|
| `RUNPOD_GPU`            | `NVIDIA GeForce RTX 4090`          | swap to `NVIDIA RTX A5000` etc.             |
| `RUNPOD_CLOUD_TYPE`     | `COMMUNITY`                        | `SECURE` for SLA-backed (~2× price)         |
| `HUMANIZER_BASE`        | `Qwen/Qwen2.5-1.5B-Instruct`       | bigger = better, watch VRAM                 |
| `HUMANIZER_TRAIN_N`     | `600`                              | training prompts                            |
| `HUMANIZER_G`           | `8`                                | GRPO group size                             |
| `HUMANIZER_LR`          | `5e-6`                             | learning rate                               |
| `HUMANIZER_BETA`        | `0.001`                            | KL penalty against frozen reference         |
| `HUMANIZER_LORA_R`      | `16`                               | LoRA rank                                   |
| `HUMANIZER_QLORA`       | `1`                                | `0` to disable 4-bit (needs 32GB+ VRAM)     |
| `ADAPTER_OUT`           | `./adapter`                        | local download path                         |

## What if it costs more than expected?

Two safety belts:
1. `train.py` writes a `done` sentinel only on success. `launch.ts` always calls `terminate()` in a `finally`, so even a crash kills the pod.
2. Set a hard cap in your RunPod account: Settings → Spending Alerts → Auto-pause at $X.

## Why hybrid (and not all-TS)?

GRPO on a 1.5B-param LLM needs:
- A real autograd engine with KV-cache-aware generation (PyTorch/JAX).
- LoRA + QLoRA adapter math (PEFT).
- Group-relative advantage + KL-penalized PPO loss (TRL).

None of that exists in TypeScript at production quality in 2026. Inference can run via `transformers.js` or ONNX Runtime Web, but not training. So we draw the line at the GRPO loop itself: TS owns provisioning, monitoring, the public API, and the SDK; Python owns the ~150 lines that touch CUDA kernels.
