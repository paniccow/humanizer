/**
 * Shared types for the cloud orchestration + serving layer.
 *
 * The training inner loop runs in Python on the GPU pod (no TS ML stack does
 * GRPO on a 1.5B-param LLM). Everything else — provisioning, log streaming,
 * adapter download, serving, client SDK — is TypeScript.
 */

export interface RunPodConfig {
  apiKey: string;
  /** GPU type ID. RunPod's name for the consumer 4090. */
  gpuTypeId: string;
  /** Container disk in GB — holds the docker image + code. */
  containerDiskInGb: number;
  /** Persistent volume in GB — survives pod restarts; holds adapters. */
  volumeInGb: number;
  /** SSH public key (as found in ~/.ssh/id_ed25519.pub). Required to log in. */
  publicSshKey: string;
  /** Optional pod name shown in the RunPod dashboard. */
  podName?: string;
  /** Cloud type — COMMUNITY = cheapest 4090s, SECURE = SLA-backed. */
  cloudType?: 'COMMUNITY' | 'SECURE' | 'ALL';
  /** Region preference; "ANY" means RunPod picks. */
  countryCode?: string;
}

export interface PodInfo {
  id: string;
  /** SSH host (publicly routable IP). */
  sshHost: string;
  /** SSH port (mapped from container port 22). */
  sshPort: number;
  desiredStatus: 'RUNNING' | 'EXITED' | 'PAUSED' | string;
  /** GPU type the pod actually got assigned. */
  gpuType?: string;
  costPerHr?: number;
}

export interface TrainConfig {
  /** Hugging Face model id of the base policy. */
  baseModel: string;
  /** Hugging Face dataset id used to source AI-labelled prompts. */
  dataset: string;
  /** How many GRPO update steps. AuthorMist used 714. */
  numSteps: number;
  /** GRPO group size (G). 8 is the AuthorMist default. */
  numGenerations: number;
  /** Inner-loop learning rate. */
  learningRate: number;
  /** KL penalty weight against the frozen reference policy. */
  betaKl: number;
  /** LoRA rank. */
  loraR: number;
  /** Use 4-bit QLoRA — required for Qwen-1.5B GRPO on 24GB. */
  useQLora: boolean;
  /** Number of training prompts streamed from the dataset. */
  numTrainPrompts: number;
  /** Number of held-out eval prompts. */
  numEvalPrompts: number;
}

export interface TrainResult {
  /** Local path the adapter was downloaded to. */
  adapterPath: string;
  /** Eval JSON pulled from the pod (base vs trained on the held-out set). */
  eval: EvalReport;
  /** Total wall-clock seconds the pod was billable. */
  podSeconds: number;
  /** Estimated dollar cost = podSeconds × hourly / 3600. */
  estimatedCostUsd: number;
}

export interface EvalReport {
  n: number;
  base: { mean_p_ai_ensemble: number; mean_similarity: number; asr_ensemble: number };
  trained: { mean_p_ai_ensemble: number; mean_similarity: number; asr_ensemble: number };
  delta_p_ai_ensemble: number;
  delta_similarity: number;
  delta_asr_ensemble: number;
}

export interface HumanizeRequest {
  text: string;
  /** Number of best-of-N candidates. 1 = trained model only. */
  n?: number;
  /** Sampling temperature. */
  temperature?: number;
  /** Apply burstiness post-processing. */
  burstiness?: boolean;
}

export interface HumanizeResponse {
  text: string;
  pAi: number;
  similarity: number;
  attempts: number;
}
