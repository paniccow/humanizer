"""Held-out detector eval — runs on a GPU pod (or anywhere with GPU).

Loads detectors NOT used in training reward. Scores the trained model's
outputs (already in eval.json) against them. The "did we generalize, or did
we just overfit to the training detectors?" verdict.

Usage:
    python heldout_eval.py /workspace/output/eval.json

Writes /workspace/output/holdout_eval.json next to the input.

The script is dependency-light: only `transformers`, `torch`. No TRL, no
PEFT, no datasets — just inference on the existing trained outputs.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Held-out detectors. Distinct from the training set:
#   - SuperAnnotate roberta-large (different team's training data)
#   - coai roberta-base (different corpus)
#   - HC3-trained (Hello-SimpleAI) — different distribution entirely
HOLDOUT_IDS = (
    "SuperAnnotate/roberta-large-llm-content-detector",
    "coai/roberta-ai-detector-v2",
    "Hello-SimpleAI/chatgpt-detector-roberta",
)


def load_detectors(device: str = "cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = []
    for det_id in HOLDOUT_IDS:
        print(f"[holdout] loading {det_id}", flush=True)
        try:
            d = AutoModelForSequenceClassification.from_pretrained(
                det_id, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
            ).to(device).eval()
            for p in d.parameters():
                p.requires_grad_(False)
            t = AutoTokenizer.from_pretrained(det_id)
            id2 = d.config.id2label
            ai_idx = next(
                i for i, l in id2.items()
                if "fake" in str(l).lower() or "label_1" in str(l).lower() or i == d.config.num_labels - 1
            )
            out.append((det_id, d, t, int(ai_idx)))
        except Exception as e:  # noqa: BLE001
            print(f"[holdout] FAILED {det_id}: {e}", flush=True)
    if not out:
        raise RuntimeError("no held-out detectors loaded")
    return out


def score_batch(detector, tokenizer, ai_idx, texts, device: str):
    import torch

    with torch.no_grad():
        enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        logits = detector(**enc).logits.float()
        return torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist()


def evaluate(eval_path: Path, device: str = "cuda") -> dict:
    """Load eval.json (with base/trained outputs), run held-out detectors,
    return holdout summary + per-example detail."""
    data = json.loads(eval_path.read_text())
    base_outputs = data["base"]["outputs"]
    trained_outputs = data["trained"]["outputs"]
    n = len(trained_outputs)
    print(f"[holdout] eval set: {n} prompts", flush=True)

    detectors = load_detectors(device=device)

    base_per: dict[str, list[float]] = {}
    trained_per: dict[str, list[float]] = {}

    for det_id, d, t, ai_idx in detectors:
        name = det_id.split("/")[-1]
        print(f"[holdout] scoring with {name}", flush=True)
        base_per[name] = score_batch(d, t, ai_idx, base_outputs, device)
        trained_per[name] = score_batch(d, t, ai_idx, trained_outputs, device)

    base_ens = [
        sum(base_per[name][i] for name in base_per) / len(base_per)
        for i in range(n)
    ]
    trained_ens = [
        sum(trained_per[name][i] for name in trained_per) / len(trained_per)
        for i in range(n)
    ]

    summary = {
        "n": n,
        "holdout_detectors": [det_id for det_id, *_ in detectors],
        "base": {
            "mean_p_ai_holdout": statistics.fmean(base_ens),
            "asr_holdout_ensemble": sum(1 for p in base_ens if p < 0.5) / n,
            "asr_per_detector": {
                name: sum(1 for p in scores if p < 0.5) / n
                for name, scores in base_per.items()
            },
        },
        "trained": {
            "mean_p_ai_holdout": statistics.fmean(trained_ens),
            "asr_holdout_ensemble": sum(1 for p in trained_ens if p < 0.5) / n,
            "asr_per_detector": {
                name: sum(1 for p in scores if p < 0.5) / n
                for name, scores in trained_per.items()
            },
        },
    }
    summary["delta_p_ai_holdout"] = (
        summary["trained"]["mean_p_ai_holdout"] - summary["base"]["mean_p_ai_holdout"]
    )
    summary["delta_asr_holdout"] = (
        summary["trained"]["asr_holdout_ensemble"] - summary["base"]["asr_holdout_ensemble"]
    )

    return {
        "summary": summary,
        "base_per_detector_p_ai": base_per,
        "trained_per_detector_p_ai": trained_per,
        "base_ensemble_p_ai": base_ens,
        "trained_ensemble_p_ai": trained_ens,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python heldout_eval.py <path/to/eval.json>", file=sys.stderr)
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[holdout] device={device}", flush=True)

    eval_path = Path(sys.argv[1]).resolve()
    if not eval_path.exists():
        print(f"missing: {eval_path}", file=sys.stderr)
        sys.exit(1)

    result = evaluate(eval_path, device=device)
    out_path = eval_path.parent / (eval_path.stem + ".holdout.json")
    out_path.write_text(json.dumps(result, indent=2))

    print("\n========== HELD-OUT EVAL SUMMARY ==========", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"\nFull output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
