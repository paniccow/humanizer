"""Pod-side: score raw + scrubbed outputs with the SAME detectors used in
training. Validates whether the deterministic scrub actually moves the
detector p_ai (ground truth) or just the pattern aggregate (proxy).

Inputs (set via env vars or first arg):
  EVAL_FILES — colon-separated paths to eval JSONs (default: all 4)
  DETECTOR_IDS — comma-separated detector ids

Output: each input file gets a sibling .scrub-eval.json with:
  per-output detector scores for: raw BASE, scrubbed BASE,
                                  raw TRAINED, scrubbed TRAINED
  ASR per (variant, detector)
  Mean p_ai per (variant, detector)
  Delta tables
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

DEFAULT_EVAL_FILES = (
    "/workspace/eval_files/run1_eval.json",
    "/workspace/eval_files/run2_eval.json",
    "/workspace/eval_files/run3_eval.json",
    "/workspace/eval_files/run4_eval.json",
)
DEFAULT_DETECTORS = (
    "openai-community/roberta-base-openai-detector",
    "openai-community/roberta-large-openai-detector",
)


def load_detectors(ids):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = []
    for det_id in ids:
        print(f"[detector] loading {det_id}", flush=True)
        try:
            d = AutoModelForSequenceClassification.from_pretrained(
                det_id, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
            ).cuda().eval()
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
            print(f"[detector] FAILED {det_id}: {e}", flush=True)
    if not out:
        raise RuntimeError("no detectors loaded")
    return out


def score(model, tok, ai_idx, texts, batch_size=8):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, ai_idx].cpu().tolist()
            out.extend(probs)
    return out


def evaluate(path: Path, detectors) -> dict:
    """Process one eval JSON. Returns full per-variant detector scores."""
    sys.path.insert(0, "/workspace/code")
    from scrub_pkg.scrub import scrub

    data = json.loads(path.read_text())
    base_outs = data["base"]["outputs"]
    trained_outs = data["trained"]["outputs"]
    n = len(trained_outs)
    print(f"[eval] {path.name}: {n} prompts", flush=True)

    base_scrub = [scrub(o).text for o in base_outs]
    trained_scrub = [scrub(o).text for o in trained_outs]

    variants = {
        "base":          base_outs,
        "base_scrub":    base_scrub,
        "trained":       trained_outs,
        "trained_scrub": trained_scrub,
    }

    per_detector_per_variant: dict[str, dict[str, list[float]]] = {}
    for det_id, d, t, ai_idx in detectors:
        det_name = det_id.split("/")[-1]
        per_detector_per_variant[det_name] = {}
        for variant_name, texts in variants.items():
            per_detector_per_variant[det_name][variant_name] = score(d, t, ai_idx, texts)

    # Build summary
    summary: dict = {"n": n, "detectors": list(per_detector_per_variant.keys())}
    for variant in variants:
        det_scores = [per_detector_per_variant[d][variant] for d in per_detector_per_variant]
        ensemble = [
            sum(det_scores[k][i] for k in range(len(det_scores))) / len(det_scores)
            for i in range(n)
        ]
        summary[variant] = {
            "mean_p_ai_ensemble": statistics.fmean(ensemble),
            "asr_ensemble": sum(1 for p in ensemble if p < 0.5) / n,
            "asr_per_detector": {
                d: sum(1 for p in per_detector_per_variant[d][variant] if p < 0.5) / n
                for d in per_detector_per_variant
            },
            "mean_p_ai_per_detector": {
                d: statistics.fmean(per_detector_per_variant[d][variant])
                for d in per_detector_per_variant
            },
        }

    # Deltas: scrub vs raw, for each base/trained
    summary["delta_p_ai_scrub_on_base"] = (
        summary["base_scrub"]["mean_p_ai_ensemble"] - summary["base"]["mean_p_ai_ensemble"]
    )
    summary["delta_p_ai_scrub_on_trained"] = (
        summary["trained_scrub"]["mean_p_ai_ensemble"] - summary["trained"]["mean_p_ai_ensemble"]
    )

    return {
        "summary": summary,
        "per_detector_per_variant": per_detector_per_variant,
        "outputs": variants,
    }


def main():
    files = os.environ.get("EVAL_FILES", ":".join(DEFAULT_EVAL_FILES)).split(":")
    det_ids = os.environ.get("DETECTOR_IDS", ",".join(DEFAULT_DETECTORS)).split(",")
    files = [Path(f) for f in files if f]

    print(f"[main] processing {len(files)} eval file(s)", flush=True)
    detectors = load_detectors(det_ids)

    for path in files:
        if not path.exists():
            print(f"[skip] {path} not found", flush=True)
            continue
        result = evaluate(path, detectors)
        out_path = path.parent / (path.stem + ".scrub-eval.json")
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[done] -> {out_path}", flush=True)
        # Show summary
        s = result["summary"]
        print(json.dumps({
            "file": path.name,
            "delta_base_scrub": s["delta_p_ai_scrub_on_base"],
            "delta_trained_scrub": s["delta_p_ai_scrub_on_trained"],
            "asr": {v: s[v]["asr_ensemble"] for v in ["base", "base_scrub", "trained", "trained_scrub"]},
            "mean_p_ai": {v: s[v]["mean_p_ai_ensemble"] for v in ["base", "base_scrub", "trained", "trained_scrub"]},
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
