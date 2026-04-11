#!/usr/bin/env python3
"""Automated evaluation metrics for XTTS sample manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def load_eval_manifest_rows(manifest_path: Path | str) -> List[Dict]:
    manifest = json.loads(Path(manifest_path).resolve().read_text(encoding="utf-8"))
    return manifest.get("samples", [])


def _tokenize_for_wer(text: str) -> List[str]:
    return " ".join(text.lower().split()).split(" ")


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = _tokenize_for_wer(reference)
    hyp = _tokenize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(ref)


def transcribe_with_faster_whisper(audio_file: Path, model_name: str = "small.en") -> str:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "faster_whisper is required for automated transcription metrics. Install it in the training/eval environment."
        ) from exc

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(str(audio_file), language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()


def run_wer_eval(
    *,
    manifest_path: Path | str,
    output_path: Path | str,
    asr_model_name: str = "small.en",
) -> Dict[str, Path | float]:
    rows = load_eval_manifest_rows(manifest_path)
    output_root = Path(output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "wer_results.csv"
    total_wer = 0.0

    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_id", "reference_text", "transcript", "wer", "audio_file"],
        )
        writer.writeheader()
        for row in rows:
            transcript = transcribe_with_faster_whisper(Path(row["audio_file"]).resolve(), model_name=asr_model_name)
            wer = compute_wer(row["text"], transcript)
            total_wer += wer
            writer.writerow(
                {
                    "prompt_id": row["prompt_id"],
                    "reference_text": row["text"],
                    "transcript": transcript,
                    "wer": f"{wer:.4f}",
                    "audio_file": row["audio_file"],
                }
            )

    average_wer = total_wer / len(rows) if rows else 0.0
    summary_json = output_root / "wer_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "manifest_path": str(Path(manifest_path).resolve()),
                "average_wer": average_wer,
                "sample_count": len(rows),
                "results_csv": str(results_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"results_csv": results_csv, "summary_json": summary_json, "average_wer": average_wer}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated WER metrics on a generated eval manifest.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--asr-model-name", default="small.en")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    return run_wer_eval(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        asr_model_name=args.asr_model_name,
    )


if __name__ == "__main__":
    main()
