#!/usr/bin/env python3
"""Create side-by-side checkpoint comparison manifests and review sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence


def _load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_manifest(
    *,
    baseline_manifest: Path | str,
    candidate_manifest: Path | str,
    output_path: Path | str,
    baseline_label: str,
    candidate_label: str,
) -> Dict[str, Path]:
    baseline_path = Path(baseline_manifest).resolve()
    candidate_path = Path(candidate_manifest).resolve()
    output_root = Path(output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline = _load_manifest(baseline_path)
    candidate = _load_manifest(candidate_path)

    baseline_samples = {sample["prompt_id"]: sample for sample in baseline["samples"]}
    candidate_samples = {sample["prompt_id"]: sample for sample in candidate["samples"]}
    prompt_ids = sorted(set(baseline_samples) & set(candidate_samples))

    comparisons = []
    for prompt_id in prompt_ids:
        base_sample = baseline_samples[prompt_id]
        candidate_sample = candidate_samples[prompt_id]
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "category": base_sample.get("category", candidate_sample.get("category", "general")),
                "text": base_sample.get("text", candidate_sample.get("text", "")),
                "baseline_label": baseline_label,
                "baseline_audio_file": base_sample["audio_file"],
                "candidate_label": candidate_label,
                "candidate_audio_file": candidate_sample["audio_file"],
            }
        )

    comparison_manifest = output_root / "comparison_manifest.json"
    comparison_manifest.write_text(
        json.dumps(
            {
                "voice_name": baseline.get("voice_name", candidate.get("voice_name")),
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    review_sheet = output_root / "review_sheet.csv"
    with review_sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "category",
                "text",
                "baseline_label",
                "baseline_audio_file",
                "candidate_label",
                "candidate_audio_file",
                "naturalness",
                "similarity",
                "pronunciation",
                "pacing",
                "expressiveness",
                "winner",
                "notes",
            ],
        )
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(
                {
                    **comparison,
                    "naturalness": "",
                    "similarity": "",
                    "pronunciation": "",
                    "pacing": "",
                    "expressiveness": "",
                    "winner": "",
                    "notes": "",
                }
            )

    return {"comparison_manifest": comparison_manifest, "review_sheet": review_sheet}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two checkpoint sample manifests.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    return build_comparison_manifest(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        output_path=args.output_path,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )


if __name__ == "__main__":
    main()
