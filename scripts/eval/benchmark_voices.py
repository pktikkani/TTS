#!/usr/bin/env python3
"""Build a benchmark matrix across local and external TTS sample manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence


def _load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_benchmark_matrix(
    *,
    manifests: Dict[str, Path | str],
    output_path: Path | str,
) -> Dict[str, Path]:
    output_root = Path(output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    loaded = {label: _load_manifest(Path(path).resolve()) for label, path in manifests.items()}
    systems = sorted(loaded)
    prompt_sets = [
        {sample["prompt_id"] for sample in manifest["samples"]}
        for manifest in loaded.values()
    ]
    common_prompt_ids = sorted(set.intersection(*prompt_sets)) if prompt_sets else []

    comparisons = []
    for prompt_id in common_prompt_ids:
        system_samples = {}
        category = "general"
        text = ""
        for label, manifest in loaded.items():
            sample = next(item for item in manifest["samples"] if item["prompt_id"] == prompt_id)
            system_samples[label] = sample["audio_file"]
            category = sample.get("category", category)
            text = sample.get("text", text)
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "text": text,
                "systems": system_samples,
            }
        )

    benchmark_manifest = output_root / "benchmark_manifest.json"
    benchmark_manifest.write_text(
        json.dumps(
            {
                "voice_name": next(iter(loaded.values()))["voice_name"] if loaded else None,
                "systems": systems,
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    review_sheet = output_root / "benchmark_review.csv"
    fieldnames = ["prompt_id", "category", "text", *systems, "winner", "notes"]
    with review_sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(
                {
                    "prompt_id": comparison["prompt_id"],
                    "category": comparison["category"],
                    "text": comparison["text"],
                    **comparison["systems"],
                    "winner": "",
                    "notes": "",
                }
            )

    return {"benchmark_manifest": benchmark_manifest, "review_sheet": review_sheet}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-system benchmark sheet from sample manifests.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Pair in the form label=/path/to/manifest.json. Repeat for each system.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    manifests = {}
    for item in args.manifest:
        label, path = item.split("=", 1)
        manifests[label] = path
    return build_benchmark_matrix(manifests=manifests, output_path=args.output_path)


if __name__ == "__main__":
    main()
