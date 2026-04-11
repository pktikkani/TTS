#!/usr/bin/env python3
"""Prepare filtered coqui manifests from an LJSpeech dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts.data.common import (
    PreparedRecord,
    duration_from_wav,
    filter_records,
    sanitize_text,
    split_records,
    write_coqui_manifest,
)


def parse_ljspeech_records(dataset_path: Path, metadata_filename: str = "metadata.csv") -> List[PreparedRecord]:
    metadata_path = dataset_path / metadata_filename
    records: List[PreparedRecord] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cols = line.rstrip("\n").split("|")
            if len(cols) < 2:
                continue
            sample_id = cols[0].strip()
            text = sanitize_text(cols[2] if len(cols) > 2 else cols[1])
            audio_file = dataset_path / "wavs" / f"{sample_id}.wav"
            if not audio_file.is_file():
                continue
            records.append(
                PreparedRecord(
                    audio_file=audio_file,
                    text=text,
                    speaker_name="ljspeech",
                    duration=duration_from_wav(audio_file),
                )
            )
    return records


def prepare_ljspeech_dataset(
    *,
    dataset_path: Path | str,
    output_path: Path | str,
    metadata_filename: str = "metadata.csv",
    min_duration: float = 1.0,
    max_duration: float = 11.5,
    max_words: Optional[int] = 40,
    max_characters: Optional[int] = 240,
    eval_ratio: float = 0.02,
    max_eval_items: int = 256,
) -> Dict[str, Path]:
    dataset_root = Path(dataset_path).resolve()
    prepared_root = Path(output_path).resolve()
    raw_records = parse_ljspeech_records(dataset_root, metadata_filename=metadata_filename)
    kept_records = filter_records(
        raw_records,
        min_duration=min_duration,
        max_duration=max_duration,
        max_words=max_words,
        max_characters=max_characters,
    )
    train_records, eval_records = split_records(kept_records, eval_ratio=eval_ratio, max_eval_items=max_eval_items)
    train_manifest = write_coqui_manifest(prepared_root / "metadata_train.csv", train_records, root_path=dataset_root)
    eval_manifest = write_coqui_manifest(prepared_root / "metadata_eval.csv", eval_records, root_path=dataset_root)
    return {
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "kept_records": len(kept_records),
        "dropped_records": len(raw_records) - len(kept_records),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare filtered LJSpeech manifests for XTTS training.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metadata-filename", default="metadata.csv")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=11.5)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--max-characters", type=int, default=240)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--max-eval-items", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    return prepare_ljspeech_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        metadata_filename=args.metadata_filename,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_words=args.max_words,
        max_characters=args.max_characters,
        eval_ratio=args.eval_ratio,
        max_eval_items=args.max_eval_items,
    )


if __name__ == "__main__":
    main()
