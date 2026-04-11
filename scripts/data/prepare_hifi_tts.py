#!/usr/bin/env python3
"""Prepare filtered coqui manifests from a Hi-Fi TTS manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts.data.common import (
    PreparedRecord,
    duration_from_wav,
    filter_records,
    read_jsonl,
    sanitize_text,
    split_records,
    write_coqui_manifest,
)


def parse_hifi_tts_records(manifest_path: Path, speaker_id: Optional[str] = None) -> List[PreparedRecord]:
    target_speaker = str(speaker_id) if speaker_id is not None else None
    records: List[PreparedRecord] = []
    for row in read_jsonl(manifest_path):
        row_speaker = str(row.get("speaker_id") or row.get("speaker") or row.get("speaker_name") or "")
        if target_speaker is not None and row_speaker != target_speaker:
            continue
        audio_file = Path(row["audio_filepath"]).resolve()
        if not audio_file.is_file():
            continue
        text = sanitize_text(row.get("text_normalized") or row.get("text") or "")
        if not text:
            continue
        duration = row.get("duration")
        if duration is None:
            if audio_file.suffix.lower() == ".wav":
                duration = duration_from_wav(audio_file)
            else:
                continue
        records.append(
            PreparedRecord(
                audio_file=audio_file,
                text=text,
                speaker_name=row_speaker or "hifi_tts",
                duration=float(duration),
            )
        )
    return records


def prepare_hifi_tts_dataset(
    *,
    manifest_path: Path | str,
    output_path: Path | str,
    speaker_id: str,
    min_duration: float = 1.0,
    max_duration: float = 11.5,
    max_words: Optional[int] = 45,
    max_characters: Optional[int] = 260,
    eval_ratio: float = 0.02,
    max_eval_items: int = 256,
) -> Dict[str, Path]:
    manifest_file = Path(manifest_path).resolve()
    prepared_root = Path(output_path).resolve()
    raw_records = parse_hifi_tts_records(manifest_file, speaker_id=speaker_id)
    kept_records = filter_records(
        raw_records,
        min_duration=min_duration,
        max_duration=max_duration,
        max_words=max_words,
        max_characters=max_characters,
    )
    train_records, eval_records = split_records(kept_records, eval_ratio=eval_ratio, max_eval_items=max_eval_items)
    train_manifest = write_coqui_manifest(prepared_root / "metadata_train.csv", train_records)
    eval_manifest = write_coqui_manifest(prepared_root / "metadata_eval.csv", eval_records)
    return {
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "kept_records": len(kept_records),
        "dropped_records": len(raw_records) - len(kept_records),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare filtered Hi-Fi TTS speaker manifests for XTTS training.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=11.5)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument("--max-characters", type=int, default=260)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--max-eval-items", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    return prepare_hifi_tts_dataset(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        speaker_id=args.speaker_id,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_words=args.max_words,
        max_characters=args.max_characters,
        eval_ratio=args.eval_ratio,
        max_eval_items=args.max_eval_items,
    )


if __name__ == "__main__":
    main()
