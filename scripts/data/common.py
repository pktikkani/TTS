#!/usr/bin/env python3
"""Common helpers for voice dataset preparation."""

from __future__ import annotations

import csv
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class PreparedRecord:
    audio_file: Path
    text: str
    speaker_name: str
    duration: float


def sanitize_text(text: str) -> str:
    return " ".join(text.split())


def count_words(text: str) -> int:
    return len([token for token in text.split(" ") if token])


def duration_from_wav(audio_file: Path) -> float:
    with wave.open(str(audio_file), "rb") as handle:
        frame_rate = handle.getframerate()
        num_frames = handle.getnframes()
    return num_frames / float(frame_rate)


def filter_records(
    records: Sequence[PreparedRecord],
    *,
    min_duration: float,
    max_duration: float,
    max_words: Optional[int] = None,
    max_characters: Optional[int] = None,
) -> List[PreparedRecord]:
    filtered: List[PreparedRecord] = []
    for record in records:
        if record.duration < min_duration or record.duration > max_duration:
            continue
        if max_words is not None and count_words(record.text) > max_words:
            continue
        if max_characters is not None and len(record.text) > max_characters:
            continue
        filtered.append(record)
    return filtered


def split_records(
    records: Sequence[PreparedRecord],
    *,
    eval_ratio: float,
    max_eval_items: int = 256,
) -> tuple[List[PreparedRecord], List[PreparedRecord]]:
    ordered = sorted(records, key=lambda item: (item.speaker_name, str(item.audio_file)))
    if len(ordered) <= 1 or eval_ratio <= 0:
        return list(ordered), []

    proposed_eval_count = int(round(len(ordered) * eval_ratio))
    eval_count = max(1, proposed_eval_count)
    eval_count = min(max_eval_items, eval_count, len(ordered) - 1)
    eval_records = ordered[:eval_count]
    train_records = ordered[eval_count:]
    return train_records, eval_records


def write_coqui_manifest(
    manifest_path: Path,
    records: Sequence[PreparedRecord],
    *,
    root_path: Optional[Path] = None,
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_file", "text", "speaker_name", "duration"],
            delimiter="|",
        )
        writer.writeheader()
        for record in records:
            audio_file = record.audio_file.resolve()
            if root_path is not None:
                try:
                    audio_value = str(audio_file.relative_to(root_path.resolve()))
                except ValueError:
                    audio_value = str(audio_file)
            else:
                audio_value = str(audio_file)
            writer.writerow(
                {
                    "audio_file": audio_value,
                    "text": record.text,
                    "speaker_name": record.speaker_name,
                    "duration": f"{record.duration:.4f}",
                }
            )
    return manifest_path


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
