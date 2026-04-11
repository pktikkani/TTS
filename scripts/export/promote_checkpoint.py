#!/usr/bin/env python3
"""Promote a chosen checkpoint into a stable served voice directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _load_registry(registry_path: Path) -> Dict:
    if registry_path.is_file():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return {"voices": []}


def promote_checkpoint_artifacts(
    *,
    voice_name: str,
    checkpoint_path: Path | str,
    config_path: Path | str,
    tokenizer_path: Path | str,
    speaker_wav: Sequence[Path | str],
    registry_path: Path | str,
    model_root: Path | str,
    label: Optional[str] = None,
    language: str = "en",
) -> Dict[str, Path]:
    checkpoint = Path(checkpoint_path).resolve()
    config = Path(config_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    speaker_files = [Path(item).resolve() for item in speaker_wav]
    registry_file = Path(registry_path).resolve()
    root = Path(model_root).resolve()
    voice_dir = root / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    promoted_checkpoint = _copy_file(checkpoint, voice_dir / "best_model.pth")
    promoted_config = _copy_file(config, voice_dir / "config.json")
    promoted_tokenizer = _copy_file(tokenizer, voice_dir / "vocab.json")
    promoted_speaker_wav = []
    for source in speaker_files:
        promoted_speaker_wav.append(_copy_file(source, voice_dir / source.name))

    registry = _load_registry(registry_file)
    voices = registry.setdefault("voices", [])
    promoted_entry = {
        "voice_name": voice_name,
        "label": label or voice_name,
        "checkpoint_path": str(promoted_checkpoint),
        "config_path": str(promoted_config),
        "tokenizer_path": str(promoted_tokenizer),
        "speaker_wav": [str(item) for item in promoted_speaker_wav],
        "language": language,
        "normalize_text": True,
        "max_chunk_chars": 220,
    }

    existing_index = next((index for index, voice in enumerate(voices) if voice["voice_name"] == voice_name), None)
    if existing_index is None:
        voices.append(promoted_entry)
    else:
        voices[existing_index] = promoted_entry

    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return {"voice_dir": voice_dir, "registry_path": registry_file}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a checkpoint into the stable served voice registry.")
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--speaker-wav", nargs="+", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--label")
    parser.add_argument("--language", default="en")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    return promote_checkpoint_artifacts(
        voice_name=args.voice_name,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        speaker_wav=args.speaker_wav,
        registry_path=args.registry_path,
        model_root=args.model_root,
        label=args.label,
        language=args.language,
    )


if __name__ == "__main__":
    main()
