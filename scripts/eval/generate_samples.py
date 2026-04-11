#!/usr/bin/env python3
"""Generate fixed evaluation sample packs for XTTS checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class EvalPrompt:
    prompt_id: str
    category: str
    text: str


def load_prompt_pack(prompt_pack_path: Path | str) -> List[EvalPrompt]:
    path = Path(prompt_pack_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    prompts = raw["prompts"] if isinstance(raw, dict) else raw
    return [EvalPrompt(prompt_id=item["prompt_id"], category=item["category"], text=item["text"]) for item in prompts]


def _build_output_file(output_dir: Path, index: int, prompt: EvalPrompt) -> Path:
    return output_dir / f"{index:02d}_{prompt.prompt_id}.wav"


def generate_eval_pack(
    *,
    prompts: Sequence[EvalPrompt],
    output_path: Path | str,
    voice_name: str,
    checkpoint_label: str,
    synthesize_fn: Callable[[EvalPrompt, Path], None],
) -> Path:
    root = Path(output_path).resolve()
    sample_dir = root / voice_name / checkpoint_label
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, str]] = []

    for index, prompt in enumerate(prompts, start=1):
        output_file = _build_output_file(sample_dir, index, prompt)
        synthesize_fn(prompt, output_file)
        samples.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "text": prompt.text,
                "audio_file": str(output_file),
            }
        )

    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "voice_name": voice_name,
                "checkpoint_label": checkpoint_label,
                "samples": samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _synthesize_with_xtts(
    *,
    prompt: EvalPrompt,
    output_file: Path,
    checkpoint_path: Path,
    config_path: Path,
    tokenizer_path: Path,
    speaker_wav: Sequence[Path],
    language: str,
):
    try:
        from scripts.inference.xtts_runtime import synthesize_to_file
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scripts.inference.xtts_runtime is required for real sample generation. "
            "Implement the inference phase or call generate_eval_pack with a custom synthesize_fn."
        ) from exc

    synthesize_to_file(
        text=prompt.text,
        output_file=output_file,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        speaker_wav=speaker_wav,
        language=language,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation sample packs for a checkpoint.")
    parser.add_argument("--prompt-pack", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--config-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--speaker-wav", nargs="+")
    parser.add_argument("--language", default="en")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    prompts = load_prompt_pack(args.prompt_pack)

    if not all([args.checkpoint_path, args.config_path, args.tokenizer_path, args.speaker_wav]):
        raise SystemExit(
            "checkpoint-path, config-path, tokenizer-path, and speaker-wav are required for CLI synthesis."
        )

    def _synth(prompt: EvalPrompt, output_file: Path):
        _synthesize_with_xtts(
            prompt=prompt,
            output_file=output_file,
            checkpoint_path=Path(args.checkpoint_path).resolve(),
            config_path=Path(args.config_path).resolve(),
            tokenizer_path=Path(args.tokenizer_path).resolve(),
            speaker_wav=[Path(item).resolve() for item in args.speaker_wav],
            language=args.language,
        )

    return generate_eval_pack(
        prompts=prompts,
        output_path=args.output_path,
        voice_name=args.voice_name,
        checkpoint_label=args.checkpoint_label,
        synthesize_fn=_synth,
    )


if __name__ == "__main__":
    main()
