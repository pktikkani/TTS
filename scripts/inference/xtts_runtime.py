#!/usr/bin/env python3
"""XTTS runtime helpers, voice registry, and local synthesis service."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from scripts.text.normalize_en import normalize_for_tts


@dataclass(frozen=True)
class VoiceSpec:
    voice_name: str
    label: str
    checkpoint_path: Path
    config_path: Path
    tokenizer_path: Path
    speaker_wav: List[Path]
    language: str = "en"
    normalize_text: bool = True
    max_chunk_chars: int = 220
    inference_preset: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 5.0,
            "speed": 1.0,
        }
    )


@dataclass(frozen=True)
class SynthesizeRequest:
    voice_name: str
    text: str
    output_dir: Optional[Path] = None
    language: Optional[str] = None
    normalize: bool = True


def load_voice_registry(registry_path: Path | str) -> Dict[str, VoiceSpec]:
    path = Path(registry_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    voices = raw["voices"] if isinstance(raw, dict) else raw
    registry: Dict[str, VoiceSpec] = {}
    for voice in voices:
        voice_spec = VoiceSpec(
            voice_name=voice["voice_name"],
            label=voice.get("label", voice["voice_name"]),
            checkpoint_path=Path(voice["checkpoint_path"]).resolve(),
            config_path=Path(voice["config_path"]).resolve(),
            tokenizer_path=Path(voice["tokenizer_path"]).resolve(),
            speaker_wav=[Path(item).resolve() for item in voice.get("speaker_wav", [])],
            language=voice.get("language", "en"),
            normalize_text=voice.get("normalize_text", True),
            max_chunk_chars=voice.get("max_chunk_chars", 220),
            inference_preset=voice.get(
                "inference_preset",
                {
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "repetition_penalty": 5.0,
                    "speed": 1.0,
                },
            ),
        )
        registry[voice_spec.voice_name] = voice_spec
    return registry


def chunk_text(text: str, max_chars: int = 220) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    sentences = []
    current = ""
    for part in _split_sentences(text):
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            sentences.append(current.strip())
        if len(part) <= max_chars:
            current = part
        else:
            sentences.extend(_split_long_fragment(part, max_chars))
            current = ""
    if current:
        sentences.append(current.strip())
    return [sentence for sentence in sentences if sentence]


def _split_sentences(text: str) -> List[str]:
    chunks = []
    buffer = ""
    for char in text:
        buffer += char
        if char in ".!?":
            chunks.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def _split_long_fragment(text: str, max_chars: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


class XttsRuntime:
    """Lazy XTTS loader for local inference."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_voice(self, voice: VoiceSpec) -> Dict[str, Any]:
        cached = self._cache.get(voice.voice_name)
        if cached is not None:
            return cached

        import torch
        import torchaudio
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(str(voice.config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(voice.checkpoint_path),
            vocab_path=str(voice.tokenizer_path),
            use_deepspeed=False,
        )
        if torch.cuda.is_available():
            model.cuda()

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(item) for item in voice.speaker_wav]
        )

        cached = {
            "config": config,
            "model": model,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "torchaudio": torchaudio,
            "torch": torch,
        }
        self._cache[voice.voice_name] = cached
        return cached

    def synthesize(self, request: SynthesizeRequest, voice: VoiceSpec) -> Dict[str, Any]:
        runtime = self._load_voice(voice)
        model = runtime["model"]
        torch = runtime["torch"]
        torchaudio = runtime["torchaudio"]

        normalized_text = normalize_for_tts(request.text) if request.normalize and voice.normalize_text else request.text
        chunks = chunk_text(normalized_text, max_chars=voice.max_chunk_chars)

        wav_segments = []
        for chunk in chunks:
            output = model.inference(
                chunk,
                request.language or voice.language,
                runtime["gpt_cond_latent"],
                runtime["speaker_embedding"],
                enable_text_splitting=False,
                **voice.inference_preset,
            )
            wav_segments.append(torch.tensor(output["wav"]))

        full_wav = torch.cat(wav_segments, dim=0).unsqueeze(0)
        output_dir = request.output_dir or Path(tempfile.mkdtemp(prefix=f"{voice.voice_name}_tts_"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{voice.voice_name}.wav"
        torchaudio.save(str(output_file), full_wav.cpu(), voice.inference_preset.get("sample_rate", 24000))
        return {
            "voice_name": voice.voice_name,
            "normalized_text": normalized_text,
            "audio_file": str(output_file),
            "chunks": chunks,
        }

    def stream(self, request: SynthesizeRequest, voice: VoiceSpec) -> Iterable[Dict[str, Any]]:
        normalized_text = normalize_for_tts(request.text) if request.normalize and voice.normalize_text else request.text
        for index, chunk in enumerate(chunk_text(normalized_text, max_chars=voice.max_chunk_chars)):
            yield {"index": index, "text": chunk, "voice_name": voice.voice_name}


class LocalXTTSService:
    def __init__(self, registry_path: Path | str):
        self.registry_path = Path(registry_path).resolve()
        self.registry = load_voice_registry(self.registry_path)
        self.runtime = XttsRuntime()

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "voices_loaded": len(self.registry)}

    def list_voices(self) -> List[Dict[str, Any]]:
        return [
            {
                "voice_name": voice.voice_name,
                "label": voice.label,
                "language": voice.language,
            }
            for voice in self.registry.values()
        ]

    def synthesize(self, request: SynthesizeRequest) -> Dict[str, Any]:
        voice = self.registry[request.voice_name]
        return self.runtime.synthesize(request, voice)

    def stream(self, request: SynthesizeRequest) -> Iterable[Dict[str, Any]]:
        voice = self.registry[request.voice_name]
        return self.runtime.stream(request, voice)


def synthesize_to_file(
    *,
    text: str,
    output_file: Path,
    checkpoint_path: Path,
    config_path: Path,
    tokenizer_path: Path,
    speaker_wav: List[Path],
    language: str = "en",
):
    voice = VoiceSpec(
        voice_name="ad_hoc",
        label="Ad Hoc",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        speaker_wav=speaker_wav,
        language=language,
    )
    runtime = XttsRuntime()
    result = runtime.synthesize(
        SynthesizeRequest(voice_name=voice.voice_name, text=text, output_dir=output_file.parent, language=language),
        voice,
    )
    Path(result["audio_file"]).replace(output_file)
    return output_file
