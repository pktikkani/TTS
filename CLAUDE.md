# TTS Project — XTTS v2 Production System

## Overview

Building a production-grade, commercial TTS system using XTTS v2 on this Coqui TTS fork.

- **Repo:** `pktikkani/TTS` (fork of `coqui-ai/TTS`), branch: `dev`
- **Model:** XTTS v2 (GPT-based, best quality, near-human output)
- **Language:** English
- **Voices:** 1 female (LJSpeech) + 1 male (Hi-Fi TTS speaker 6097)
- **Use case:** Commercial product
- **Training infra:** AWS with NVIDIA GPUs
- **Dev machine:** M4 Max MacBook Pro, 128GB RAM

## Key Decisions

- XTTS v2 chosen over VITS for superior quality (voice cloning, multilingual, near-human)
- XTTS v2 code is MPL-2.0 (commercial OK). Pretrained weights are CPML (Coqui is defunct). Plan: use pretrained for fine-tuning now, retrain from scratch on own data if product scales.
- Datasets: LJSpeech (public domain), Hi-Fi TTS 6097 (CC-BY-4.0) — both commercial-friendly
- Single voice first, multi-speaker expansion later

## Architecture

XTTS v2 consists of:
- **GPT backbone** (30 layers, 1024 dim, 16 heads) — `TTS/tts/layers/xtts/gpt.py`
- **DVAE** (discrete VAE for audio tokenization) — `TTS/tts/layers/xtts/dvae.py` — frozen during training
- **HiFiGAN decoder** (mel to waveform @ 24kHz) — `TTS/tts/layers/xtts/hifigan_decoder.py` — frozen during training
- **PerceiverResampler** (XTTS v2 conditioning) — `TTS/tts/layers/xtts/perceiver_encoder.py`

Only the GPT component is fine-tuned. DVAE and HiFiGAN are frozen.

## Critical File Paths

| File | Role |
|------|------|
| `recipes/ljspeech/xtts_v2/train_gpt_xtts.py` | Reference training recipe (177 lines) |
| `TTS/tts/models/xtts.py` | Core model: inference(), inference_stream(), get_conditioning_latents() |
| `TTS/tts/layers/xtts/trainer/gpt_trainer.py` | Training wrapper: GPTTrainer, GPTTrainerConfig, GPTArgs |
| `TTS/tts/layers/xtts/trainer/dataset.py` | XTTSDataset: audio loading, tokenization, collation |
| `TTS/tts/datasets/formatters.py` | Dataset formatters — needs hifi_tts() addition for Phase 2 |
| `TTS/tts/layers/xtts/gpt.py` | GPT backbone implementation |
| `TTS/api.py` | High-level TTS API for inference |
| `TTS/server/server.py` | Flask REST API server (reference, replacing with FastAPI) |
| `TTS/demos/xtts_ft_demo/utils/formatter.py` | Auto-formatter using Faster-Whisper (reference) |

## Training Parameters (from recipe)

- Batch size: 3, grad accumulation: 84 (effective batch: 252)
- LR: 5e-6, AdamW (betas=[0.9, 0.96], eps=1e-8, weight_decay=1e-2)
- Scheduler: MultiStepLR (milestones at 50K*18, 150K*18, 300K*18 steps)
- Max audio length: 255,995 samples (~11.6s at 22050Hz)
- Max text length: 200 tokens
- Audio: 22050Hz input, 24000Hz output
- VRAM: ~40GB for batch_size=3 (needs A100 80GB)
- Pretrained weights auto-downloaded from coqui.gateway.scarf.sh

## Dataset Formats

**LJSpeech** (existing formatter):
- Pipe-delimited CSV: `id|phoneme_text|normalized_text`
- Audio: `wavs/<id>.wav` at 22050Hz
- ~13,100 clips, ~24 hours

**Hi-Fi TTS** (needs new formatter):
- JSONL manifest: `{"audio_filepath", "text", "duration", "text_normalized"}`
- Audio: FLAC at 44.1kHz (must resample to 22050Hz)
- Speaker 6097: ~30-40 hours, clean condition

## Implementation Plan

### Phase 0: Project Foundation (Days 1-2)
1. Create directory structure: `scripts/{data,training,serving,export}`, `configs/`, `docker/`, `reference_audio/{female,male}`, `tests/`
2. AWS setup: S3 bucket, IAM roles, VPC

### Phase 1: Female Voice — LJSpeech (Days 3-14)
1. **Data prep** (`c5.2xlarge`, ~$0.34/hr): Download LJSpeech, filter duration 1-11.5s, filter tokens < 190, validate integrity. Expected: ~12,500 clips.
2. **Training** (A100 80GB): Create `train_female.py` based on recipe. Save every 5K steps. ~20-40 hours for 50K steps. Cost: ~$150-240.
3. **Evaluation**: MCD < 6.0 dB, PESQ > 3.0, speaker similarity > 0.85, listening test with 50 sentences.
4. **Local validation**: Test on M4 Max — streaming TTFB < 500ms, RTF < 0.5x.

### Phase 2: Male Voice — Hi-Fi TTS (Days 15-28)
1. **Add hifi_tts formatter** to `TTS/tts/datasets/formatters.py`
2. **Data prep**: Download speaker 6097, resample 44.1kHz -> 22050Hz, filter duration/tokens.
3. **Training**: Same hyperparameters as Phase 1. Cost: ~$150-240.
4. **Combined testing**: A/B test both voices, verify no cross-speaker leakage.
5. **Cache speaker latents**: Pre-compute conditioning for both voices -> `speakers_xtts.pth`

### Phase 3: Production Serving (Days 29-45)
1. **FastAPI server**: `POST /api/v1/synthesize`, `WebSocket /api/v1/stream`, `GET /api/v1/voices`, `GET /health`
2. **Optimizations**: DeepSpeed (2-3x speedup), KV cache (default), FP16 (7GB -> 3.5GB VRAM)
3. **Docker**: CUDA 12.1 base, gunicorn + uvicorn
4. **AWS deployment**: `g5.xlarge` (A10G, ~$1/hr), ALB + auto-scaling
5. **Monitoring**: Prometheus metrics, structured logging, CloudWatch
6. **Load testing**: Target < 2s latency (50 words), streaming TTFB < 500ms

## Cost Summary

| Item | Cost |
|------|------|
| Phase 1 (female voice) | ~$235 |
| Phase 2 (male voice) | ~$290 |
| Phase 3 (serving dev) | ~$100 |
| **Total setup** | **~$625** |
| Production hosting | ~$750/mo (g5.xlarge 24/7) |

## AWS Instance Recommendations

| Use | Instance | GPU | VRAM | Cost/hr |
|-----|----------|-----|------|---------|
| Data prep | `c5.2xlarge` | None | - | $0.34 |
| Training | `p4d.24xlarge` | 8x A100 | 640GB | $32.77 |
| Training (single) | A100 spot | 1x A100 | 80GB | ~$5-8 |
| Serving (dev) | `g5.xlarge` | 1x A10G | 24GB | $1.01 |
| Serving (prod) | `g5.xlarge` | 1x A10G | 24GB | $1.01 |

## Current Status

**Phase:** 0 — Plan created, ready to start implementation on AWS.
**Next step:** Set up AWS infra, create directory structure, start Phase 1 data preparation.

## Commands

```bash
# Install the framework
pip install -e .[all]

# Run training (example)
CUDA_VISIBLE_DEVICES="0" python scripts/training/train_female.py

# Run inference server
python scripts/serving/fastapi_server.py

# Quick test with pretrained model
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --text "Hello" --speaker_wav ref.wav --language_idx en --out_path output.wav
```
