---
name: TTS Project Goal and Setup
description: Building a commercial production-grade TTS model using Coqui TTS framework (XTTS v2), AWS GPUs, English, single voice first
type: project
---
Goal: Build a production-grade TTS model for **commercial use** using the Coqui TTS framework.

**Setup:**
- Repo: pktikkani/TTS (fork of coqui-ai/TTS), branch: dev
- Local machine: M4 Max MacBook Pro, 128GB RAM — dev/debug/small experiments
- Training infra: AWS with NVIDIA GPUs (p4d.24xlarge, 8x A100 40GB)
- The repo contains the Coqui TTS framework source code

**Decisions (2026-04-10):**
- Language: English
- Model: **XTTS v2** — best quality, near-human output, voice cloning capable
- License strategy: XTTS v2 code is MPL-2.0 (commercial OK). Pretrained weights are CPML (non-commercial) but Coqui company is defunct/unenforced. Plan: use pretrained weights for fine-tuning now, retrain from scratch on own data if product scales commercially.
- Voice: Single voice first → multi-speaker later
- Target: 1 great male voice + 1 great female voice, human-like quality
- **Commercial product** — will own weights long-term
- Datasets: Public domain / commercial-use-OK only
  - Female: LJSpeech (public domain, 24hrs)
  - Male: Hi-Fi TTS speaker 6097 (CC-BY-4.0)

**Phases:**
1. Single female voice (LJSpeech + XTTS v2 fine-tune) — prove the pipeline
2. Single male voice (Hi-Fi TTS 6097 + XTTS v2)
3. Multi-speaker expansion
4. (Future) Retrain from scratch with own data for fully owned commercial weights
