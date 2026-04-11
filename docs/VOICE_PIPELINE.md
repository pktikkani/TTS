# Voice Pipeline

This repo now includes a full in-house English voice pipeline built around reusable XTTS training, data prep, evaluation, serving, and benchmarking.

Two practical quality notes:
- Use your existing 100k-step model as the bootstrap checkpoint for both voices instead of starting from the stock XTTS weights.
- Use 3 to 5 clean reference clips per voice when you promote a production checkpoint. The configs accept a list of reference files already; one clip is acceptable for a baseline but weaker for final evaluation and serving.

## 1. Prepare Datasets

Female LJSpeech:

```bash
python3 scripts/data/prepare_ljspeech.py \
  --dataset-path /home/ec2-user/data/LJSpeech-1.1 \
  --output-path /home/ec2-user/data/prepared/ljspeech
```

Male Hi-Fi TTS speaker 6097:

```bash
python3 scripts/data/prepare_hifi_tts.py \
  --manifest-path /home/ec2-user/data/hi_fi_tts/manifest.jsonl \
  --speaker-id 6097 \
  --output-path /home/ec2-user/data/prepared/hifi_tts_6097
```

## 2. Train Voices

Use your existing 100k-step model as the bootstrap checkpoint:

```bash
XTTS_BASELINE_CHECKPOINT=/path/to/your/100k-step/best_model.pth \
torchrun --nproc_per_node=8 scripts/training/train_female.py
```

Prepared female config:

```bash
XTTS_BASELINE_CHECKPOINT=/path/to/your/100k-step/best_model.pth \
torchrun --nproc_per_node=8 scripts/training/train_voice.py \
  --config configs/voices/female_ljspeech_prepared.json
```

Male config:

```bash
XTTS_BASELINE_CHECKPOINT=/path/to/your/100k-step/best_model.pth \
torchrun --nproc_per_node=8 scripts/training/train_male.py
```

If a newer checkpoint already exists under the configured output root for the same run name, training resumes from that checkpoint automatically. Otherwise the baseline checkpoint is used as the XTTS initializer.

## 3. Generate Eval Packs

```bash
python3 scripts/eval/generate_samples.py \
  --prompt-pack configs/eval/english_core.json \
  --output-path outputs/eval \
  --voice-name female \
  --checkpoint-label candidate \
  --checkpoint-path /path/to/best_model.pth \
  --config-path /path/to/config.json \
  --tokenizer-path /path/to/vocab.json \
  --speaker-wav /path/to/reference.wav
```

Compare baseline vs candidate:

```bash
python3 scripts/eval/compare_runs.py \
  --baseline-manifest outputs/eval/female/baseline/manifest.json \
  --candidate-manifest outputs/eval/female/candidate/manifest.json \
  --output-path outputs/eval/female/comparison \
  --baseline-label baseline \
  --candidate-label candidate
```

This writes:
- `comparison_manifest.json`
- `review_sheet.csv`

Use the review sheet to score naturalness, similarity, pronunciation, pacing, and expressiveness.

Run automated WER metrics on the same pack:

```bash
python3 scripts/eval/automated_metrics.py \
  --manifest-path outputs/eval/female/candidate/manifest.json \
  --output-path outputs/eval/female/candidate/metrics \
  --asr-model-name small.en
```

This writes:
- `wer_results.csv`
- `wer_summary.json`

## 4. Benchmark Against Vendors

Once you have exported or collected vendor samples for the same prompt pack:

```bash
python3 scripts/eval/benchmark_voices.py \
  --output-path outputs/benchmark/female \
  --manifest local=outputs/eval/female/candidate/manifest.json \
  --manifest elevenlabs=/path/to/elevenlabs_manifest.json \
  --manifest cartesia=/path/to/cartesia_manifest.json
```

This creates a unified benchmark review sheet across systems.

## 5. Promote A Checkpoint

Promote the chosen checkpoint into the stable served model registry:

```bash
python3 scripts/export/promote_checkpoint.py \
  --voice-name female \
  --checkpoint-path /path/to/best_model.pth \
  --config-path /path/to/config.json \
  --tokenizer-path /path/to/vocab.json \
  --speaker-wav /path/to/reference.wav \
  --registry-path configs/voices/registry.json \
  --model-root /home/ec2-user/models/voices \
  --label "Female Flagship"
```

## 6. Run The API

The API reads `configs/voices/registry.json` by default:

```bash
.venv-codex/bin/uvicorn scripts.serving.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /api/v1/voices`
- `POST /api/v1/synthesize`
- `WebSocket /api/v1/stream`

## 7. Text Frontend

The English normalizer lives at `scripts/text/normalize_en.py` and handles:
- currency
- dates
- times
- acronyms
- emails
- URLs

Use it in product-facing synthesis requests before inference when you need deterministic spoken forms.

## 8. XTTS Inference Defaults

The runtime defaults are tuned for XTTS v2 rather than generic TTS decoding:
- `temperature=0.75`
- `top_p=0.85`
- `repetition_penalty=10.0`
- `speed=1.0`

Only supported XTTS inference keys are passed through from the registry presets, so unsupported fields cannot crash evaluation runs.
