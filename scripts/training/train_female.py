#!/usr/bin/env python3
"""XTTS v2 fine-tuning on LJSpeech (female voice).

Optimized for p4d.24xlarge (8x A100 40GB).
Launch with: torchrun --nproc_per_node=8 scripts/training/train_female.py

Effective batch size: 2 * 16 * 8 = 256 (close to recommended 252).
"""

import os
import subprocess
import sys
import functools

import torch

# PyTorch 2.6+ defaults weights_only=True, but TTS checkpoints use pickle.
# Patch torch.load to default to weights_only=False for compatibility.
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# === Paths ===
DATASET_PATH = "/home/ec2-user/data/LJSpeech-1.1"
OUT_PATH = "/home/ec2-user/training_output/female"
S3_BUCKET = "s3://pm-tts-model/checkpoints/female/"

# === Logging ===
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# === Training Parameters (8x A100 40GB) ===
BATCH_SIZE = 2          # fits in 40GB VRAM
GRAD_ACUMM_STEPS = 16   # effective batch = 2 * 16 * 8 GPUs = 256
OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  # must be False for multi-GPU DDP
START_WITH_EVAL = True

# === Dataset ===
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path=DATASET_PATH,
    meta_file_train=os.path.join(DATASET_PATH, "metadata.csv"),
    language="en",
)
DATASETS_CONFIG_LIST = [config_dataset]

# === Pretrained Model Files ===
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

# Download pretrained files if needed (only on rank 0 to avoid race conditions)
_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
if _rank == 0:
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

# Simple file-based barrier: non-rank-0 processes wait for all files to exist
import time
if _rank != 0:
    _all_files = [DVAE_CHECKPOINT, MEL_NORM_FILE, TOKENIZER_FILE, XTTS_CHECKPOINT]
    while not all(os.path.isfile(f) for f in _all_files):
        time.sleep(2)

# === Speaker Reference ===
SPEAKER_REFERENCE = ["/home/ec2-user/TTS/reference_audio/female/LJ001-0002.wav"]
LANGUAGE = config_dataset.language


def main():
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 secs
        min_conditioning_length=66150,    # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,            # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        epochs=124,  # ~100K steps (124 * 811 steps/epoch ≈ 100,564)
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS v2.0 fine-tuning on LJSpeech - female voice (p4d.24xlarge 8xA100-40GB)",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=5000,           # save every 5K steps
        save_n_checkpoints=2,     # keep last 2 checkpoints
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "The quick brown fox jumps over the lazy dog near the bank of the river.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Resume from latest checkpoint if available
    _checkpoint_dir = os.path.join(OUT_PATH, "GPT_XTTS_v2.0_LJSpeech_FT-April-10-2026_08+47AM-3c5f7c31")
    _best_model = os.path.join(_checkpoint_dir, "best_model.pth")
    _restore = _best_model if os.path.isfile(_best_model) else None

    trainer = Trainer(
        TrainerArgs(
            restore_path=_restore,
            skip_train_epoch=False,
            start_with_eval=False,  # skip eval on resume
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

    # Post-training: sync checkpoints to S3 and shutdown
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        print("\n=== Training complete. Syncing to S3... ===")
        subprocess.run(
            ["aws", "s3", "sync", OUT_PATH, S3_BUCKET],
            check=False,
        )
        print("=== S3 sync complete. Shutting down instance... ===")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
