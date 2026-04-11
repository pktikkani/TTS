#!/usr/bin/env python3
"""Shared XTTS voice training helpers and entrypoint."""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_LR_SCHEDULER_PARAMS = {
    "milestones": [30000, 70000, 90000],
    "gamma": 0.5,
    "last_epoch": -1,
}


@dataclass(frozen=True)
class VoiceTrainingConfig:
    voice_name: str
    dataset_formatter: str
    dataset_path: Path
    meta_file_train: Path
    output_path: Path
    reference_audio: List[Path]
    dataset_name: str = "voice_dataset"
    language: str = "en"
    meta_file_val: Optional[Path] = None
    run_name: str = "GPT_XTTS_v2.0_FT"
    project_name: str = "XTTS_trainer"
    run_description: str = "GPT XTTS voice fine-tuning"
    dashboard_logger: str = "tensorboard"
    logger_uri: Optional[str] = None
    batch_size: int = 2
    grad_accum_steps: int = 16
    epochs: int = 124
    batch_group_size: int = 48
    eval_batch_size: Optional[int] = None
    num_loader_workers: int = 8
    eval_split_max_size: int = 256
    eval_split_size: float = 0.01
    print_step: int = 50
    plot_step: int = 100
    log_model_step: int = 1000
    save_step: int = 5000
    save_n_checkpoints: int = 2
    print_eval: bool = False
    optimizer: str = "AdamW"
    optimizer_wd_only_on_weights: bool = False
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}
    )
    lr: float = 5e-6
    lr_scheduler: str = "MultiStepLR"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LR_SCHEDULER_PARAMS))
    start_with_eval: bool = True
    max_conditioning_length: int = 132300
    min_conditioning_length: int = 66150
    max_wav_length: int = 255995
    max_text_length: int = 200
    sample_rate: int = 22050
    dvae_sample_rate: int = 22050
    output_sample_rate: int = 24000
    baseline_checkpoint: Optional[Path] = None
    test_sentences: List[Any] = field(default_factory=list)
    s3_bucket: Optional[str] = None
    sync_to_s3_after_train: bool = False


def _resolve_path(config_root: Path, raw_value: Optional[str]) -> Optional[Path]:
    if raw_value in (None, ""):
        return None
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (config_root / path).resolve()


def _resolve_dataset_file(dataset_path: Path, raw_value: Optional[str]) -> Optional[Path]:
    if raw_value in (None, ""):
        return None
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (dataset_path / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_voice_config(config_path: Path | str) -> VoiceTrainingConfig:
    config_path = Path(config_path).resolve()
    raw = _load_json(config_path)
    config_root = config_path.parent
    dataset_path = _resolve_path(config_root, raw["dataset_path"])
    assert dataset_path is not None

    meta_file_train = _resolve_dataset_file(dataset_path, raw["meta_file_train"])
    assert meta_file_train is not None
    meta_file_val = _resolve_dataset_file(dataset_path, raw.get("meta_file_val"))
    output_path = _resolve_path(config_root, raw["output_path"])
    assert output_path is not None

    return VoiceTrainingConfig(
        voice_name=raw["voice_name"],
        dataset_formatter=raw["dataset_formatter"],
        dataset_name=raw.get("dataset_name", raw["voice_name"]),
        dataset_path=dataset_path,
        meta_file_train=meta_file_train,
        meta_file_val=meta_file_val,
        output_path=output_path,
        language=raw.get("language", "en"),
        run_name=raw.get("run_name", "GPT_XTTS_v2.0_FT"),
        project_name=raw.get("project_name", "XTTS_trainer"),
        run_description=raw.get("run_description", "GPT XTTS voice fine-tuning"),
        dashboard_logger=raw.get("dashboard_logger", "tensorboard"),
        logger_uri=raw.get("logger_uri"),
        batch_size=raw.get("batch_size", 2),
        grad_accum_steps=raw.get("grad_accum_steps", 16),
        epochs=raw.get("epochs", 124),
        batch_group_size=raw.get("batch_group_size", 48),
        eval_batch_size=raw.get("eval_batch_size"),
        num_loader_workers=raw.get("num_loader_workers", 8),
        eval_split_max_size=raw.get("eval_split_max_size", 256),
        eval_split_size=raw.get("eval_split_size", 0.01),
        print_step=raw.get("print_step", 50),
        plot_step=raw.get("plot_step", 100),
        log_model_step=raw.get("log_model_step", 1000),
        save_step=raw.get("save_step", 5000),
        save_n_checkpoints=raw.get("save_n_checkpoints", 2),
        print_eval=raw.get("print_eval", False),
        optimizer=raw.get("optimizer", "AdamW"),
        optimizer_wd_only_on_weights=raw.get("optimizer_wd_only_on_weights", False),
        optimizer_params=raw.get(
            "optimizer_params",
            {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        ),
        lr=raw.get("lr", 5e-6),
        lr_scheduler=raw.get("lr_scheduler", "MultiStepLR"),
        lr_scheduler_params=raw.get("lr_scheduler_params", dict(DEFAULT_LR_SCHEDULER_PARAMS)),
        start_with_eval=raw.get("start_with_eval", True),
        max_conditioning_length=raw.get("max_conditioning_length", 132300),
        min_conditioning_length=raw.get("min_conditioning_length", 66150),
        max_wav_length=raw.get("max_wav_length", 255995),
        max_text_length=raw.get("max_text_length", 200),
        sample_rate=raw.get("sample_rate", 22050),
        dvae_sample_rate=raw.get("dvae_sample_rate", 22050),
        output_sample_rate=raw.get("output_sample_rate", 24000),
        reference_audio=[_resolve_path(config_root, item) for item in raw.get("reference_audio", [])],
        baseline_checkpoint=_resolve_path(config_root, raw.get("baseline_checkpoint")),
        test_sentences=raw.get("test_sentences", []),
        s3_bucket=raw.get("s3_bucket"),
        sync_to_s3_after_train=raw.get("sync_to_s3_after_train", False),
    )


def apply_runtime_overrides(
    config: VoiceTrainingConfig,
    *,
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    baseline_checkpoint: Optional[str] = None,
    reference_audio: Optional[Sequence[str]] = None,
) -> VoiceTrainingConfig:
    updates: Dict[str, Any] = {}
    if dataset_path:
        dataset_path_obj = Path(dataset_path).resolve()
        updates["dataset_path"] = dataset_path_obj
        if config.meta_file_train.parent == config.dataset_path:
            updates["meta_file_train"] = dataset_path_obj / config.meta_file_train.name
        if config.meta_file_val is not None and config.meta_file_val.parent == config.dataset_path:
            updates["meta_file_val"] = dataset_path_obj / config.meta_file_val.name
    if output_path:
        updates["output_path"] = Path(output_path).resolve()
    if baseline_checkpoint:
        updates["baseline_checkpoint"] = Path(baseline_checkpoint).resolve()
    if reference_audio:
        updates["reference_audio"] = [Path(item).resolve() for item in reference_audio]
    return replace(config, **updates)


def load_baseline_checkpoint(baseline_checkpoint: Optional[Path | str]) -> Optional[Path]:
    if baseline_checkpoint in (None, ""):
        return None
    path = Path(baseline_checkpoint).resolve()
    if path.is_dir():
        latest_checkpoint = _pick_preferred_checkpoint(path)
        return latest_checkpoint
    return path if path.is_file() else None


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"checkpoint_(\d+)\.pth$", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.stat().st_mtime


def _pick_preferred_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoint_candidates = sorted(run_dir.glob("checkpoint_*.pth"), key=_checkpoint_sort_key)
    if checkpoint_candidates:
        return checkpoint_candidates[-1]
    best_model = run_dir / "best_model.pth"
    if best_model.is_file():
        return best_model
    other_candidates = sorted(run_dir.glob("*.pth"), key=lambda item: item.stat().st_mtime)
    if other_candidates:
        return other_candidates[-1]
    return None


def _iter_run_directories(output_path: Path, run_name: str) -> Iterable[Path]:
    if not output_path.exists():
        return []
    candidates = []
    if output_path.is_dir():
        candidates.append(output_path)
        for child in output_path.iterdir():
            if child.is_dir() and (child.name == run_name or child.name.startswith(f"{run_name}-")):
                candidates.append(child)
    return candidates


def find_latest_training_checkpoint(output_path: Path | str, run_name: str) -> Optional[Path]:
    output_root = Path(output_path).resolve()
    checkpoints: List[Path] = []
    for run_dir in _iter_run_directories(output_root, run_name):
        checkpoint = _pick_preferred_checkpoint(run_dir)
        if checkpoint is not None:
            checkpoints.append(checkpoint)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item.stat().st_mtime)


def resolve_restore_path(
    output_path: Path | str,
    run_name: str,
    explicit_restore_path: Optional[Path | str] = None,
) -> Optional[Path]:
    if explicit_restore_path not in (None, ""):
        return Path(explicit_restore_path).resolve()
    return find_latest_training_checkpoint(output_path, run_name)


def build_test_sentence_payloads(
    test_sentences: Sequence[Any],
    reference_audio: Sequence[Path | str],
    language: str,
) -> List[Dict[str, Any]]:
    speaker_wav = [Path(item).resolve() for item in reference_audio]
    payloads: List[Dict[str, Any]] = []
    for item in test_sentences:
        if isinstance(item, str):
            payloads.append({"text": item, "speaker_wav": speaker_wav, "language": language})
            continue
        payload = dict(item)
        payload.setdefault("speaker_wav", speaker_wav)
        payload.setdefault("language", language)
        payloads.append(payload)
    return payloads


def _patch_torch_load():
    import torch

    if getattr(torch.load, "__name__", "") == "_patched_torch_load":
        return torch

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    return torch


def _download_pretrained_files(checkpoints_out_path: Path) -> Dict[str, Path]:
    from TTS.utils.manage import ModelManager

    checkpoints_out_path.mkdir(parents=True, exist_ok=True)
    dvae_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    mel_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    tokenizer_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    xtts_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    config_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    files = {
        "dvae_checkpoint": checkpoints_out_path / Path(dvae_link).name,
        "mel_norm_file": checkpoints_out_path / Path(mel_link).name,
        "tokenizer_file": checkpoints_out_path / Path(tokenizer_link).name,
        "xtts_checkpoint": checkpoints_out_path / Path(xtts_link).name,
        "xtts_config_file": checkpoints_out_path / Path(config_link).name,
    }

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        if not files["dvae_checkpoint"].is_file() or not files["mel_norm_file"].is_file():
            ModelManager._download_model_files([mel_link, dvae_link], str(checkpoints_out_path), progress_bar=True)
        if (
            not files["tokenizer_file"].is_file()
            or not files["xtts_checkpoint"].is_file()
            or not files["xtts_config_file"].is_file()
        ):
            ModelManager._download_model_files(
                [tokenizer_link, xtts_link, config_link],
                str(checkpoints_out_path),
                progress_bar=True,
            )
    else:
        required = list(files.values())
        while not all(item.is_file() for item in required):
            time.sleep(2)

    return files


def train_voice(
    config: VoiceTrainingConfig,
    *,
    explicit_restore_path: Optional[Path | str] = None,
    explicit_baseline_checkpoint: Optional[Path | str] = None,
):
    _patch_torch_load()

    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
        XttsAudioConfig,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)
    pretrained_files = _download_pretrained_files(config.output_path / "XTTS_v2.0_original_model_files")
    initializer_checkpoint = (
        load_baseline_checkpoint(explicit_baseline_checkpoint)
        or load_baseline_checkpoint(config.baseline_checkpoint)
        or pretrained_files["xtts_checkpoint"]
    )
    restore_path = resolve_restore_path(config.output_path, config.run_name, explicit_restore_path)

    dataset_config = BaseDatasetConfig(
        formatter=config.dataset_formatter,
        dataset_name=config.dataset_name,
        path=str(config.dataset_path),
        meta_file_train=str(config.meta_file_train),
        meta_file_val=str(config.meta_file_val) if config.meta_file_val is not None else None,
        language=config.language,
    )

    model_args = GPTArgs(
        max_conditioning_length=config.max_conditioning_length,
        min_conditioning_length=config.min_conditioning_length,
        debug_loading_failures=False,
        max_wav_length=config.max_wav_length,
        max_text_length=config.max_text_length,
        mel_norm_file=str(pretrained_files["mel_norm_file"]),
        dvae_checkpoint=str(pretrained_files["dvae_checkpoint"]),
        xtts_checkpoint=str(initializer_checkpoint),
        tokenizer_file=str(pretrained_files["tokenizer_file"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=config.sample_rate,
        dvae_sample_rate=config.dvae_sample_rate,
        output_sample_rate=config.output_sample_rate,
    )
    trainer_config = GPTTrainerConfig(
        epochs=config.epochs,
        output_path=str(config.output_path),
        model_args=model_args,
        run_name=config.run_name,
        project_name=config.project_name,
        run_description=config.run_description,
        dashboard_logger=config.dashboard_logger,
        logger_uri=config.logger_uri,
        audio=audio_config,
        batch_size=config.batch_size,
        batch_group_size=config.batch_group_size,
        eval_batch_size=config.eval_batch_size or config.batch_size,
        num_loader_workers=config.num_loader_workers,
        eval_split_max_size=config.eval_split_max_size,
        print_step=config.print_step,
        plot_step=config.plot_step,
        log_model_step=config.log_model_step,
        save_step=config.save_step,
        save_n_checkpoints=config.save_n_checkpoints,
        save_checkpoints=True,
        print_eval=config.print_eval,
        optimizer=config.optimizer,
        optimizer_wd_only_on_weights=config.optimizer_wd_only_on_weights,
        optimizer_params=config.optimizer_params,
        lr=config.lr,
        lr_scheduler=config.lr_scheduler,
        lr_scheduler_params=config.lr_scheduler_params,
        test_sentences=build_test_sentence_payloads(
            config.test_sentences,
            config.reference_audio,
            config.language,
        ),
    )

    model = GPTTrainer.init_from_config(trainer_config)
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=config.meta_file_val is None,
        eval_split_max_size=trainer_config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=str(restore_path) if restore_path is not None else None,
            skip_train_epoch=False,
            start_with_eval=config.start_with_eval and restore_path is None,
            grad_accum_steps=config.grad_accum_steps,
        ),
        trainer_config,
        output_path=str(config.output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    if config.sync_to_s3_after_train and config.s3_bucket:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank == 0:
            subprocess.run(["aws", "s3", "sync", str(config.output_path), config.s3_bucket], check=False)

    return {
        "baseline_checkpoint": initializer_checkpoint,
        "restore_path": restore_path,
        "output_path": config.output_path,
        "xtts_config_file": pretrained_files["xtts_config_file"],
        "tokenizer_file": pretrained_files["tokenizer_file"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XTTS voice with shared config.")
    parser.add_argument("--config", help="Path to a voice JSON config.")
    parser.add_argument("--dataset-path", help="Override dataset root path.")
    parser.add_argument("--output-path", help="Override output root path.")
    parser.add_argument("--baseline-checkpoint", help="Bootstrap checkpoint for transfer learning.")
    parser.add_argument("--restore-path", help="Explicit checkpoint to resume from.")
    parser.add_argument(
        "--reference-audio",
        nargs="+",
        help="Override speaker reference wavs used for eval sentence synthesis.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, *, default_config_path: Optional[Path] = None):
    args = parse_args(argv)
    config_path = Path(args.config).resolve() if args.config else default_config_path
    if config_path is None:
        raise SystemExit("A voice config path is required.")
    config = load_voice_config(config_path)
    config = apply_runtime_overrides(
        config,
        dataset_path=args.dataset_path or os.environ.get("XTTS_DATASET_PATH"),
        output_path=args.output_path or os.environ.get("XTTS_OUTPUT_PATH"),
        baseline_checkpoint=args.baseline_checkpoint or os.environ.get("XTTS_BASELINE_CHECKPOINT"),
        reference_audio=args.reference_audio,
    )
    return train_voice(
        config,
        explicit_restore_path=args.restore_path or os.environ.get("XTTS_RESTORE_PATH"),
        explicit_baseline_checkpoint=args.baseline_checkpoint or os.environ.get("XTTS_BASELINE_CHECKPOINT"),
    )


if __name__ == "__main__":
    main()
