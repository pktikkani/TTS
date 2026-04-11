#!/usr/bin/env python3
"""XTTS v2 fine-tuning wrapper for the male Hi-Fi TTS voice."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.voice_training import main


if __name__ == "__main__":
    main(default_config_path=REPO_ROOT / "configs" / "voices" / "male_hifi_tts_6097.json")
