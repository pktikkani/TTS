import json
import tempfile
import unittest
from pathlib import Path

from scripts.training.voice_training import (
    apply_runtime_overrides,
    build_test_sentence_payloads,
    find_latest_training_checkpoint,
    load_baseline_checkpoint,
    load_voice_config,
    resolve_restore_path,
)


class TestVoiceTrainingConfig(unittest.TestCase):
    def test_load_voice_config_resolves_paths_relative_to_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            (root / "data").mkdir()
            (root / "refs").mkdir()
            (root / "checkpoints").mkdir()
            config_path = root / "voice.json"
            config_path.write_text(
                json.dumps(
                    {
                        "voice_name": "female_ljspeech",
                        "dataset_formatter": "ljspeech",
                        "dataset_path": "data",
                        "meta_file_train": "metadata.csv",
                        "output_path": "runs/female",
                        "reference_audio": ["refs/ref.wav"],
                        "baseline_checkpoint": "checkpoints/baseline.pth",
                        "test_sentences": ["One.", "Two."],
                    }
                ),
                encoding="utf-8",
            )

            config = load_voice_config(config_path)

            self.assertEqual(config.dataset_path, root / "data")
            self.assertEqual(config.output_path, root / "runs" / "female")
            self.assertEqual(config.reference_audio, [root / "refs" / "ref.wav"])
            self.assertEqual(config.baseline_checkpoint, root / "checkpoints" / "baseline.pth")

    def test_find_latest_training_checkpoint_prefers_latest_run_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir).resolve()
            older_run = output_root / "GPT_Female-April-10-2026_08+47AM-aaaa111"
            newer_run = output_root / "GPT_Female-April-11-2026_08+47AM-bbbb222"
            older_run.mkdir()
            newer_run.mkdir()
            (older_run / "checkpoint_10000.pth").write_text("", encoding="utf-8")
            (newer_run / "checkpoint_15000.pth").write_text("", encoding="utf-8")

            checkpoint = find_latest_training_checkpoint(output_root, "GPT_Female")

            self.assertEqual(checkpoint, newer_run / "checkpoint_15000.pth")

    def test_load_baseline_checkpoint_returns_existing_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            baseline = root / "baseline" / "best_model.pth"
            baseline.parent.mkdir()
            baseline.write_text("", encoding="utf-8")

            baseline_path = load_baseline_checkpoint(baseline)

            self.assertEqual(baseline_path, baseline)

    def test_resolve_restore_path_prefers_existing_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            output_root = root / "runs"
            run_dir = output_root / "GPT_Female-April-11-2026_08+47AM-bbbb222"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint_15000.pth"
            checkpoint.write_text("", encoding="utf-8")

            restore_path = resolve_restore_path(output_path=output_root, run_name="GPT_Female")

            self.assertEqual(restore_path, checkpoint)

    def test_build_test_sentence_payloads_uses_reference_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = (Path(tmpdir) / "ref.wav").resolve()
            reference.write_text("", encoding="utf-8")

            payloads = build_test_sentence_payloads(
                test_sentences=["Hello world.", "The time is 5 PM."],
                reference_audio=[reference],
                language="en",
            )

            self.assertEqual(payloads[0]["speaker_wav"], [reference])
            self.assertEqual(payloads[1]["language"], "en")

    def test_dataset_override_preserves_external_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            config_path = root / "voice.json"
            external_manifest = root / "prepared" / "metadata_train.csv"
            external_manifest.parent.mkdir(parents=True)
            external_manifest.write_text("", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "voice_name": "male_hifi",
                        "dataset_formatter": "coqui",
                        "dataset_path": "data/raw",
                        "meta_file_train": str(external_manifest),
                        "output_path": "runs/male",
                        "reference_audio": [],
                        "test_sentences": []
                    }
                ),
                encoding="utf-8",
            )

            config = load_voice_config(config_path)
            overridden = apply_runtime_overrides(config, dataset_path=str(root / "data" / "new_raw"))

            self.assertEqual(overridden.meta_file_train, external_manifest)


if __name__ == "__main__":
    unittest.main()
