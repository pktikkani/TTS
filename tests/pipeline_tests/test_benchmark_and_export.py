import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.benchmark_voices import build_benchmark_matrix
from scripts.export.promote_checkpoint import promote_checkpoint_artifacts


class TestBenchmarkAndExport(unittest.TestCase):
    def test_build_benchmark_matrix_aligns_multiple_systems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            manifests = {}
            for label in ["local", "elevenlabs", "cartesia"]:
                manifest_path = root / f"{label}.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "voice_name": "female",
                            "samples": [
                                {
                                    "prompt_id": "narration_01",
                                    "category": "narration",
                                    "text": "Launch day is finally here.",
                                    "audio_file": str(root / f"{label}.wav"),
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests[label] = manifest_path

            result = build_benchmark_matrix(manifests=manifests, output_path=root / "benchmark")

            matrix = json.loads(result["benchmark_manifest"].read_text(encoding="utf-8"))
            with result["review_sheet"].open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(matrix["comparisons"]), 1)
            self.assertEqual(matrix["systems"], ["cartesia", "elevenlabs", "local"])
            self.assertIn("winner", rows[0])

    def test_promote_checkpoint_artifacts_copies_files_and_updates_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_dir = root / "run"
            source_dir.mkdir()
            checkpoint = source_dir / "best_model.pth"
            config = source_dir / "config.json"
            tokenizer = source_dir / "vocab.json"
            speaker_wav = source_dir / "ref.wav"
            for path in [checkpoint, config, tokenizer, speaker_wav]:
                path.write_text(path.name, encoding="utf-8")

            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps({"voices": []}), encoding="utf-8")
            model_root = root / "models"

            result = promote_checkpoint_artifacts(
                voice_name="female",
                checkpoint_path=checkpoint,
                config_path=config,
                tokenizer_path=tokenizer,
                speaker_wav=[speaker_wav],
                registry_path=registry_path,
                model_root=model_root,
                label="Female Flagship",
            )

            registry = json.loads(registry_path.read_text(encoding="utf-8"))

            self.assertTrue((result["voice_dir"] / "best_model.pth").is_file())
            self.assertEqual(registry["voices"][0]["voice_name"], "female")
            self.assertEqual(registry["voices"][0]["checkpoint_path"], str(result["voice_dir"] / "best_model.pth"))


if __name__ == "__main__":
    unittest.main()
