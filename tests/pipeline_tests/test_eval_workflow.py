import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.compare_runs import build_comparison_manifest
from scripts.eval.generate_samples import EvalPrompt, generate_eval_pack


class TestEvalWorkflow(unittest.TestCase):
    def test_generate_eval_pack_writes_manifest_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir).resolve() / "samples"
            prompts = [
                EvalPrompt(prompt_id="narration_01", category="narration", text="Launch day is finally here."),
                EvalPrompt(prompt_id="numbers_01", category="numbers", text="The invoice total is $149.50."),
            ]

            def fake_synthesizer(prompt: EvalPrompt, output_file: Path):
                output_file.write_text(prompt.text, encoding="utf-8")

            manifest_path = generate_eval_pack(
                prompts=prompts,
                output_path=output_path,
                voice_name="female",
                checkpoint_label="step_100k",
                synthesize_fn=fake_synthesizer,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["voice_name"], "female")
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertTrue((output_path / "female" / "step_100k" / "01_narration_01.wav").is_file())

    def test_build_comparison_manifest_aligns_prompt_ids_and_creates_review_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            baseline_dir.mkdir()
            candidate_dir.mkdir()

            baseline_manifest = baseline_dir / "manifest.json"
            candidate_manifest = candidate_dir / "manifest.json"
            baseline_manifest.write_text(
                json.dumps(
                    {
                        "voice_name": "female",
                        "samples": [
                            {
                                "prompt_id": "narration_01",
                                "category": "narration",
                                "text": "Launch day is finally here.",
                                "audio_file": str(baseline_dir / "narration.wav"),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_manifest.write_text(
                json.dumps(
                    {
                        "voice_name": "female",
                        "samples": [
                            {
                                "prompt_id": "narration_01",
                                "category": "narration",
                                "text": "Launch day is finally here.",
                                "audio_file": str(candidate_dir / "narration.wav"),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = build_comparison_manifest(
                baseline_manifest=baseline_manifest,
                candidate_manifest=candidate_manifest,
                output_path=root / "comparison",
                baseline_label="baseline",
                candidate_label="candidate",
            )

            comparison = json.loads(result["comparison_manifest"].read_text(encoding="utf-8"))
            with result["review_sheet"].open("r", encoding="utf-8") as handle:
                review_rows = list(csv.DictReader(handle))

            self.assertEqual(len(comparison["comparisons"]), 1)
            self.assertEqual(comparison["comparisons"][0]["baseline_label"], "baseline")
            self.assertEqual(review_rows[0]["winner"], "")
            self.assertEqual(review_rows[0]["prompt_id"], "narration_01")


if __name__ == "__main__":
    unittest.main()
