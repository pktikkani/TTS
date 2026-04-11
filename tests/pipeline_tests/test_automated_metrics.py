import tempfile
import unittest
from pathlib import Path

from scripts.eval.automated_metrics import compute_wer, load_eval_manifest_rows


class TestAutomatedMetrics(unittest.TestCase):
    def test_compute_wer(self):
        self.assertEqual(compute_wer("hello world", "hello world"), 0.0)
        self.assertEqual(compute_wer("hello world", "hello"), 0.5)
        self.assertEqual(compute_wer("launch day is finally here", "launch day was finally here"), 0.2)

    def test_load_eval_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir).resolve() / "manifest.json"
            manifest.write_text(
                """{
  "voice_name": "female",
  "samples": [
    {
      "prompt_id": "narration_01",
      "category": "narration",
      "text": "Launch day is finally here.",
      "audio_file": "/tmp/sample.wav"
    }
  ]
}""",
                encoding="utf-8",
            )

            rows = load_eval_manifest_rows(manifest)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_id"], "narration_01")


if __name__ == "__main__":
    unittest.main()
