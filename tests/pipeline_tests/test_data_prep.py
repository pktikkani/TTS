import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.data.prepare_hifi_tts import prepare_hifi_tts_dataset
from scripts.data.prepare_ljspeech import prepare_ljspeech_dataset


def _write_wav(path: Path, duration_seconds: float, sample_rate: int = 22050):
    num_frames = max(1, int(duration_seconds * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)


def _read_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


class TestDataPreparation(unittest.TestCase):
    def test_prepare_ljspeech_dataset_filters_and_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            dataset_path = root / "LJSpeech-1.1"
            output_path = root / "prepared"
            wavs_path = dataset_path / "wavs"
            wavs_path.mkdir(parents=True)

            _write_wav(wavs_path / "LJ001-0001.wav", duration_seconds=2.0)
            _write_wav(wavs_path / "LJ001-0002.wav", duration_seconds=0.4)
            _write_wav(wavs_path / "LJ001-0003.wav", duration_seconds=3.0)

            (dataset_path / "metadata.csv").write_text(
                "\n".join(
                    [
                        "LJ001-0001|unused|A valid sample.",
                        "LJ001-0002|unused|Too short.",
                        "LJ001-0003|unused|Another valid sample."
                    ]
                ),
                encoding="utf-8",
            )

            result = prepare_ljspeech_dataset(
                dataset_path=dataset_path,
                output_path=output_path,
                min_duration=1.0,
                max_duration=10.0,
                eval_ratio=0.5,
            )

            train_rows = _read_manifest(result["train_manifest"])
            eval_rows = _read_manifest(result["eval_manifest"])
            all_rows = train_rows + eval_rows

            self.assertEqual(len(all_rows), 2)
            self.assertEqual({row["text"] for row in all_rows}, {"A valid sample.", "Another valid sample."})
            self.assertTrue(all(row["audio_file"].startswith("wavs/") for row in all_rows))

    def test_prepare_hifi_tts_dataset_filters_to_target_speaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            manifest_path = root / "hi_fi_tts_manifest.jsonl"
            output_path = root / "prepared_6097"
            speaker_audio = root / "audio"
            speaker_audio.mkdir()

            valid_file = speaker_audio / "6097_001.flac"
            skipped_other_speaker = speaker_audio / "92_001.flac"
            skipped_too_long = speaker_audio / "6097_002.flac"
            for path in [valid_file, skipped_other_speaker, skipped_too_long]:
                path.write_text("", encoding="utf-8")

            rows = [
                {
                    "audio_filepath": str(valid_file),
                    "text_normalized": "A valid male sample.",
                    "duration": 3.2,
                    "speaker_id": "6097",
                },
                {
                    "audio_filepath": str(skipped_other_speaker),
                    "text_normalized": "Wrong speaker.",
                    "duration": 3.0,
                    "speaker_id": "92",
                },
                {
                    "audio_filepath": str(skipped_too_long),
                    "text_normalized": "Too long for the model.",
                    "duration": 20.0,
                    "speaker_id": "6097",
                },
            ]
            manifest_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            result = prepare_hifi_tts_dataset(
                manifest_path=manifest_path,
                output_path=output_path,
                speaker_id="6097",
                min_duration=1.0,
                max_duration=11.5,
                eval_ratio=0.5,
            )

            train_rows = _read_manifest(result["train_manifest"])
            eval_rows = _read_manifest(result["eval_manifest"])
            all_rows = train_rows + eval_rows

            self.assertEqual(len(all_rows), 1)
            self.assertEqual(all_rows[0]["speaker_name"], "6097")
            self.assertEqual(Path(all_rows[0]["audio_file"]), valid_file)
            self.assertEqual(all_rows[0]["text"], "A valid male sample.")


if __name__ == "__main__":
    unittest.main()
