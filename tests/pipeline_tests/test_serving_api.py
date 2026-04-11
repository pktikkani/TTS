import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from scripts.inference.xtts_runtime import chunk_text, load_voice_registry, sanitize_inference_preset
from scripts.serving.api import create_app


class FakeService:
    def __init__(self):
        self.calls = []

    def health(self):
        return {"status": "ok", "voices_loaded": 2}

    def list_voices(self):
        return [
            {"voice_name": "female", "language": "en"},
            {"voice_name": "male", "language": "en"},
        ]

    def synthesize(self, request):
        self.calls.append(request)
        return {
            "voice_name": request.voice_name,
            "normalized_text": request.text.lower(),
            "audio_file": "/tmp/output.wav",
            "chunks": [request.text],
        }

    def stream(self, request):
        yield {"index": 0, "text": request.text, "audio_file": "/tmp/output.wav"}


class TestServingApi(unittest.TestCase):
    def test_load_voice_registry_and_chunk_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "voices": [
                            {
                                "voice_name": "female",
                                "label": "Female",
                                "checkpoint_path": str(root / "female_best_model.pth"),
                                "config_path": str(root / "config.json"),
                                "tokenizer_path": str(root / "vocab.json"),
                                "speaker_wav": [str(root / "female.wav")],
                                "language": "en"
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            registry = load_voice_registry(registry_path)
            chunks = chunk_text("Sentence one. Sentence two is slightly longer.", max_chars=18)

            self.assertIn("female", registry)
            self.assertEqual(registry["female"].language, "en")
            self.assertEqual(registry["female"].inference_preset["temperature"], 0.75)
            self.assertGreaterEqual(len(chunks), 2)

    def test_sanitize_inference_preset_filters_unknown_keys(self):
        preset = sanitize_inference_preset(
            {
                "temperature": 0.75,
                "repetition_penalty": 10.0,
                "speed": 1.0,
                "sample_rate": 24000,
                "unexpected": 123,
            }
        )

        self.assertEqual(preset["temperature"], 0.75)
        self.assertEqual(preset["repetition_penalty"], 10.0)
        self.assertEqual(preset["speed"], 1.0)
        self.assertNotIn("sample_rate", preset)
        self.assertNotIn("unexpected", preset)

    def test_api_routes_use_service_contract(self):
        service = FakeService()
        client = TestClient(create_app(service))

        health = client.get("/health")
        voices = client.get("/api/v1/voices")
        synth = client.post("/api/v1/synthesize", json={"voice_name": "female", "text": "Hello there."})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(voices.status_code, 200)
        self.assertEqual(len(voices.json()["voices"]), 2)
        self.assertEqual(synth.status_code, 200)
        self.assertEqual(synth.json()["voice_name"], "female")
        self.assertEqual(service.calls[0].text, "Hello there.")


if __name__ == "__main__":
    unittest.main()
