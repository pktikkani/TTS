#!/usr/bin/env python3
"""FastAPI serving layer for local XTTS voices."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from scripts.inference.xtts_runtime import LocalXTTSService, SynthesizeRequest


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "voices" / "registry.json"


class SynthesizeBody(BaseModel):
    voice_name: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    output_dir: Optional[str] = None
    language: Optional[str] = "en"
    normalize: bool = True


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="XTTS Voice API", version="0.1.0")
    app.state.service = service or LocalXTTSService(DEFAULT_REGISTRY_PATH)

    @app.get("/health")
    def health():
        return app.state.service.health()

    @app.get("/api/v1/voices")
    def list_voices():
        return {"voices": app.state.service.list_voices()}

    @app.post("/api/v1/synthesize")
    def synthesize(body: SynthesizeBody):
        try:
            request = SynthesizeRequest(
                voice_name=body.voice_name,
                text=body.text,
                output_dir=Path(body.output_dir).resolve() if body.output_dir else None,
                language=body.language,
                normalize=body.normalize,
            )
            return app.state.service.synthesize(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown voice: {body.voice_name}") from exc

    @app.websocket("/api/v1/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        payload = await websocket.receive_json()
        request = SynthesizeRequest(
            voice_name=payload["voice_name"],
            text=payload["text"],
            output_dir=Path(payload["output_dir"]).resolve() if payload.get("output_dir") else None,
            language=payload.get("language", "en"),
            normalize=payload.get("normalize", True),
        )
        try:
            for chunk in app.state.service.stream(request):
                await websocket.send_json(chunk)
        except KeyError as exc:
            await websocket.send_json({"error": f"Unknown voice: {request.voice_name}"})
        finally:
            await websocket.close()

    return app


app = create_app()
