from __future__ import annotations

from typing import Any

from .config import RuntimeConfig
from .orchestrator import Orchestrator


def dump_request_model(model: Any, exclude: set[str]) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, exclude=exclude)
    return model.dict(by_alias=True, exclude=exclude)


def create_app(config: RuntimeConfig):
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("fastapi and pydantic are required to run the HTTP API") from exc

    class SwitchRequest(BaseModel):
        model: str

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list[dict[str, Any]]
        session_id: str | None = Field(default=None, alias="session_id")
        temperature: float | None = None
        top_p: float | None = None
        max_tokens: int | None = None
        stream: bool | None = False

        class Config:
            populate_by_name = True
            extra = "allow"

    app = FastAPI(title="HotModelReplacement", version="0.1.0")
    orchestrator = Orchestrator(config)
    app.state.orchestrator = orchestrator

    @app.on_event("startup")
    def _startup() -> None:
        orchestrator.start_default()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        orchestrator.stop()

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = orchestrator.state()
        return {"status": "switching" if state["switching"] else "ok", **state}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "hotmodel"}
                for name in sorted(config.models)
            ],
        }

    @app.get("/admin/state")
    def admin_state() -> dict[str, Any]:
        return orchestrator.state()

    @app.post("/admin/switch")
    def switch(request: SwitchRequest) -> dict[str, Any]:
        try:
            report = orchestrator.switch_model(request.model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "previous_model": report.previous_model,
            "active_model": report.active_model,
            "elapsed_seconds": report.elapsed_seconds,
            "gpu_settled": report.gpu_settled,
            "gpu_memory": report.gpu_memory,
        }

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest, x_hotmodel_session: str | None = Header(default=None)) -> Any:
        payload = dump_request_model(request, exclude={"messages", "session_id"})
        payload = {key: value for key, value in payload.items() if value is not None}
        session_id = request.session_id or x_hotmodel_session
        try:
            if payload.get("stream"):
                stream = orchestrator.chat_stream(session_id, request.messages, payload)
                return StreamingResponse(stream, media_type="text/event-stream")
            return orchestrator.chat(session_id, request.messages, payload)
        except RuntimeError as exc:
            message = str(exc)
            status = 503 if "switch" in message or "no active" in message else 500
            raise HTTPException(status_code=status, detail=message) from exc

    return app
