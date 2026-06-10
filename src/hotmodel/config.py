from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    port: int = 0
    host: str = "127.0.0.1"
    binary: str = "llama-server"
    ctx_size: int = 8192
    gpu_layers: int | None = None
    threads: int | None = None
    parallel: int | None = None
    extra_args: tuple[str, ...] = ()
    router_id: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def llama_model_id(self) -> str:
        return self.router_id or self.name


@dataclass(frozen=True)
class RouterSpec:
    binary: str = "llama-server"
    host: str = "127.0.0.1"
    port: int = 28000
    models_dir: Path | None = None
    models_preset: Path | None = None
    models_max: int = 1
    models_autoload: bool = False
    ctx_size: int = 8192
    parallel: int = 1
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    kv_unified: bool = True
    cache_ram_mb: int | None = 1024
    cache_idle_slots: bool = True
    flash_attn: str | None = "auto"
    no_ui: bool = True
    no_webui: bool | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class RuntimeConfig:
    host: str
    port: int
    state_path: Path
    active_model: str | None
    switch_policy: str
    backend_mode: str
    switch_drain_timeout_seconds: float
    gpu_settle_timeout_seconds: float
    gpu_settle_memory_mb: int | None
    system_prompt: str
    max_session_messages: int | None
    max_prompt_chars: int | None
    max_prompt_tokens: int | None
    token_budget_mode: str
    token_budget_chars_per_token: float
    router: RouterSpec | None
    models: dict[str, ModelSpec] = field(default_factory=dict)


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    server = raw.get("server", {})
    session = raw.get("session", {})
    preset = raw.get("preset", {})
    llama = raw.get("llama", {})
    router_raw = raw.get("router", {})
    models_raw: dict[str, Any] = raw.get("models", {})

    llama_binary = str(llama.get("binary", "llama-server"))
    llama_host = str(llama.get("host", "127.0.0.1"))
    backend_mode = str(llama.get("backend_mode", "router"))
    router = _load_router(config_path, llama_binary, llama_host, router_raw) if backend_mode == "router" else None

    models: dict[str, ModelSpec] = {}
    for name, item in models_raw.items():
        model_host = str(item.get("host", llama_host))
        model_binary = str(item.get("binary", llama_binary))
        models[name] = ModelSpec(
            name=name,
            path=_resolve_path(config_path, str(item["path"])),
            port=int(item.get("port", 0)),
            host=model_host,
            binary=model_binary,
            ctx_size=int(item.get("ctx_size", 8192)),
            gpu_layers=_optional_int(item.get("gpu_layers")),
            threads=_optional_int(item.get("threads")),
            parallel=_optional_int(item.get("parallel")),
            extra_args=tuple(str(arg) for arg in item.get("extra_args", [])),
            router_id=str(item.get("router_id", name)),
        )

    active_model = server.get("active_model")
    runtime = RuntimeConfig(
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 18080)),
        state_path=_resolve_path(config_path, str(server.get("state_path", ".hotmodel/state.json"))),
        active_model=str(active_model) if active_model else None,
        switch_policy=str(server.get("switch_policy", "zero_overlap")),
        backend_mode=backend_mode,
        switch_drain_timeout_seconds=float(server.get("switch_drain_timeout_seconds", 300)),
        gpu_settle_timeout_seconds=float(server.get("gpu_settle_timeout_seconds", 0)),
        gpu_settle_memory_mb=_optional_int(server.get("gpu_settle_memory_mb")),
        system_prompt=str(preset.get("system_prompt", "")),
        max_session_messages=_optional_int(session.get("max_session_messages")),
        max_prompt_chars=_optional_int(session.get("max_prompt_chars")),
        max_prompt_tokens=_optional_int(session.get("max_prompt_tokens")),
        token_budget_mode=str(session.get("token_budget_mode", "auto")),
        token_budget_chars_per_token=float(session.get("token_budget_chars_per_token", 4.0)),
        router=router,
        models=models,
    )
    _validate(runtime)
    return runtime


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _optional_path(config_path: Path, value: Any) -> Path | None:
    if value is None:
        return None
    return _resolve_path(config_path, str(value))


def _load_router(config_path: Path, binary: str, host: str, raw: dict[str, Any]) -> RouterSpec:
    return RouterSpec(
        binary=str(raw.get("binary", binary)),
        host=str(raw.get("host", host)),
        port=int(raw.get("port", 28000)),
        models_dir=_optional_path(config_path, raw.get("models_dir")),
        models_preset=_optional_path(config_path, raw.get("models_preset")),
        models_max=int(raw.get("models_max", 1)),
        models_autoload=bool(raw.get("models_autoload", False)),
        ctx_size=int(raw.get("ctx_size", 8192)),
        parallel=int(raw.get("parallel", 1)),
        cache_type_k=str(raw.get("cache_type_k", "q8_0")),
        cache_type_v=str(raw.get("cache_type_v", "q8_0")),
        kv_unified=bool(raw.get("kv_unified", True)),
        cache_ram_mb=_optional_int(raw.get("cache_ram_mb", 1024)),
        cache_idle_slots=bool(raw.get("cache_idle_slots", True)),
        flash_attn=str(raw["flash_attn"]) if raw.get("flash_attn") is not None else None,
        no_ui=bool(raw.get("no_ui", raw.get("no_webui", True))),
        no_webui=_optional_bool(raw.get("no_webui")),
        extra_args=tuple(str(arg) for arg in raw.get("extra_args", [])),
    )


def _validate(config: RuntimeConfig) -> None:
    if config.switch_policy != "zero_overlap":
        raise ValueError("only switch_policy='zero_overlap' is currently supported")
    if config.backend_mode not in {"router", "process"}:
        raise ValueError("backend_mode must be 'router' or 'process'")
    if config.backend_mode == "router" and config.router is None:
        raise ValueError("router config is required for backend_mode='router'")
    if config.switch_drain_timeout_seconds < 0:
        raise ValueError("switch_drain_timeout_seconds must be >= 0")
    if config.max_session_messages is not None and config.max_session_messages < 1:
        raise ValueError("max_session_messages must be >= 1")
    if config.max_prompt_chars is not None and config.max_prompt_chars < 1:
        raise ValueError("max_prompt_chars must be >= 1")
    if config.max_prompt_tokens is not None and config.max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be >= 1")
    if config.token_budget_mode not in {"auto", "llama", "estimate"}:
        raise ValueError("token_budget_mode must be 'auto', 'llama', or 'estimate'")
    if config.token_budget_chars_per_token <= 0:
        raise ValueError("token_budget_chars_per_token must be > 0")
    if config.router is not None:
        if config.router.models_max != 1:
            raise ValueError("router.models_max must be 1 for zero_overlap switching")
        if config.router.parallel < 1:
            raise ValueError("router.parallel must be >= 1")
        if config.router.ctx_size < 1:
            raise ValueError("router.ctx_size must be >= 1")
    if config.backend_mode == "process":
        for name, model in config.models.items():
            if model.port <= 0:
                raise ValueError(f"model '{name}' must define a positive port for backend_mode='process'")
    if config.active_model and config.active_model not in config.models:
        raise ValueError(f"active_model '{config.active_model}' is not defined")
    if not config.models:
        raise ValueError("at least one model must be configured")
