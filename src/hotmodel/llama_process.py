from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import subprocess
import time
import urllib.error
import urllib.request

from .backend import get_json, post_json
from .config import ModelSpec
from .config import RouterSpec


class ManagedBackend(Protocol):
    model: ModelSpec

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def is_running(self) -> bool:
        ...

    def wait_ready(self, timeout_seconds: float) -> bool:
        ...


@dataclass
class LlamaServerProcess:
    model: ModelSpec
    startup_timeout_seconds: float = 120
    process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.is_running():
            return
        args = self._build_args()
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not self.wait_ready(self.startup_timeout_seconds):
            self.stop()
            raise RuntimeError(f"llama-server for '{self.model.name}' did not become healthy")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait_ready(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            if _health_ok(self.model.base_url):
                return True
            time.sleep(0.5)
        return False

    def _build_args(self) -> list[str]:
        args = [
            self.model.binary,
            "-m",
            str(Path(self.model.path)),
            "--host",
            self.model.host,
            "--port",
            str(self.model.port),
            "-c",
            str(self.model.ctx_size),
        ]
        if self.model.gpu_layers is not None:
            args.extend(["-ngl", str(self.model.gpu_layers)])
        if self.model.threads is not None:
            args.extend(["--threads", str(self.model.threads)])
        if self.model.parallel is not None:
            args.extend(["--parallel", str(self.model.parallel)])
        args.extend(self.model.extra_args)
        return args


def _health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


@dataclass
class LlamaRouterProcess:
    router: RouterSpec
    startup_timeout_seconds: float = 120
    process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return self.router.base_url

    def start(self) -> None:
        if self.is_running():
            return
        self.process = subprocess.Popen(
            self._build_args(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not self.wait_ready(self.startup_timeout_seconds):
            self.stop()
            raise RuntimeError("llama-server router did not become healthy")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait_ready(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            if _health_ok(self.base_url):
                return True
            time.sleep(0.5)
        return False

    def load_model(self, model: ModelSpec) -> None:
        post_json(self.base_url, "/models/load", {"model": model.llama_model_id}, timeout_seconds=600)
        self._wait_model_status(model.llama_model_id, "loaded", timeout_seconds=600)

    def unload_model(self, model: ModelSpec) -> None:
        post_json(self.base_url, "/models/unload", {"model": model.llama_model_id}, timeout_seconds=600)
        self._wait_model_status(model.llama_model_id, "unloaded", timeout_seconds=120)

    def _wait_model_status(self, model_id: str, expected: str, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            try:
                data = get_json(self.base_url, "/models", timeout_seconds=15)
            except RuntimeError:
                time.sleep(0.5)
                continue
            for item in data.get("data", []):
                if item.get("id") != model_id:
                    continue
                last_status = _model_status_value(item)
                if last_status == expected:
                    return
                if _model_status_failed(item):
                    raise RuntimeError(f"model '{model_id}' failed while waiting for {expected}")
            time.sleep(0.5)
        raise RuntimeError(f"model '{model_id}' did not reach status '{expected}', last status: {last_status}")

    def _build_args(self) -> list[str]:
        args = [
            self.router.binary,
            "--host",
            self.router.host,
            "--port",
            str(self.router.port),
            "--models-max",
            str(self.router.models_max),
            "--ctx-size",
            str(self.router.ctx_size),
            "--parallel",
            str(self.router.parallel),
            "--cache-type-k",
            self.router.cache_type_k,
            "--cache-type-v",
            self.router.cache_type_v,
        ]
        if self.router.models_dir is not None:
            args.extend(["--models-dir", str(self.router.models_dir)])
        if self.router.models_preset is not None:
            args.extend(["--models-preset", str(self.router.models_preset)])
        if self.router.models_autoload:
            args.append("--models-autoload")
        else:
            args.append("--no-models-autoload")
        args.append("--kv-unified" if self.router.kv_unified else "--no-kv-unified")
        if self.router.cache_ram_mb is not None:
            args.extend(["--cache-ram", str(self.router.cache_ram_mb)])
        args.append("--cache-idle-slots" if self.router.cache_idle_slots else "--no-cache-idle-slots")
        if self.router.flash_attn is not None:
            args.extend(["--flash-attn", self.router.flash_attn])
        if self.router.no_webui:
            args.append("--no-webui")
        args.extend(self.router.extra_args)
        return args


def _model_status_value(item: dict[str, object]) -> str | None:
    status = item.get("status")
    if isinstance(status, str):
        return status.lower()
    if isinstance(status, dict):
        value = status.get("value")
        if isinstance(value, str):
            return value.lower()
    loaded = item.get("loaded")
    if isinstance(loaded, bool):
        return "loaded" if loaded else "unloaded"
    return None


def _model_status_failed(item: dict[str, object]) -> bool:
    status = item.get("status")
    if isinstance(status, dict):
        failed = status.get("failed")
        if isinstance(failed, bool):
            return failed
        value = status.get("value")
        if isinstance(value, str) and value.lower() in {"failed", "error"}:
            return True
    if isinstance(status, str) and status.lower() in {"failed", "error"}:
        return True
    return False
