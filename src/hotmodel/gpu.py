from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time


@dataclass(frozen=True)
class GpuMemorySnapshot:
    total_mb: int | None
    per_gpu_mb: list[int] | None
    captured_at: float

    def as_dict(self) -> dict[str, object]:
        return {
            "total_mb": self.total_mb,
            "per_gpu_mb": self.per_gpu_mb,
            "captured_at": self.captured_at,
        }


class GpuMemoryProbe:
    def used_memory_mb(self) -> int | None:
        values = self.used_memory_by_gpu_mb()
        if not values:
            return None
        return sum(values)

    def used_memory_by_gpu_mb(self) -> list[int] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        values: list[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(line))
            except ValueError:
                return None
        if not values:
            return None
        return values

    def snapshot(self) -> GpuMemorySnapshot:
        values = self.used_memory_by_gpu_mb()
        return GpuMemorySnapshot(
            total_mb=sum(values) if values else None,
            per_gpu_mb=values,
            captured_at=time.time(),
        )

    def wait_until_below(self, threshold_mb: int, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            used = self.used_memory_mb()
            if used is None:
                return False
            if used <= threshold_mb:
                return True
            time.sleep(0.5)
        return False
