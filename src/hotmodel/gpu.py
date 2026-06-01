from __future__ import annotations

import subprocess
import time


class GpuMemoryProbe:
    def used_memory_mb(self) -> int | None:
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
        return sum(values)

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
