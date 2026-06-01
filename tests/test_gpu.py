from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from hotmodel.gpu import GpuMemoryProbe


class GpuMemoryProbeTests(unittest.TestCase):
    def test_used_memory_by_gpu_parses_nvidia_smi_output(self) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="100\n250\n",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=result):
            probe = GpuMemoryProbe()

            self.assertEqual(probe.used_memory_by_gpu_mb(), [100, 250])
            self.assertEqual(probe.used_memory_mb(), 350)

    def test_snapshot_marks_missing_nvidia_smi_as_unavailable(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            snapshot = GpuMemoryProbe().snapshot()

        self.assertIsNone(snapshot.total_mb)
        self.assertIsNone(snapshot.per_gpu_mb)


if __name__ == "__main__":
    unittest.main()
