from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest


class PackageMetadataTests(unittest.TestCase):
    def test_dev_extra_installs_fastapi_testclient_dependency(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
        dependency_names = {
            re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower()
            for dependency in dev_dependencies
        }

        self.assertIn(
            "httpx",
            dependency_names,
            "dev extra must install the httpx module used by FastAPI TestClient",
        )
        self.assertNotIn(
            "httpx2",
            dependency_names,
            "FastAPI TestClient imports httpx, not the httpx2 distribution",
        )


if __name__ == "__main__":
    unittest.main()
