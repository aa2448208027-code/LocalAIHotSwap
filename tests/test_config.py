from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hotmodel.config import load_config


class ConfigTests(unittest.TestCase):
    def test_router_model_path_is_resolved_relative_to_config_without_port(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path = root / "models.toml"
            config_path.write_text(
                """
[server]
active_model = "small"

[llama]
backend_mode = "router"

[router]
models_preset = "llama-models.ini"
models_max = 1

[models.small]
path = "models/small.gguf"
router_id = "small"
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.models["small"].path, root / "models" / "small.gguf")
            self.assertEqual(config.models["small"].port, 0)
            self.assertEqual(config.router.models_preset, root / "llama-models.ini")

    def test_process_mode_requires_model_port(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config_path = Path(raw) / "models.toml"
            config_path.write_text(
                """
[llama]
backend_mode = "process"

[models.small]
path = "small.gguf"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "positive port"):
                load_config(config_path)

    def test_legacy_no_webui_key_is_still_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config_path = Path(raw) / "models.toml"
            config_path.write_text(
                """
[llama]
backend_mode = "router"

[router]
models_max = 1
no_webui = false

[models.small]
path = "small.gguf"
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertIs(config.router.no_ui, False)
            self.assertIs(config.router.no_webui, False)


if __name__ == "__main__":
    unittest.main()
