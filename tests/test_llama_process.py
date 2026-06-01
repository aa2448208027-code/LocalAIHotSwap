from __future__ import annotations

import unittest

from hotmodel.config import RouterSpec
from hotmodel.llama_process import LlamaRouterProcess, _model_status_failed, _model_status_value


class LlamaRouterProcessTests(unittest.TestCase):
    def test_router_args_include_memory_controls(self) -> None:
        process = LlamaRouterProcess(
            RouterSpec(
                port=28000,
                models_max=1,
                models_autoload=False,
                ctx_size=4096,
                parallel=1,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                kv_unified=True,
                cache_ram_mb=512,
                cache_idle_slots=True,
                flash_attn="auto",
                no_webui=True,
                extra_args=("--metrics",),
            )
        )

        args = process._build_args()

        self.assertIn("--models-max", args)
        self.assertIn("1", args)
        self.assertIn("--no-models-autoload", args)
        self.assertIn("--ctx-size", args)
        self.assertIn("4096", args)
        self.assertIn("--parallel", args)
        self.assertIn("--cache-type-k", args)
        self.assertIn("--cache-type-v", args)
        self.assertIn("--kv-unified", args)
        self.assertIn("--cache-ram", args)
        self.assertIn("512", args)
        self.assertIn("--cache-idle-slots", args)
        self.assertIn("--flash-attn", args)
        self.assertIn("auto", args)
        self.assertIn("--no-webui", args)
        self.assertIn("--metrics", args)

    def test_model_status_parser_accepts_common_shapes(self) -> None:
        self.assertEqual(_model_status_value({"status": {"value": "LOADED"}}), "loaded")
        self.assertEqual(_model_status_value({"status": "UNLOADED"}), "unloaded")
        self.assertEqual(_model_status_value({"loaded": True}), "loaded")
        self.assertEqual(_model_status_value({"loaded": False}), "unloaded")
        self.assertIsNone(_model_status_value({"id": "model"}))

    def test_model_status_parser_detects_failures(self) -> None:
        self.assertTrue(_model_status_failed({"status": {"failed": True}}))
        self.assertTrue(_model_status_failed({"status": {"value": "ERROR"}}))
        self.assertTrue(_model_status_failed({"status": "FAILED"}))
        self.assertFalse(_model_status_failed({"status": {"value": "loaded"}}))


if __name__ == "__main__":
    unittest.main()
