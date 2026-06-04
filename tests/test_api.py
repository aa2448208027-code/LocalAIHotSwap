from __future__ import annotations

import unittest

from hotmodel.api import dump_request_model


class DumpRequestModelTests(unittest.TestCase):
    def test_uses_pydantic_v2_model_dump_when_available(self) -> None:
        class Model:
            def model_dump(self, by_alias: bool, exclude: set[str]):
                return {"by_alias": by_alias, "exclude": exclude, "source": "v2"}

        result = dump_request_model(Model(), exclude={"messages"})

        self.assertEqual(result, {"by_alias": True, "exclude": {"messages"}, "source": "v2"})

    def test_falls_back_to_pydantic_v1_dict(self) -> None:
        class Model:
            def dict(self, by_alias: bool, exclude: set[str]):
                return {"by_alias": by_alias, "exclude": exclude, "source": "v1"}

        result = dump_request_model(Model(), exclude={"messages"})

        self.assertEqual(result, {"by_alias": True, "exclude": {"messages"}, "source": "v1"})


if __name__ == "__main__":
    unittest.main()
