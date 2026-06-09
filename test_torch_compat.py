import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "FLAG" / "FLAG"))

from torch_compat import load_legacy_torch


class TorchCompatTests(unittest.TestCase):
    def test_load_legacy_torch_forces_weights_only_false(self):
        captured = {}

        def fake_load(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"ok": True}

        with patch("torch.load", side_effect=fake_load):
            result = load_legacy_torch("/tmp/example.pt")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["path"], "/tmp/example.pt")
        self.assertEqual(captured["kwargs"]["weights_only"], False)


if __name__ == "__main__":
    unittest.main()
