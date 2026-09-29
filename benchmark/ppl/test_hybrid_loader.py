import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from loader import apply_hybrid_config_from_env


class HybridWidthOverrideTest(unittest.TestCase):
    def _load(self, *, force, env_bits="6"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.json"
            path.write_text(json.dumps({
                "format": "scmp_llm_hybrid_v1",
                "default": "sc",
                "int_bits": 7,
                "schedule": {"qk": ["int7", "sc"]},
            }))
            model = SimpleNamespace(config=SimpleNamespace())
            env = {
                "SC_HYBRID_CONFIG_JSON": str(path),
                "SC_HYBRID_INT_BITS": str(env_bits),
                "SC_HYBRID_FORCE_INT_BITS": "1" if force else "0",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                apply_hybrid_config_from_env(model)
            return model.config

    def test_force_makes_target_width_override_legacy_json(self):
        cfg = self._load(force=True, env_bits="6")
        self.assertEqual(cfg.sc_hybrid_int_bits, 6)
        self.assertTrue(cfg.sc_hybrid_force_int_bits)
        # The historical schedule remains auditable; runtime force semantics
        # map this backend to cfg.sc_hybrid_int_bits.
        self.assertEqual(cfg.sc_hybrid_schedule[("qk", 0)], "int7")

    def test_json_remains_authoritative_without_force(self):
        cfg = self._load(force=False, env_bits="6")
        self.assertEqual(cfg.sc_hybrid_int_bits, 7)
        self.assertFalse(cfg.sc_hybrid_force_int_bits)


if __name__ == "__main__":
    unittest.main()
