import tempfile
import unittest
from pathlib import Path

import yaml

from runner.render_resolved_config import parse_override, set_nested


class RenderResolvedConfigTests(unittest.TestCase):
    def test_parse_override_uses_yaml_types(self):
        key, value = parse_override("training.epochs=4")
        self.assertEqual(key, "training.epochs")
        self.assertEqual(value, 4)

        key, value = parse_override("checkpoint.resume=null")
        self.assertEqual(key, "checkpoint.resume")
        self.assertIsNone(value)

    def test_set_nested_updates_mapping(self):
        payload = {"training": {"epochs": 200}}
        set_nested(payload, "training.epochs", 4)
        set_nested(payload, "data.max_patches", 16)
        self.assertEqual(payload["training"]["epochs"], 4)
        self.assertEqual(payload["data"]["max_patches"], 16)

    def test_yaml_round_trip_for_rendered_output(self):
        payload = {"data": {"batch_size": 4}, "checkpoint": {"resume": "foo"}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "resolved.yaml"
            with output_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

            with output_path.open("r", encoding="utf-8") as handle:
                rendered = yaml.safe_load(handle)

        self.assertEqual(rendered["data"]["batch_size"], 4)
        self.assertEqual(rendered["checkpoint"]["resume"], "foo")


if __name__ == "__main__":
    unittest.main()
