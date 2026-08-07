from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from game_cls.config import (
    list_available_layers,
    load_config,
    load_config_with_sources,
)
from game_cls.config_schema import ConfigSchemaError


class RecipeLayerTests(unittest.TestCase):
    def test_example_recipe_merges_contract_profile_presets(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        # contract layer
        self.assertEqual(config["decision"]["threshold"], 0.99)
        self.assertEqual(config["pair"]["test_delta"], 2)
        self.assertEqual(config["model"]["trainable_name_contains"], "cls")
        self.assertEqual(config["data"]["width"], 448)
        # profile layer
        self.assertEqual(config["device"]["accelerator"], "cpu")
        # preset layers
        self.assertFalse(config["augmentation"]["enabled"])
        self.assertEqual(config["evaluation"]["val_quick_every_steps"], 10)
        # recipe's own fields win
        self.assertEqual(config["train"]["local_batch_size"], 4)

    def test_recipe_body_overrides_preset(self) -> None:
        config = load_config(
            "configs/recipes/example_debug.yaml",
            ["augmentation.enabled=true"],
        )
        self.assertTrue(config["augmentation"]["enabled"])

    def test_preset_selection_via_cli_override(self) -> None:
        config = load_config(
            "configs/recipes/example_debug.yaml",
            ["presets.augmentation=standard"],
        )
        self.assertTrue(config["augmentation"]["enabled"])
        self.assertTrue(config["augmentation"]["random_affine"]["enabled"])

    def test_profile_switch_via_cli_override(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml", ["profile=cuda_1p"])
        self.assertEqual(config["device"]["accelerator"], "cuda")

    def test_unknown_preset_lists_available(self) -> None:
        with self.assertRaises(ConfigSchemaError) as ctx:
            load_config(
                "configs/recipes/example_debug.yaml",
                ["presets.dataloader=turbo"],
            )
        message = str(ctx.exception)
        self.assertIn("Unknown dataloader preset 'turbo'", message)
        self.assertIn("stable", message)
        self.assertIn("throughput", message)

    def test_unknown_profile_lists_available(self) -> None:
        with self.assertRaises(ConfigSchemaError) as ctx:
            load_config("configs/recipes/example_debug.yaml", ["profile=npu_99p"])
        self.assertIn("Available: cpu_debug", str(ctx.exception))

    def test_unknown_preset_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe.yaml"
            recipe.write_text(
                "profile: cpu_debug\npresets:\n  optimizer: fast\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigSchemaError, "unknown preset group"):
                load_config(recipe)

    def test_recipe_rejects_base_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe.yaml"
            recipe.write_text(
                "profile: cpu_debug\nbase: other.yaml\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigSchemaError, "not 'base'"):
                load_config(recipe)

    def test_layer_file_cannot_be_a_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiles = Path(directory) / "profiles"
            profiles.mkdir()
            (profiles / "bad.yaml").write_text(
                "profile: cpu_debug\ndevice:\n  accelerator: cpu\n",
                encoding="utf-8",
            )
            recipe = Path(directory) / "recipe.yaml"
            recipe.write_text("profile: bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigSchemaError, "plain config sections"):
                load_config(recipe)

    def test_sources_track_every_layer(self) -> None:
        _, sources = load_config_with_sources(
            "configs/recipes/example_debug.yaml",
            ["presets.augmentation=standard"],
        )
        self.assertTrue(
            sources["decision.threshold"].endswith("dual_frame_binary.yaml")
        )
        self.assertTrue(sources["device.accelerator"].endswith("cpu_debug.yaml"))
        self.assertTrue(
            sources["augmentation.random_affine.degrees"].endswith("standard.yaml")
        )
        self.assertEqual(
            sources["presets.augmentation"],
            "override:presets.augmentation=standard",
        )

    def test_production_template_validates(self) -> None:
        config = load_config("configs/recipes/game_cls_production.yaml")
        self.assertEqual(config["device"]["accelerator"], "npu")
        self.assertTrue(config["distributed"]["enabled"])
        self.assertEqual(config["evaluation"]["selection_metric"], "composite")
        # Constrained low-FPR selection comes from the production preset, not
        # from a recipe-local copy.
        self.assertEqual(config["evaluation"]["selection_mode"], "constrained")


class InitCommandTests(unittest.TestCase):
    def test_init_writes_a_valid_recipe(self) -> None:
        from game_cls.cli import build_parser, cmd_init

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "my_recipe.yaml"
            parser = build_parser()
            args = parser.parse_args(
                [
                    "init",
                    "--profile",
                    "npu_1p",
                    "--name",
                    "demo",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(cmd_init(args), 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn("profile: npu_1p", text)
            self.assertIn("REPLACE_ME", text)
            # The generated recipe must parse; placeholder factory is a
            # schema-valid string even though doctor will flag it.
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(Path(__file__).resolve().parents[1])
                config = load_config(output)
            finally:
                os.chdir(old_cwd)
            self.assertEqual(config["device"]["accelerator"], "npu")

    def test_init_refuses_unknown_profile(self) -> None:
        from game_cls.cli import build_parser, cmd_init

        parser = build_parser()
        args = parser.parse_args(
            ["init", "--profile", "npu_99p", "--output", "unused.yaml"]
        )
        self.assertEqual(cmd_init(args), 2)

    def test_list_available_layers(self) -> None:
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parents[1])
            layers = list_available_layers()
        finally:
            os.chdir(old_cwd)
        self.assertIn("npu_8p", layers["profiles"])
        self.assertIn("dual_frame_binary", layers["task_profiles"])
        self.assertIn("augmentation/standard", layers["presets"])
        self.assertIn("evaluation/production", layers["presets"])


if __name__ == "__main__":
    unittest.main()
