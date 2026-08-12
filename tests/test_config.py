from pathlib import Path

from clstrainer_lite.config import load_config


def test_config_requires_train_and_test_not_val_root(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
data:
  train_root: train
  test_root: test
  val_ratio: 0.2
train:
  n_val_step: 10
  n_test_step: 20
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded["data"]["train_root"] == "train"
    assert loaded["data"]["test_root"] == "test"
    assert "val_root" not in loaded["data"]


def test_legacy_delta_migrates_to_fixed_range(tmp_path: Path):
    config = tmp_path / "legacy.yaml"
    config.write_text(
        """
data:
  train_root: train
  test_root: test
  delta: 3
train:
  n_val_step: 10
  n_test_step: 20
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded["data"]["train_delta_range"] == [3, 3]
    assert loaded["data"]["eval_delta"] == 3
    assert "delta" not in loaded["data"]
