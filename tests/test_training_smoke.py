from pathlib import Path

from clstrainer_lite.config import load_config
from clstrainer_lite.distributed import Runtime
from clstrainer_lite.trainer import train
from tests.helpers import make_pair_dataset


def test_cpu_training_runs_periodic_val_test_and_writes_curves(tmp_path: Path):
    make_pair_dataset(tmp_path, frames=4, size=(16, 20))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
experiment:
  name: smoke
  output_dir: {tmp_path / 'runs'}
  seed: 123
data:
  train_root: {tmp_path / 'train'}
  test_root: {tmp_path / 'test'}
  val_ratio: 0.25
  delta: 1
  image_size: [16, 20]
sampler:
  enabled: true
  class_probability: [0.5, 0.5]
  game_balance_alpha: 0.5
model:
  factory: clstrainer_lite.models:build_tiny_model
  kwargs: {{width: 4}}
train:
  epochs: 2
  batch_size: 4
  num_workers: 0
  learning_rate: 0.01
  scheduler: none
  decision_threshold: 0.5
  log_every_steps: 0
  n_val_step: 2
  n_test_step: 3
runtime:
  accelerator: cpu
  backend: gloo
  amp: false
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    import torch

    runtime = Runtime(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), backend=None)
    summary = train(config, runtime)
    run_dir = Path(summary["run_dir"])

    assert (run_dir / "history.json").is_file()
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "loss_curve.png").is_file()
    assert (run_dir / "f1_curve.png").is_file()
    assert (run_dir / "class_metrics_curve.png").is_file()
    assert (run_dir / "metrics_detail.json").is_file()
    assert (run_dir / "best_val_diagnostics.png").is_file()
    assert (run_dir / "final_test_diagnostics.png").is_file()
    assert (run_dir / "checkpoints" / "best_model.pt").is_file()
    assert (run_dir / "checkpoints" / "last_model.pt").is_file()
    assert (run_dir / "checkpoints" / "last_checkpoint.pt").is_file()

    import json

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    val_steps = [row["step"] for row in history if row["val_loss"] is not None]
    test_steps = [row["step"] for row in history if row["test_loss"] is not None]
    assert val_steps == [2, 4, 6, 8, 10]
    assert test_steps == [3, 6, 9, 10]
    assert all(row["train_loss"] is not None for row in history)
    assert history[-1]["val_samples"] == 6
    assert history[-1]["test_samples"] == 12
    assert summary["best_val_step"] in val_steps
    assert "per_game" in summary["best_val_metrics"]
    assert "0" in summary["best_val_metrics"]["per_class"]
    assert "1" in summary["best_val_metrics"]["per_class"]

    import torch

    checkpoint = torch.load(
        run_dir / "checkpoints" / "last_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["global_step"] == 10
