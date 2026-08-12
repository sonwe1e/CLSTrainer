from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from tests.helpers import make_pair_dataset


def test_two_process_gloo_ddp_periodic_eval_smoke(tmp_path: Path):
    make_pair_dataset(tmp_path, frames=4, size=(12, 16))
    run_root = tmp_path / "runs"
    config = tmp_path / "ddp.yaml"
    config.write_text(
        f"""
experiment:
  name: ddp_smoke
  output_dir: {run_root}
  seed: 7
data:
  train_root: {tmp_path / 'train'}
  test_root: {tmp_path / 'test'}
  val_ratio: 0.25
  delta: 1
  image_size: [12, 16]
sampler:
  enabled: true
  class_probability: [0.5, 0.5]
  game_balance_alpha: 0.5
model:
  factory: clstrainer_lite.models:build_tiny_model
  kwargs: {{width: 4}}
train:
  epochs: 1
  batch_size: 2
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
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m",
        "clstrainer_lite.cli",
        "train",
        "--config",
        str(config),
    ]
    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout
    runs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "checkpoints" / "best_model.pt").is_file()
    assert (run_dir / "loss_curve.png").is_file()
    assert (run_dir / "f1_curve.png").is_file()
    assert (run_dir / "class_metrics_curve.png").is_file()
    assert (run_dir / "metrics_detail.json").is_file()
    assert (run_dir / "best_val_diagnostics.png").is_file()
    assert (run_dir / "final_test_diagnostics.png").is_file()
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    # Validation/test use no-padding samplers: global sample counts stay exact.
    assert history[-1]["val_samples"] == 6
    assert history[-1]["test_samples"] == 12
