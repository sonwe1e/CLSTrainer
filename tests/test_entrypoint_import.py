def test_training_entrypoint_imports():
    from game_cls.engine.trainer import run_training

    assert callable(run_training)
