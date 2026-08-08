def test_training_entrypoint_imports():
    from game_cls.engine.training.loop import run_training

    assert callable(run_training)


def test_release_command_is_not_rewritten_as_train(monkeypatch):
    import game_cls.cli.init as cli_init

    captured = {}

    class _Parser:
        def parse_args(self, argv):
            captured["argv"] = argv
            return type("Args", (), {"func": staticmethod(lambda _args: 0)})()

    monkeypatch.setattr(cli_init, "build_parser", lambda: _Parser())
    assert cli_init.main(["release", "check"]) == 0
    assert captured["argv"] == ["release", "check"]


def test_root_option_is_not_rewritten_as_train(monkeypatch):
    import game_cls.cli.init as cli_init

    captured = {}

    class _Parser:
        def parse_args(self, argv):
            captured["argv"] = argv
            return type("Args", (), {"func": staticmethod(lambda _args: 0)})()

    monkeypatch.setattr(cli_init, "build_parser", lambda: _Parser())
    assert cli_init.main(["--version"]) == 0
    assert captured["argv"] == ["--version"]
