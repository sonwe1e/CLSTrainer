from __future__ import annotations

import unittest

from game_cls.model.freeze_policy import configure_trainable_parameters


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.count = count
        self.requires_grad = True

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(self) -> None:
        self.parameters = {
            "backbone.weight": FakeParameter(100),
            "head.Classifier.weight": FakeParameter(10),
            "head.cls.weight": FakeParameter(8),
            "head.cls.bias": FakeParameter(2),
        }

    def named_parameters(self):
        return self.parameters.items()


class FreezePolicyTests(unittest.TestCase):
    def test_only_lowercase_cls_is_trainable(self) -> None:
        model = FakeModel()
        summary = configure_trainable_parameters(model)
        self.assertEqual(summary.trainable_names, ("head.cls.weight", "head.cls.bias"))
        self.assertEqual(summary.trainable_count, 10)
        self.assertEqual(summary.frozen_count, 110)
        self.assertFalse(model.parameters["head.Classifier.weight"].requires_grad)

    def test_fails_when_no_parameter_matches(self) -> None:
        model = FakeModel()
        with self.assertRaises(RuntimeError):
            configure_trainable_parameters(model, "missing")


if __name__ == "__main__":
    unittest.main()
