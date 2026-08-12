import torch

from clstrainer_lite.metrics import BinaryAccumulator


def test_binary_f1_from_logits():
    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    targets = torch.tensor([1, 0, 0, 1])
    acc = BinaryAccumulator(torch.device("cpu"))
    acc.update(torch.tensor(0.25), logits, targets, threshold=0.5)
    result = acc.compute()
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["f1"] == 0.5
    assert result["loss"] == 0.25
