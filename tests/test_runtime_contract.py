from clstrainer_lite.distributed import _resolve_backend


def test_backend_mapping_keeps_hccl_for_npu():
    assert _resolve_backend("npu", "auto") == "hccl"
    assert _resolve_backend("cuda", "auto") == "nccl"
    assert _resolve_backend("cpu", "auto") == "gloo"
