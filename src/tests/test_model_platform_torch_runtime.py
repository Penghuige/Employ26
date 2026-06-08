from src.model_platform.torch_runtime import resolve_model_path, resolve_torch_device


def test_resolve_model_path_reads_nested_inventory():
    assert resolve_model_path("bge_base_zh_v1_5").name == "bge-base-zh-v1.5"


def test_resolve_torch_device_can_force_cpu():
    assert resolve_torch_device(prefer_cuda=False) == "cpu"
