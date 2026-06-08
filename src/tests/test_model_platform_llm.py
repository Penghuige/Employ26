from src.model_platform.config import load_model_runtime_config
from src.model_platform.llm import VLLMHTTPClient


def test_vllm_client_complete_text(monkeypatch):
    runtime = load_model_runtime_config()

    def fake_chat_completion(**kwargs):
        return {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("src.model_platform.llm.chat_completion", fake_chat_completion)
    client = VLLMHTTPClient(runtime)
    assert client.complete_text(system_prompt="s", user_prompt="u") == "hello"


def test_vllm_client_complete_json(monkeypatch):
    runtime = load_model_runtime_config()

    def fake_chat_completion(**kwargs):
        return {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr("src.model_platform.llm.chat_completion", fake_chat_completion)
    client = VLLMHTTPClient(runtime)
    assert client.complete_json(system_prompt="s", user_prompt="u") == {"ok": True}
