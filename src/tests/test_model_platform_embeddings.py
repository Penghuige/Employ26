from functools import lru_cache

import src.model_platform.embeddings as embeddings


def test_embedding_model_is_cached(monkeypatch):
    calls = []

    class FakeEmbeddingModel:
        pass

    @lru_cache(maxsize=8)
    def fake_loader(model_path, device):
        calls.append((model_path, device))
        return FakeEmbeddingModel()

    monkeypatch.setattr(embeddings, "_load_sentence_transformer", fake_loader)
    first = embeddings.get_embedding_model("bge", device="cpu")
    second = embeddings.get_embedding_model("bge", device="cpu")
    assert first is second
    assert len(calls) == 1
