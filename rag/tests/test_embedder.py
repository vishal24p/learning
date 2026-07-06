"""Unit test: Embedder accepts both .text objects and plain strings."""
from __future__ import annotations


class FakeEmbModel:
    def __init__(self):
        self.calls = []

    def get_text_embedding(self, text):
        self.calls.append(text)
        return [float(len(text)), 0.0, 0.0]


def test_embed_chunk_unwraps_tuple_result():
    from src.embeddings.embedder import Embedder

    fake = FakeEmbModel()
    emb = Embedder.__new__(Embedder)
    emb.embed_model = fake
    assert emb.embed_chunk("hi") == [2.0, 0.0, 0.0]


def test_embed_chunks_handles_objects_and_strings():
    from src.embeddings.embedder import Embedder

    class Node:
        def __init__(self, text):
            self.text = text

    fake = FakeEmbModel()
    emb = Embedder.__new__(Embedder)
    emb.embed_model = fake
    out = emb.embed_chunks([Node("abc"), "hello"])
    assert out == [
        {"text": "abc", "embedding": [3.0, 0.0, 0.0]},
        {"text": "hello", "embedding": [5.0, 0.0, 0.0]},
    ]
