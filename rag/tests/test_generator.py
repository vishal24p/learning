"""Unit tests for Generator using a fake chat LLM."""
from __future__ import annotations

from llama_index.core.llms import ChatMessage, ChatResponse


class FakeChatLLM:
    """Mimics llama_index.llms.ollama.Ollama's chat() / complete() shape."""
    def __init__(self, reply: str = "answer-from-fake") -> None:
        self.last_chat = None
        self.last_complete = None
        self.reply = reply

    def chat(self, messages, **kwargs):
        self.last_chat = list(messages)
        return ChatResponse(message=ChatMessage(role="assistant", content=self.reply))

    def complete(self, prompt, **kwargs):
        self.last_complete = (prompt, kwargs)
        return ChatResponse(message=ChatMessage(role="assistant", content=self.reply))


def _gen_with_fake(reply="ok"):
    from src.generation.generator import Generator

    g = Generator.__new__(Generator)
    g.llm = FakeChatLLM(reply=reply)
    return g


def test_generate_uses_system_and_user_chat_messages():
    g = _gen_with_fake(reply="use llm")
    out = g.generate("how do headings work", ["# H1 is topmost", "## H2 sits below"])
    assert "use llm" in out

    msgs = g.llm.last_chat
    assert len(msgs) == 2
    sys_text = msgs[0].content.lower()
    assert "i don't know" in sys_text or "do not invent" in sys_text
    user_text = msgs[1].content
    assert "[1]" in user_text and "[2]" in user_text
    assert "how do headings work" in user_text


def test_generate_accepts_dict_contexts():
    g = _gen_with_fake()
    g.generate(
        "q",
        [{"content": "alpha"}, {"content": "beta"}],
    )
    user_text = g.llm.last_chat[1].content
    assert "[1] alpha" in user_text
    assert "[2] beta" in user_text
