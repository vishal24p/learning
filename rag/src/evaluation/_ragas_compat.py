"""Stub for ragas 0.2.x compat: langchain-community 0.4.x removed the
``langchain_community.chat_models.VertexAI`` alias, but ragas still
imports it. We re-export ``from langchain_community.llms.VertexAI`` here
so the import succeeds without pulling in google-cloud-aiplatform.
"""
from langchain_community.llms import VertexAI as _LegacyVertex

# Mirror the attributes ragas might look at; harmless if unused.
class _ChatVertexStub:
    pass


# Alias the legacy LLM class under the chat_models namespace so
# ``from langchain_community.chat_models.vertexai import ChatVertexAI``
# resolves in ragas.
ChatVertexAI = _LegacyVertex
