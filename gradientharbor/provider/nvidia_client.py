"""LangChain ChatOpenAI adapter targeting NVIDIA's OpenAI-compatible endpoint."""
from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
class NVIDIAConfigurationError(RuntimeError): pass

def get_client(api_key: str | None = None, model: str | None = None, temperature: float = 0.2, max_tokens: int = 3000):
    key=(api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
    if not key:
        raise NVIDIAConfigurationError("NVIDIA API key is missing. Add NVIDIA_API_KEY to .env or enter it in the app sidebar.")
    return ChatOpenAI(model=model or os.getenv("NVIDIA_MODEL", DEFAULT_MODEL), api_key=key, base_url=os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL),
                      temperature=temperature, max_tokens=max_tokens, timeout=90, max_retries=2)

def chat(messages: list[dict], *, api_key: str | None = None, model: str | None = None,
         temperature: float = 0.2, max_tokens: int = 3000) -> str:
    llm=get_client(api_key, model, temperature, max_tokens)
    response=llm.invoke(messages)
    content=response.content
    if isinstance(content, list):
        content="\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
    if not content: raise RuntimeError("Nemotron returned an empty response.")
    return str(content).strip()

def check_client(api_key: str | None = None, model: str | None = None) -> str:
    return chat([{"role":"system","content":"You are a concise assistant."},
                 {"role":"user","content":"Reply with exactly: NVIDIA Nemotron connection successful."}],
                api_key=api_key, model=model, temperature=0, max_tokens=64)
