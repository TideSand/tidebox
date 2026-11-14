"""Provider adapters: the provider's wire format <-> the tape's normalized records.

A provider is an API shape (Anthropic Messages, OpenAI Chat Completions / Responses), not a
harness. A recording uses an adapter to normalize what it records; the controller uses it to
turn a tape step back into the bytes a real harness expects, and to match an incoming request
to a step. 

Adapters are looked up by the `provider` string stored in `TapeMeta`.
"""

from __future__ import annotations

from tidebox.api.providers.base import Provider

_PROVIDERS: dict[str, type[Provider]] = {}


def register(cls: type[Provider]) -> type[Provider]:
    _PROVIDERS[cls.name] = cls
    return cls


def get_provider(name: str) -> Provider:
    _load()
    try:
        return _PROVIDERS[name]()
    except KeyError:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(_PROVIDERS)}") from None


def all_providers() -> list[Provider]:
    _load()
    return [cls() for cls in _PROVIDERS.values()]


def detect_provider(path: str, headers: dict[str, str] | None = None) -> Provider | None:
    """The provider whose model-call endpoint a request path is, or None for anything else
    (health checks, token counting, model listings)."""
    for provider in all_providers():
        if provider.is_model_call(path):
            return provider
    return None


def _load() -> None:
    if _PROVIDERS:
        return
    from tidebox.api.providers.anthropic import AnthropicProvider
    from tidebox.api.providers.openai import OpenAIProvider

    register(AnthropicProvider)
    register(OpenAIProvider)
