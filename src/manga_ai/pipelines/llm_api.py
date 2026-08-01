"""Remote LLM clients for story and panel planning."""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import requests


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("response") or item.get("value")
                if value is not None:
                    parts.append(str(value))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        value = content.get("text") or content.get("content") or content.get("response") or content.get("value")
        if value is not None:
            return str(value)
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


@dataclass
class CloudflareChatClient:
    """Small OpenAI-compatible wrapper over Cloudflare Workers AI text models."""

    account_id: str
    api_token: str

    def __post_init__(self):
        self.chat = SimpleNamespace(completions=_CloudflareChatCompletions(self.account_id, self.api_token))


class _CloudflareChatCompletions:
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token

    def create(self, model: str, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None, **kwargs):
        if not self.account_id:
            raise ValueError("Missing Cloudflare account ID. Set CLOUDFLARE_ACCOUNT_ID.")
        if not self.api_token:
            raise ValueError("Missing Cloudflare API token. Set CLOUDFLARE_API_TOKEN.")
        if not model:
            raise ValueError("Missing Cloudflare LLM model. Set LLM_MODEL.")

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model}"
        payload = {"messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 401:
                raise RuntimeError(
                    "Cloudflare LLM authentication failed (401). Check CLOUDFLARE_API_TOKEN, "
                    "Workers AI permissions, and CLOUDFLARE_ACCOUNT_ID account scope. "
                    f"Cloudflare response: {response.text[:1000]}"
                ) from exc
            raise RuntimeError(f"Cloudflare LLM request failed: {response.status_code} {response.text[:1000]}") from exc

        payload = response.json()
        if not payload.get("success", True):
            raise RuntimeError(f"Cloudflare LLM request failed: {payload}")

        result = payload.get("result", payload)
        content = ""
        if isinstance(result, dict):
            content = (
                result.get("response")
                or result.get("text")
                or result.get("content")
                or result.get("answer")
                or ""
            )
        elif isinstance(result, str):
            content = result
        elif isinstance(result, list):
            content = result
        content = _content_to_text(content)
        if not content:
            raise RuntimeError(f"Cloudflare LLM response did not include text: {payload}")

        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def create_llm_client(config):
    provider = getattr(config.model, "llm_provider", "huggingface").lower()
    if provider in ("none", "fallback", "local"):
        return None
    if provider == "cloudflare":
        token = (
            getattr(config.image_api, "cloudflare_api_token", None)
            or getattr(config.image_api, "api_key", None)
            or getattr(config.model, "REMOVED_TOKENtoken", None)
        )
        return CloudflareChatClient(
            account_id=getattr(config.image_api, "cloudflare_account_id", None),
            api_token=token,
        )
    if provider == "huggingface":
        try:
            from openai import OpenAI
        except ImportError:
            return None
        token = getattr(config.model, "REMOVED_TOKENtoken", None)
        if not token:
            return None
        return OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
    raise ValueError(f"Unsupported LLM provider: {provider}")
