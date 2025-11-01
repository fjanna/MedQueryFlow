"""Multi-provider LLM client with automatic backend detection."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Container describing an LLM prompt invocation."""

    prompt: str
    model: str = "auto"
    temperature: float = 0.3


class LLMClient:
    """Dispatch chat requests to the first detected LLM provider."""

    _PROVIDER_PRIORITY = (
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("GEMINI_API_KEY", "gemini"),
        ("DEEPSEEK_API_KEY", "deepseek"),
    )

    _DEFAULT_MODELS: Dict[str, str] = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-sonnet-20240229",
        "gemini": "gemini-1.5-pro",
        "deepseek": "deepseek-chat",
    }

    _MODEL_ALIASES: Dict[str, Dict[str, str]] = {
        "deepseek": {
            "gpt-4o-mini": "deepseek-chat",
            "gpt-4o": "deepseek-chat",
            "gpt-4-turbo": "deepseek-chat",
            "claude-3-sonnet-20240229": "deepseek-chat",
            "gemini-1.5-pro": "deepseek-chat",
        }
    }

    _SUPPORTED_MODELS: Dict[str, Tuple[str, ...]] = {
        "deepseek": ("deepseek-chat", "deepseek-reasoner"),
    }

    def __init__(self) -> None:
        provider, api_key = self._detect_provider()
        self.provider = provider
        self.api_key = api_key
        self._client = None
        self._default_model = self._DEFAULT_MODELS[provider]
        self._deepseek_fallback = False
        self._initialize_client()
        logger.info("✅ 已启用 %s 模型后端。", self.provider.upper())

    def _detect_provider(self) -> Tuple[str, str]:
        for env_var, provider in self._PROVIDER_PRIORITY:
            api_key = os.environ.get(env_var)
            if api_key:
                logger.debug(
                    "Detected API key", extra={"provider": provider, "env_var": env_var}
                )
                return provider, api_key
        raise RuntimeError("未检测到任何 LLM Key")

    def _initialize_client(self) -> None:
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("未找到 openai SDK，请安装 openai>=1.0") from exc
            self._client = OpenAI(api_key=self.api_key, base_url="https://api.openai.com/v1")
            return

        if self.provider == "deepseek":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("未找到 openai SDK，请安装 openai>=1.0") from exc
            try:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com/v1",
                )
            except (TypeError, AttributeError) as exc:
                logger.warning(
                    "OpenAI SDK initialization failed for DeepSeek, switching to HTTP fallback: %s",
                    exc,
                )
                self._deepseek_fallback = True
                self._client = None
            return

        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("未找到 anthropic SDK，请安装 anthropic") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
            return

        if self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise RuntimeError(
                    "未找到 google-generativeai SDK，请安装 google-generativeai"
                ) from exc
            genai.configure(api_key=self.api_key)
            self._client = genai
            return

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _resolve_model(self, request: LLMRequest) -> str:
        if request.model and request.model != "auto":
            requested = request.model
            alias = self._MODEL_ALIASES.get(self.provider, {}).get(requested)
            if alias:
                logger.debug(
                    "Using provider model alias",
                    extra={
                        "provider": self.provider,
                        "requested": requested,
                        "resolved": alias,
                    },
                )
                return alias

            supported = self._SUPPORTED_MODELS.get(self.provider)
            if supported and requested not in supported:
                logger.warning(
                    "Model %s is not supported for provider %s; falling back to default %s",
                    requested,
                    self.provider,
                    self._default_model,
                )
                return self._default_model
            return requested
        return self._default_model

    def _fallback_deepseek(self, prompt: str, model: str, temperature: float) -> str:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model or "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = requests.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail: str
            try:
                detail_payload = response.json()
                detail = str(detail_payload)
            except ValueError:
                detail = response.text
            logger.error("DeepSeek HTTP fallback failed: %s", detail)
            raise RuntimeError(f"DeepSeek request failed with status {response.status_code}: {detail}") from exc
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek response structure: {data}") from exc

    def _extract_openai_message(self, response) -> str:
        message = response.choices[0].message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(item.text)
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def generate(self, request: LLMRequest) -> str:
        model = self._resolve_model(request)
        logger.debug(
            "Invoking LLM",
            extra={"provider": self.provider, "model": model, "temperature": request.temperature},
        )

        if self.provider == "deepseek" and self._deepseek_fallback:
            content = self._fallback_deepseek(request.prompt, model, request.temperature)
            logger.debug("Received response", extra={"provider": "deepseek", "length": len(content)})
            return content

        if self.provider in {"openai", "deepseek"}:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
            )
            content = self._extract_openai_message(response)
            logger.debug(
                "Received response",
                extra={"provider": self.provider, "length": len(content)},
            )
            return content

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": request.prompt}],
                    }
                ],
                temperature=request.temperature,
                max_tokens=4096,
            )
            parts = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            content = "".join(parts)
            logger.debug("Received response", extra={"provider": "anthropic", "length": len(content)})
            return content

        if self.provider == "gemini":
            generative_model = self._client.GenerativeModel(model)
            response = generative_model.generate_content(
                request.prompt,
                generation_config={"temperature": request.temperature},
            )
            text = getattr(response, "text", None)
            if not text and getattr(response, "candidates", None):
                texts = []
                for candidate in response.candidates:
                    if getattr(candidate, "content", None):
                        for part in candidate.content.parts:
                            if getattr(part, "text", None):
                                texts.append(part.text)
                text = "".join(texts)
            if not text:
                logger.error("Gemini 返回内容为空")
                return ""
            logger.debug("Received response", extra={"provider": "gemini", "length": len(text)})
            return text

        raise ValueError(f"Unsupported provider: {self.provider}")
