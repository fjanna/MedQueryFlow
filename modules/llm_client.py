"""Multi-provider LLM client with automatic backend detection."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Container describing an LLM prompt invocation."""

    prompt: str
    model: str = "auto"
    temperature: float = 0.3


class LLMClient:
    """Dispatch chat requests to the first detected LLM provider."""

    _ENV_TO_PROVIDER = {
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        "GEMINI_API_KEY": "gemini",
        "DEEPSEEK_API_KEY": "deepseek",
    }

    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-sonnet-20240229",
        "gemini": "gemini-1.5-pro",
        "deepseek": "deepseek-chat",
    }

    def __init__(self) -> None:
        provider, api_key = self._detect_provider()
        self.provider = provider
        self.api_key = api_key
        self._client = self._initialize_client(provider, api_key)
        self._default_model = self._DEFAULT_MODELS[provider]
        logger.info(
            "Initialized LLM client", extra={"provider": provider, "default_model": self._default_model}
        )

    def _detect_provider(self) -> Tuple[str, str]:
        for env_var, provider in self._ENV_TO_PROVIDER.items():
            api_key = os.environ.get(env_var)
            if api_key:
                logger.debug("Detected API key", extra={"provider": provider, "env_var": env_var})
                return provider, api_key
        raise RuntimeError("未检测到任何 LLM Key")

    def _initialize_client(self, provider: str, api_key: str):
        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("未找到 openai SDK，请安装 openai>=1.0") from exc
            return OpenAI(api_key=api_key)

        if provider == "deepseek":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("未找到 openai SDK，请安装 openai>=1.0") from exc
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("未找到 anthropic SDK，请安装 anthropic") from exc
            return anthropic.Anthropic(api_key=api_key)

        if provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise RuntimeError(
                    "未找到 google-generativeai SDK，请安装 google-generativeai"
                ) from exc
            genai.configure(api_key=api_key)
            return genai

        raise ValueError(f"Unsupported provider: {provider}")

    def _resolve_model(self, request: LLMRequest) -> str:
        if request.model and request.model != "auto":
            return request.model
        return self._default_model

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
                messages=[{"role": "user", "content": request.prompt}],
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
            if not text:
                logger.error("Gemini 返回内容为空")
                return ""
            logger.debug("Received response", extra={"provider": "gemini", "length": len(text)})
            return text

        raise ValueError(f"Unsupported provider: {self.provider}")

