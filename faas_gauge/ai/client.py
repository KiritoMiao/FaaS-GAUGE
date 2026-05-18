"""Main AI client implementation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from .config import ProviderConfig, load_config
from .exceptions import (
    AuthenticationError,
    ModelError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)
from .utils import calculate_cost, extract_python_code, parse_error_response


@dataclass
class AIResponse:
    """Response object containing all metrics and results."""

    raw_result: str
    python_code: str
    input_tokens: int
    output_tokens: int
    tokens_per_second: float | None
    cost: float | None
    rtt_sec: float
    provider: str
    model: str
    error: str | None = None


class AIClient:
    """Unified client for multiple AI providers."""

    def __init__(self, config_path: str | None = None, data_dir: str | Path | None = None):
        self.config = load_config(config_path, data_dir)
        self.ai_clients: dict[str, Any] = {}
        self.logger = self._setup_logging()
        self._initialize_clients()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("faas_gauge.ai")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_clients(self) -> None:
        for provider_name, provider_config in self.config.providers.items():
            try:
                self.ai_clients[provider_name] = openai.OpenAI(
                    api_key=provider_config.api_key,
                    base_url=provider_config.api_base,
                    timeout=provider_config.timeout,
                    max_retries=provider_config.max_retries,
                )
            except Exception as exc:
                raise ProviderError(
                    f"Failed to initialize {provider_name}", provider_name, exc
                ) from exc

    def _prepare_messages(
        self, question: str, provider_name: str, model: str
    ) -> list[dict[str, str]]:
        provider_config = self.config.get_provider(provider_name)
        if model in provider_config.chain_as_system_models:
            return [
                {
                    "role": "user",
                    "content": self.config.system_prompt + "\n\n" + question,
                }
            ]
        return [
            {
                "role": provider_config.system_prompt_id,
                "content": self.config.system_prompt,
            },
            {"role": "user", "content": question},
        ]

    def _prepare_completion_prompt(self, question: str, provider_name: str, model: str) -> str:
        del provider_name, model
        return self.config.system_prompt + "\n\n" + question

    def _handle_api_error(self, error: Exception, provider: str, model: str) -> Exception:
        error_info = parse_error_response(error)
        if error_info["is_auth_error"]:
            return AuthenticationError(
                f"Authentication failed for {provider}",
                provider=provider,
                original_error=error,
            )
        if error_info["is_rate_limit"]:
            return RateLimitError(
                f"Rate limit exceeded for {provider}",
                provider=provider,
                original_error=error,
            )
        if error_info["is_quota_error"]:
            return QuotaExceededError(
                f"Quota exceeded for {provider}",
                provider=provider,
                original_error=error,
            )
        return ModelError(
            f"API error for {provider}/{model}: {error}",
            provider=provider,
            model=model,
            original_error=error,
        )

    def ask(self, question: str, provider: str, model: str) -> AIResponse:
        if provider not in self.ai_clients:
            raise ProviderError(f"Provider '{provider}' not available", provider)

        client = self.ai_clients[provider]
        provider_config = self.config.get_provider(provider)
        start_time = time.time()

        try:
            if model in provider_config.completion_models:
                prompt = self._prepare_completion_prompt(question, provider, model)
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0.7,
                )
                raw_result = response.choices[0].text
            else:
                messages = self._prepare_messages(question, provider, model)
                response = client.chat.completions.create(model=model, messages=messages)
                raw_result = response.choices[0].message.content

            rtt_sec = time.time() - start_time
            python_code = extract_python_code(raw_result)
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            tokens_per_second = output_tokens / rtt_sec if rtt_sec > 0 else None
            cost = calculate_cost(
                model,
                input_tokens,
                output_tokens,
                self.config.model_pricing,
                self.config.model_mappings,
            )

            return AIResponse(
                raw_result=raw_result,
                python_code=python_code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokens_per_second=tokens_per_second,
                cost=cost,
                rtt_sec=rtt_sec,
                provider=provider,
                model=model,
            )
        except Exception as exc:
            rtt_sec = time.time() - start_time
            handled_error = self._handle_api_error(exc, provider, model)
            return AIResponse(
                raw_result=f"Error: {str(handled_error)}",
                python_code="",
                input_tokens=0,
                output_tokens=0,
                tokens_per_second=None,
                cost=None,
                rtt_sec=rtt_sec,
                provider=provider,
                model=model,
                error=str(handled_error),
            )

    def providers(self) -> list[str]:
        return self.config.get_available_providers()

    def models(self, provider: str) -> list[str]:
        return self.config.get_available_models(provider)

    def status(self) -> dict[str, dict[str, Any]]:
        provider_status: dict[str, dict[str, Any]] = {}
        for provider_name in self.config.providers:
            provider_status[provider_name] = {
                "connected": provider_name in self.ai_clients,
                "models": self.models(provider_name),
                "config": {
                    "api_base": self.config.providers[provider_name].api_base,
                    "timeout": self.config.providers[provider_name].timeout,
                    "max_retries": self.config.providers[provider_name].max_retries,
                },
            }
        return provider_status

    def add_provider(self, name: str, api_base: str, api_key: str, **kwargs: Any) -> None:
        provider_config = ProviderConfig(api_base=api_base, api_key=api_key, **kwargs)
        self.config.add_provider(name, provider_config)
        try:
            self.ai_clients[name] = openai.OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=provider_config.timeout,
                max_retries=provider_config.max_retries,
            )
        except Exception as exc:
            raise ProviderError(f"Failed to initialize new provider {name}", name, exc) from exc

    def update_system_prompt(self, new_prompt: str) -> None:
        self.config.system_prompt = new_prompt
