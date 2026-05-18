"""
Configuration management for the AI client.

Reads provider config from credentials.json (via faas_gauge.config).
No YAML dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faas_gauge.config import get_data_dir, load_credentials

from .exceptions import ConfigurationError


@dataclass
class ProviderConfig:
    """Configuration for a single AI provider."""

    api_base: str
    api_key: str
    system_prompt_id: str = "system"
    models: list[str] = field(default_factory=list)
    chain_as_system_models: list[str] = field(default_factory=list)
    completion_models: list[str] = field(default_factory=list)
    timeout: int = 30
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.api_base:
            raise ConfigurationError("api_base is required")
        if not self.api_key:
            raise ConfigurationError("api_key is required")

    def all_models(self) -> list[str]:
        """Return all available models across all categories."""
        return self.models + self.chain_as_system_models + self.completion_models


@dataclass
class Config:
    """Main configuration for the AI client."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    system_prompt: str = ""
    model_pricing: dict[str, tuple] = field(default_factory=dict)
    model_mappings: dict[str, str] = field(default_factory=dict)
    default_timeout: int = 30
    default_max_retries: int = 3
    log_level: str = "WARNING"

    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider config by name. Raises ConfigurationError if not found."""
        if name not in self.providers:
            raise ConfigurationError(f"Provider '{name}' not found in configuration")
        return self.providers[name]

    def add_provider(self, name: str, config: ProviderConfig) -> None:
        """Add or replace a provider configuration."""
        self.providers[name] = config

    def get_available_providers(self) -> list[str]:
        """Return list of configured provider names."""
        return list(self.providers.keys())

    def get_available_models(self, provider_name: str) -> list[str]:
        """Return all models for a provider."""
        return self.get_provider(provider_name).all_models()


def load_config(
    config_path: str | None = None,
    data_dir: str | Path | None = None,
) -> Config:
    """
    Load AI configuration from credentials.json.

    Args:
        config_path: Explicit path to credentials.json. If None, auto-discovers.
        data_dir: Explicit data directory for system prompt resolution.

    Returns:
        Config with all providers loaded.
    """
    creds: dict[str, Any] = load_credentials(config_path)
    config = Config()

    settings = creds.get("settings", {})
    config.default_timeout = settings.get("default_timeout", 30)
    config.log_level = settings.get("log_level", "WARNING")

    prompt_file = settings.get("system_prompt_file")
    if prompt_file:
        resolved_data_dir = Path(data_dir) if data_dir else get_data_dir()
        prompt_path = resolved_data_dir / prompt_file
        if prompt_path.is_file():
            config.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    ai_providers = creds.get("ai_providers", {})
    for provider_name, provider_data in ai_providers.items():
        try:
            provider_config = ProviderConfig(
                api_base=provider_data["api_base"],
                api_key=provider_data["api_key"],
                system_prompt_id=provider_data.get("system_prompt_id", "system"),
                models=provider_data.get("models", []),
                chain_as_system_models=provider_data.get("chain_as_system_models", []),
                completion_models=provider_data.get("completion_models", []),
                timeout=provider_data.get("timeout", config.default_timeout),
                max_retries=provider_data.get("max_retries", 3),
            )
            config.add_provider(provider_name, provider_config)
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing required field in provider '{provider_name}': {exc}"
            ) from exc

    for provider_name in config.providers:
        env_key = f"FAAS_GAUGE_AI_{provider_name.upper()}_API_KEY"
        if env_key in os.environ:
            config.providers[provider_name].api_key = os.environ[env_key]

    return config
