"""Tests for faas_gauge.ai.client module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from faas_gauge.ai.client import AIClient, AIResponse
from faas_gauge.ai.config import Config, ProviderConfig


def _make_config() -> Config:
    config = Config(system_prompt="You are a coding assistant.")
    config.add_provider(
        "openai",
        ProviderConfig(
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
            system_prompt_id="system",
            models=["gpt-4o-mini"],
            chain_as_system_models=[],
            completion_models=[],
            timeout=30,
            max_retries=3,
        ),
    )
    return config


@patch("faas_gauge.ai.client.openai.OpenAI")
@patch("faas_gauge.ai.client.load_config")
def test_ai_client_constructor(mock_load_config: MagicMock, mock_openai: MagicMock) -> None:
    """AIClient loads config and initializes provider clients."""
    config = _make_config()
    mock_load_config.return_value = config

    client = AIClient(config_path="/tmp/credentials.json")

    assert client.config is config
    mock_load_config.assert_called_once_with("/tmp/credentials.json", None)
    mock_openai.assert_called_once()


@patch("faas_gauge.ai.client.openai.OpenAI")
@patch("faas_gauge.ai.client.load_config")
def test_providers_returns_names(mock_load_config: MagicMock, mock_openai: MagicMock) -> None:
    config = _make_config()
    mock_load_config.return_value = config

    client = AIClient()

    assert client.providers() == ["openai"]


@patch("faas_gauge.ai.client.openai.OpenAI")
@patch("faas_gauge.ai.client.load_config")
def test_models_returns_provider_models(
    mock_load_config: MagicMock, mock_openai: MagicMock
) -> None:
    config = _make_config()
    mock_load_config.return_value = config

    client = AIClient()

    assert client.models("openai") == ["gpt-4o-mini"]


@patch("faas_gauge.ai.client.openai.OpenAI")
@patch("faas_gauge.ai.client.load_config")
def test_ask_returns_ai_response(mock_load_config: MagicMock, mock_openai: MagicMock) -> None:
    config = _make_config()
    mock_load_config.return_value = config

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="```python\nprint('ok')\n```"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
    )
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = fake_response
    mock_openai.return_value = openai_client

    client = AIClient()
    result = client.ask("Write python", provider="openai", model="gpt-4o-mini")

    assert isinstance(result, AIResponse)
    assert result.raw_result == "```python\nprint('ok')\n```"
    assert result.python_code == "print('ok')"
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.error is None


@patch("faas_gauge.ai.client.openai.OpenAI")
@patch("faas_gauge.ai.client.load_config")
def test_ask_handles_api_error(mock_load_config: MagicMock, mock_openai: MagicMock) -> None:
    config = _make_config()
    mock_load_config.return_value = config

    openai_client = MagicMock()
    openai_client.chat.completions.create.side_effect = Exception("rate limit exceeded")
    mock_openai.return_value = openai_client

    client = AIClient()
    result = client.ask("Write python", provider="openai", model="gpt-4o-mini")

    assert isinstance(result, AIResponse)
    assert result.error is not None
    assert "Rate limit exceeded for openai" in result.error
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@patch("faas_gauge.ai.client.openai.OpenAI")
@patch("faas_gauge.ai.client.load_config")
def test_status_returns_connection_info(
    mock_load_config: MagicMock, mock_openai: MagicMock
) -> None:
    config = _make_config()
    mock_load_config.return_value = config

    client = AIClient()
    status = client.status()

    assert "openai" in status
    assert status["openai"]["connected"] is True
    assert status["openai"]["models"] == ["gpt-4o-mini"]
    assert status["openai"]["config"]["api_base"] == "https://api.openai.com/v1"
