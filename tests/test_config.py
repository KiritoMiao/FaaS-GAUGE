"""Tests for faas_gauge.config module."""

import json
import os
from pathlib import Path

import pytest

from faas_gauge.config import get_data_dir, load_credentials


@pytest.fixture
def sample_credentials(tmp_path: Path) -> Path:
    """Create a temporary credentials.json file."""
    creds = {
        "ai_providers": {
            "openai": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "sk-test-key",
                "system_prompt_id": "developer",
                "models": ["gpt-4o-mini"],
                "chain_as_system_models": [],
                "completion_models": [],
                "timeout": 30,
                "max_retries": 3,
            }
        },
        "aws": {"profile": "test", "region": "us-east-1"},
        "settings": {"log_level": "DEBUG", "default_timeout": 60},
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    creds_path = config_dir / "credentials.json"
    creds_path.write_text(json.dumps(creds), encoding="utf-8")
    return creds_path


def test_load_credentials_explicit_path(sample_credentials: Path) -> None:
    """Load credentials from an explicit path."""
    creds = load_credentials(str(sample_credentials))
    assert "ai_providers" in creds
    assert creds["ai_providers"]["openai"]["api_key"] == "sk-test-key"


def test_load_credentials_missing_file() -> None:
    """Raise FileNotFoundError for missing credentials."""
    with pytest.raises(FileNotFoundError):
        load_credentials("/nonexistent/credentials.json")


def test_get_data_dir_explicit(tmp_path: Path) -> None:
    """get_data_dir with FAAS_GAUGE_DATA_DIR env var."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config").mkdir()

    os.environ["FAAS_GAUGE_DATA_DIR"] = str(data_dir)
    try:
        result = get_data_dir()
        assert result == data_dir
    finally:
        del os.environ["FAAS_GAUGE_DATA_DIR"]


def test_get_data_dir_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_data_dir finds data/ in CWD."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config").mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAAS_GAUGE_DATA_DIR", raising=False)
    result = get_data_dir()
    assert result == data_dir
