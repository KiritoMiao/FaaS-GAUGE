"""
Configuration loading and data directory discovery.

Auto-discovery search order for data directory:
  1. FAAS_GAUGE_DATA_DIR environment variable
  2. ./data/ relative to CWD
  3. Walk up parent directories looking for data/config/ (max 5 levels)

Auto-discovery search order for credentials.json:
  1. FAAS_GAUGE_CREDENTIALS environment variable
  2. {data_dir}/config/credentials.json
"""

import json
import os
from pathlib import Path
from typing import Any


def get_data_dir() -> Path:
    """
    Discover the data directory.

    Returns:
        Path to the data directory.

    Raises:
        FileNotFoundError: If no data directory can be found.
    """
    env_dir = os.environ.get("FAAS_GAUGE_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.is_dir():
            return env_path
        raise FileNotFoundError(f"FAAS_GAUGE_DATA_DIR points to non-existent directory: {env_dir}")

    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir() and (cwd_data / "config").is_dir():
        return cwd_data

    current = Path.cwd()
    for _ in range(5):
        candidate = current / "data"
        if candidate.is_dir() and (candidate / "config").is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        "Cannot find data directory. Set FAAS_GAUGE_DATA_DIR or create data/config/ in your project."
    )


def load_credentials(path: str | None = None) -> dict[str, Any]:
    """
    Load credentials from a JSON file.

    Args:
        path: Explicit path to credentials.json. If None, auto-discovers.

    Returns:
        Parsed credentials dictionary.

    Raises:
        FileNotFoundError: If credentials file cannot be found.
    """
    if path is not None:
        creds_path = Path(path)
        if not creds_path.is_file():
            raise FileNotFoundError(f"Credentials file not found: {path}")
        return _read_json(creds_path)

    env_creds = os.environ.get("FAAS_GAUGE_CREDENTIALS")
    if env_creds:
        creds_path = Path(env_creds)
        if not creds_path.is_file():
            raise FileNotFoundError(
                f"FAAS_GAUGE_CREDENTIALS points to non-existent file: {env_creds}"
            )
        return _read_json(creds_path)

    try:
        data_dir = get_data_dir()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Cannot find credentials.json. "
            "Set FAAS_GAUGE_CREDENTIALS or create data/config/credentials.json"
        ) from exc

    creds_path = data_dir / "config" / "credentials.json"
    if not creds_path.is_file():
        raise FileNotFoundError(
            f"Credentials file not found at {creds_path}. "
            "Copy credentials.json.example and fill in your API keys."
        )
    return _read_json(creds_path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data
