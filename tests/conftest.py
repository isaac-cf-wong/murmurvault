"""Configuration and fixtures for pytest."""

from __future__ import annotations

import pytest

from murmurvault import config as config_mod


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point murmurvault at a fresh vault and a config file that does not exist.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The effective configuration, with embeddings and reranking disabled.
    """
    monkeypatch.setenv("MURMURVAULT_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("MURMURVAULT_VAULT", str(tmp_path / "vault"))
    cfg = config_mod.load()
    cfg["embedding"]["backend"] = "none"
    cfg["rerank"]["backend"] = "none"
    return cfg
