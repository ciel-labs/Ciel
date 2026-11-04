"""Tests for src/Ciel/config.py — application configuration.

All tests are offline and fast. They verify:
- Default values when no env vars are set
- Env var override via monkeypatch
- SecretStr masking of api_key
- has_api_key() logic
- Immutability (frozen=True)

We never use a real API key in these tests.
"""

import pytest
from pydantic import SecretStr

from Ciel.config import AppConfig


class TestDefaults:
    """AppConfig constructed with no env vars must have correct defaults."""

    def test_project_name_default(self):
        config = AppConfig()
        assert config.project_name == "Ciel"

    def test_version_default(self):
        config = AppConfig()
        assert config.version == "0.1.0"

    def test_api_key_default_is_empty(self):
        """An unconfigured api_key must be an empty SecretStr, not None."""
        config = AppConfig()
        assert config.api_key.get_secret_value() == ""

    def test_api_base_url_default_is_qwen(self):
        config = AppConfig()
        assert "dashscope.aliyuncs.com" in config.api_base_url

    def test_model_default(self):
        config = AppConfig()
        assert config.model == "qwen-max"

    def test_request_timeout_default(self):
        config = AppConfig()
        assert config.request_timeout == 30.0

    def test_max_retries_default(self):
        config = AppConfig()
        assert config.max_retries == 3

    def test_has_api_key_false_by_default(self):
        """has_api_key() must return False when no key is configured."""
        config = AppConfig()
        assert config.has_api_key() is False


class TestEnvVarOverride:
    """Settings can be overridden via CIEL_* environment variables."""

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("CIEL_API_KEY", "test-secret-key-123")
        config = AppConfig()
        assert config.api_key.get_secret_value() == "test-secret-key-123"

    def test_has_api_key_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("CIEL_API_KEY", "some-key")
        config = AppConfig()
        assert config.has_api_key() is True

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("CIEL_MODEL", "qwen-turbo")
        config = AppConfig()
        assert config.model == "qwen-turbo"

    def test_api_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("CIEL_API_BASE_URL", "http://localhost:8080/v1")
        config = AppConfig()
        assert config.api_base_url == "http://localhost:8080/v1"

    def test_request_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("CIEL_REQUEST_TIMEOUT", "60.0")
        config = AppConfig()
        assert config.request_timeout == 60.0

    def test_max_retries_from_env(self, monkeypatch):
        monkeypatch.setenv("CIEL_MAX_RETRIES", "5")
        config = AppConfig()
        assert config.max_retries == 5

    def test_unknown_env_var_ignored(self, monkeypatch):
        """Extra CIEL_* env vars must not raise an error (extra='ignore')."""
        monkeypatch.setenv("CIEL_UNKNOWN_SETTING", "whatever")
        AppConfig()  # must not raise


class TestSecretStrMasking:
    """api_key must never appear in repr or str output."""

    def test_api_key_masked_in_repr(self, monkeypatch):
        monkeypatch.setenv("CIEL_API_KEY", "super-secret-key")
        config = AppConfig()
        representation = repr(config)
        assert "super-secret-key" not in representation

    def test_api_key_masked_in_str(self, monkeypatch):
        monkeypatch.setenv("CIEL_API_KEY", "super-secret-key")
        config = AppConfig()
        assert "super-secret-key" not in str(config)

    def test_api_key_is_secret_str_type(self):
        """api_key must be a SecretStr instance, not a plain str."""
        config = AppConfig()
        assert isinstance(config.api_key, SecretStr)

    def test_get_secret_value_returns_plain_string(self, monkeypatch):
        """get_secret_value() must return the actual key string."""
        monkeypatch.setenv("CIEL_API_KEY", "my-plain-key")
        config = AppConfig()
        assert config.api_key.get_secret_value() == "my-plain-key"


class TestImmutability:
    """AppConfig must be frozen (immutable after construction)."""

    def test_cannot_set_project_name(self):
        config = AppConfig()
        with pytest.raises(Exception):  # ValidationError or TypeError
            config.project_name = "changed"  # type: ignore[misc]

    def test_cannot_set_model(self):
        config = AppConfig()
        with pytest.raises(Exception):
            config.model = "changed"  # type: ignore[misc]

    def test_cannot_set_api_key(self):
        config = AppConfig()
        with pytest.raises(Exception):
            config.api_key = SecretStr("changed")  # type: ignore[misc]
