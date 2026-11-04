"""Application configuration.

All LLM-related settings are read from environment variables prefixed
with CIEL_. A .env file in the project root is loaded automatically
if it exists, but it is never required — env vars take precedence.

Environment variables
---------------------
CIEL_API_KEY          API key for the Qwen cloud service. Required when
                      making real API calls; safe to omit in tests that
                      mock the HTTP layer.
CIEL_API_BASE_URL     OpenAI-compatible base URL.
                      Default: https://dashscope.aliyuncs.com/compatible-mode/v1
CIEL_MODEL            Model identifier. Default: qwen-max
CIEL_REQUEST_TIMEOUT  Per-request HTTP timeout in seconds. Default: 30.0
CIEL_MAX_RETRIES      Max retry attempts on transient errors. Default: 3

Security notes
--------------
- api_key is stored as SecretStr. It is never included in repr() or
  str() output, which prevents accidental leakage into logs.
- Use config.api_key.get_secret_value() only at the moment a request
  is built. Never store the plain string in a variable that outlives
  the request.
- project_name and version are intentionally not overridable via env
  to prevent accidental identity spoofing.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Configuration for the Ciel agent.

    Pydantic-settings automatically reads fields from:
    1. Environment variables (highest priority)
    2. A .env file in the working directory (if present)
    3. The default values defined here (lowest priority)

    All LLM-related env vars must be prefixed with CIEL_ (see module
    docstring). project_name and version are hard-coded constants and
    cannot be overridden via env.
    """

    model_config = SettingsConfigDict(
        env_prefix="CIEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",       # silently ignore unknown CIEL_* env vars
        frozen=True,          # immutable after construction, like before
    )

    # ── Project identity (hard-coded, not from env) ────────────────────
    # These are intentionally excluded from env-var override.
    project_name: str = "Ciel"
    version: str = "0.1.0"

    # ── LLM settings (all from CIEL_* env vars) ───────────────────────

    api_key: SecretStr = SecretStr("")
    """API key for the Qwen cloud service.

    Stored as SecretStr — will appear as '**********' in repr/logs.
    Only call .get_secret_value() at the moment you build an HTTP request.
    An empty string means no key is configured; LLMClient will raise
    LLMAuthError when it attempts a real call without a key.
    """

    api_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    """OpenAI-compatible base URL for the Qwen cloud API.

    Override with CIEL_API_BASE_URL to point at a local proxy, a
    different cloud region, or a local llama.cpp server in Phase 10.
    """

    model: str = "qwen-max"
    """Qwen model identifier sent in every API request.

    Common values:
      qwen-max    — highest capability (default)
      qwen-plus   — balanced speed/cost
      qwen-turbo  — fastest, lowest cost
    """

    request_timeout: float = 30.0
    """Per-request HTTP timeout in seconds.

    If the API does not respond within this window, LLMClient raises
    LLMTimeoutError. Increase for slow connections or large prompts.
    Must be positive.
    """

    max_retries: int = 3
    """Maximum number of retry attempts for transient errors (429, 5xx).

    Set to 0 to disable retries entirely.
    Must be non-negative.
    """

    def has_api_key(self) -> bool:
        """Return True if an API key has been configured.

        Use this to give a clear error message before attempting a call,
        rather than letting the API return a 401.
        """
        return bool(self.api_key.get_secret_value())
