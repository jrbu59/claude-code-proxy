import os
import sys
from typing import Dict, List, Optional, Set

from urllib.parse import urlparse

from src.core.tool_policies import normalize_group_names

# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")
        
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.log_file_path = os.environ.get("LOG_FILE_PATH", "logs/claude-proxy.log")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        
        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        self.retry_backoff_seconds = float(
            os.environ.get("RETRY_BACKOFF_SECONDS", "1.0")
        )
        
        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")
        
        # Feature flags
        disable_tools_env = os.environ.get("DISABLE_TOOLS", "false").lower().strip()
        self.disable_tools = disable_tools_env in ("1", "true", "yes", "on")

        # Reasoning settings for GPT-5 Azure models
        reasoning_default_env = os.environ.get("REASONING_DEFAULT_ENABLED", "false").lower().strip()
        self.reasoning_default_enabled = reasoning_default_env in ("1", "true", "yes", "on")
        self.reasoning_effort = os.environ.get("REASONING_EFFORT")
        self.reasoning_verbosity = os.environ.get("REASONING_VERBOSITY")
        prefixes_raw = os.environ.get("REASONING_MODEL_PREFIXES", "gpt-5")
        self.reasoning_model_prefixes = tuple(
            prefix.strip() for prefix in prefixes_raw.split(",") if prefix.strip()
        ) or ("gpt-5",)
        chat_prefixes_raw = os.environ.get("REASONING_CHAT_PREFIXES", "")
        self.reasoning_chat_prefixes = tuple(
            prefix.strip() for prefix in chat_prefixes_raw.split(",") if prefix.strip()
        )
        self.reasoning_chat_max_output = int(
            os.environ.get("REASONING_CHAT_MAX_OUTPUT_TOKENS", "16384")
        )
        self.provider_max_output_tokens = int(
            os.environ.get("PROVIDER_MAX_OUTPUT_TOKENS", "0")
        )
        self.tool_groups_disabled_global = normalize_group_names(
            self._split_env_list(os.environ.get("TOOL_GROUPS_DISABLED"))
        )
        self.tool_groups_disabled_by_provider: Dict[str, List[str]] = {}
        prefix = "TOOL_GROUPS_DISABLED_"
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix):
                key = env_key[len(prefix) :].lower()
                disabled = normalize_group_names(self._split_env_list(env_val))
                if disabled:
                    self.tool_groups_disabled_by_provider[key] = disabled

        buffer_prefixes_raw = os.environ.get(
            "BUFFER_STREAM_MODEL_PREFIXES", "glm-"
        )
        self.buffer_stream_model_prefixes = tuple(
            prefix.strip().lower()
            for prefix in buffer_prefixes_raw.split(",")
            if prefix.strip()
        )
        self.buffer_stream_rate_limit_ms = int(
            os.environ.get("BUFFER_STREAM_RATE_LIMIT_MS", "400")
        )
        self.buffer_stream_cache_ttl_seconds = int(
            os.environ.get("BUFFER_STREAM_CACHE_TTL_SECONDS", "60")
        )
        
    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True

    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True
            
        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key
    
    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}
        
        # Get all environment variables
        env_vars = dict(os.environ)
        
        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith('CUSTOM_HEADER_'):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix
                
                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace('_', '-')
                    custom_headers[header_name] = env_value
        
        return custom_headers

    def should_buffer_stream_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        if not self.buffer_stream_model_prefixes:
            return False
        normalized = model_name.lower()
        return any(
            normalized.startswith(prefix)
            for prefix in self.buffer_stream_model_prefixes
        )

    @property
    def buffer_stream_rate_limit_seconds(self) -> float:
        if self.buffer_stream_rate_limit_ms <= 0:
            return 0.0
        return max(self.buffer_stream_rate_limit_ms / 1000.0, 0.0)

    @staticmethod
    def _split_env_list(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        return [part.strip() for part in raw.split(",")]

    def resolve_provider_key(self, base_url: Optional[str] = None) -> Optional[str]:
        url = (base_url or self.openai_base_url or "").strip()
        if not url:
            return None

        lowered = url.lower()
        if "dashscope.aliyuncs.com" in lowered:
            return "dashscope"
        if "azure.com" in lowered or "openai.azure" in lowered:
            return "azure"
        if "api.openai.com" in lowered:
            return "openai"

        parsed = urlparse(lowered if "://" in lowered else f"https://{lowered}")
        hostname = parsed.hostname or ""
        if hostname:
            return hostname.split(".")[0]
        return None

    def get_disabled_tool_groups(self, provider_key: Optional[str]) -> List[str]:
        disabled: Set[str] = set(self.tool_groups_disabled_global)
        if provider_key:
            disabled.update(
                self.tool_groups_disabled_by_provider.get(provider_key.lower(), [])
            )
        return list(disabled)

try:
    config = Config()
    print(f" Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
