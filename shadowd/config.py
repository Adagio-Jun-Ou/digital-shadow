#!/usr/bin/env python3
"""
Shadow AI Configuration System.

Loads configuration from environment variables and optional YAML config file.
Priority: Environment > Config file > Defaults

Usage:
    from shadowd.config import get_config
    config = get_config()
    print(config.llm.model)  # "glm-5"
"""

import copy
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("shadowd.config")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_MAX_PORT = 65535
_MIN_PORT = 1


@dataclass
class LLMConfig:
    model: str = "glm-5"
    api_base: str = "https://ollama.com/v1"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    max_retries: int = 3
    retry_base_delay: float = 1.0
    streaming_enabled: bool = True


@dataclass
class MemoryConfig:
    core_limit: int = 2000
    working_limit: int = 500
    archival_limit: int = 10000
    observation_limit: int = 1000
    insight_limit: int = 200
    decision_limit: int = 500
    save_interval_seconds: int = 300


@dataclass
class ShikigamiConfig:
    bt_scan_timeout: int = 10
    wifi_scan_timeout: int = 10
    cmd_timeout: int = 10
    allowed_commands: list = field(default_factory=lambda: ["bluetoothctl", "nmcli", "uptime", "echo"])
    llm_timeout: float = 60.0


@dataclass
class ObserverConfig:
    terminal_poll_seconds: float = 2.0
    clipboard_poll_seconds: float = 3.0
    window_poll_seconds: float = 5.0
    git_poll_seconds: float = 30.0
    watch_dirs: list = field(default_factory=list)
    proactive_interval_seconds: int = 300


@dataclass 
class SecurityConfig:
    auth_enabled: bool = False
    admin_email: str = "admin@shadow.local"
    admin_password: str = ""
    secret_key: str = ""
    cors_origins: str = "http://localhost:3000,http://localhost:8765,http://127.0.0.1:8765"
    rate_limit_per_minute: int = 60


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"


@dataclass
class ShadowConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    shikigami: ShikigamiConfig = field(default_factory=ShikigamiConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    data_dir: str = ""
    
    @classmethod
    def from_env(cls) -> "ShadowConfig":
        import os
        config = cls()
        config.llm.model = os.environ.get("OLLAMA_MODEL", config.llm.model)
        config.llm.api_base = os.environ.get("OLLAMA_API_BASE", config.llm.api_base)
        config.llm.api_key = os.environ.get("OLLAMA_API_KEY", config.llm.api_key)
        try:
            temp = float(os.environ.get("SHADOW_LLM_TEMPERATURE", config.llm.temperature))
            config.llm.temperature = max(0.0, min(2.0, temp))
        except (ValueError, TypeError):
            config.llm.temperature = 0.7
        try:
            tokens = int(os.environ.get("SHADOW_LLM_MAX_TOKENS", config.llm.max_tokens))
            config.llm.max_tokens = max(1, min(32768, tokens))
        except (ValueError, TypeError):
            config.llm.max_tokens = 2048
        try:
            retries = int(os.environ.get("SHADOW_LLM_MAX_RETRIES", config.llm.max_retries))
            config.llm.max_retries = max(0, min(10, retries))
        except (ValueError, TypeError):
            config.llm.max_retries = 3
        
        try:
            core = int(os.environ.get("SHADOW_MEMORY_CORE_LIMIT", config.memory.core_limit))
            config.memory.core_limit = max(1, min(100000, core))
        except (ValueError, TypeError):
            config.memory.core_limit = 2000
        try:
            obs = int(os.environ.get("SHADOW_OBSERVATION_LIMIT", config.memory.observation_limit))
            config.memory.observation_limit = max(1, min(100000, obs))
        except (ValueError, TypeError):
            config.memory.observation_limit = 1000
        try:
            interval = int(os.environ.get("SHADOW_SAVE_INTERVAL", config.memory.save_interval_seconds))
            config.memory.save_interval_seconds = max(10, min(86400, interval))
        except (ValueError, TypeError):
            config.memory.save_interval_seconds = 300
        
        config.security.auth_enabled = os.environ.get("SHADOW_AUTH_ENABLED", "").lower() in ("1", "true", "yes")
        config.security.admin_password = os.environ.get("SHADOW_ADMIN_PASSWORD", config.security.admin_password)
        config.security.secret_key = os.environ.get("SHADOW_SECRET_KEY", config.security.secret_key)
        config.security.cors_origins = os.environ.get("SHADOW_CORS_ORIGINS", config.security.cors_origins)
        # Sanitize cors_origins — strip control chars
        config.security.cors_origins = _SANITIZE_CONTENT_RE.sub('', config.security.cors_origins)
        
        if config.security.auth_enabled and not config.security.secret_key:
            logger.error("SHADOW_AUTH_ENABLED is true but SHADOW_SECRET_KEY is not set. Auth will not work properly.")
        
        config.server.host = os.environ.get("SHADOW_HOST", config.server.host)
        # Sanitize host — strip control chars and angle brackets
        config.server.host = _SANITIZE_STRICT_RE.sub('', config.server.host)
        try:
            port = int(os.environ.get("SHADOW_PORT", config.server.port))
            config.server.port = max(_MIN_PORT, min(_MAX_PORT, port))
        except (ValueError, TypeError):
            config.server.port = 8765
        config.server.log_level = os.environ.get("SHADOW_LOG_LEVEL", config.server.log_level).upper()
        if config.server.log_level not in _VALID_LOG_LEVELS:
            config.server.log_level = "INFO"
        config.data_dir = os.environ.get("SHADOW_DATA_DIR", "")
        
        try:
            proactive = int(os.environ.get("SHADOW_PROACTIVE_INTERVAL", config.observer.proactive_interval_seconds))
            config.observer.proactive_interval_seconds = max(10, min(86400, proactive))
        except (ValueError, TypeError):
            config.observer.proactive_interval_seconds = 300
        
        return config
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ShadowConfig":
        import os
        if not HAS_YAML:
            logger.warning("PyYAML not installed. Using default/config from env only.")
            return cls.from_env()
        
        if not path.exists():
            logger.info("Config file %s not found. Using defaults.", path)
            return cls.from_env()
        
        # Path traversal protection — resolve and verify path
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            logger.error("Invalid config path: %s", path)
            return cls.from_env()
        
        try:
            with open(resolved) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("Invalid YAML in config %s: %s", path, type(e).__name__)
            return cls.from_env()
        except (OSError, ValueError) as e:
            logger.error("Failed to read config %s: %s", path, type(e).__name__)
            return cls.from_env()
        
        if not isinstance(data, dict):
            logger.error("Config file %s did not produce a dict. Using defaults.", path)
            return cls.from_env()
        
        config = cls.from_env()
        
        def _expect_type(value, expected_type, field_name):
            if isinstance(value, expected_type):
                return value
            type_name = getattr(expected_type, '__name__', None) or str(expected_type)
            logger.warning("Config field '%s': expected %s, got %s; ignoring YAML override", field_name, type_name, type(value).__name__)
            return None

        if "llm" in data:
            llm = data["llm"]
            config.llm.model = llm.get("model", config.llm.model)
            config.llm.api_base = llm.get("api_base", config.llm.api_base)
            v = llm.get("temperature", config.llm.temperature)
            validated = _expect_type(v, (int, float), "llm.temperature")
            if validated is not None:
                config.llm.temperature = float(validated)
            v = llm.get("max_tokens", config.llm.max_tokens)
            validated = _expect_type(v, int, "llm.max_tokens")
            if validated is not None:
                config.llm.max_tokens = validated
        
        if "memory" in data:
            mem = data["memory"]
            v = mem.get("core_limit", config.memory.core_limit)
            validated = _expect_type(v, int, "memory.core_limit")
            if validated is not None:
                config.memory.core_limit = validated
            v = mem.get("observation_limit", config.memory.observation_limit)
            validated = _expect_type(v, int, "memory.observation_limit")
            if validated is not None:
                config.memory.observation_limit = validated
            v = mem.get("save_interval_seconds", config.memory.save_interval_seconds)
            validated = _expect_type(v, int, "memory.save_interval_seconds")
            if validated is not None:
                config.memory.save_interval_seconds = validated
        
        if "server" in data:
            srv = data["server"]
            config.server.host = srv.get("host", config.server.host)
            v = srv.get("port", config.server.port)
            validated = _expect_type(v, int, "server.port")
            if validated is not None:
                config.server.port = validated
            config.server.log_level = srv.get("log_level", config.server.log_level)
        
        if "observer" in data:
            obs = data["observer"]
            v = obs.get("proactive_interval_seconds", config.observer.proactive_interval_seconds)
            validated = _expect_type(v, int, "observer.proactive_interval_seconds")
            if validated is not None:
                config.observer.proactive_interval_seconds = max(10, min(86400, validated))
            watch_dirs = obs.get("watch_dirs", config.observer.watch_dirs)
            if isinstance(watch_dirs, list):
                config.observer.watch_dirs = [str(d) for d in watch_dirs]
        
        if "shikigami" in data:
            shiki = data["shikigami"]
            v = shiki.get("cmd_timeout", config.shikigami.cmd_timeout)
            validated = _expect_type(v, int, "shikigami.cmd_timeout")
            if validated is not None:
                config.shikigami.cmd_timeout = max(1, min(300, validated))
            v = shiki.get("llm_timeout", config.shikigami.llm_timeout)
            validated = _expect_type(v, (int, float), "shikigami.llm_timeout")
            if validated is not None:
                config.shikigami.llm_timeout = max(1.0, min(600.0, float(validated)))
            allowed = shiki.get("allowed_commands", config.shikigami.allowed_commands)
            if isinstance(allowed, list):
                config.shikigami.allowed_commands = [str(c) for c in allowed[:50]]
        
        logger.info("Loaded config from %s", path)
        return config


_SANITIZE_STRICT_RE = __import__('re').compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = __import__('re').compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

_config: Optional[ShadowConfig] = None
_config_lock = threading.Lock()


def get_config() -> ShadowConfig:
    global _config
    with _config_lock:
        if _config is None:
            import os
            config_path = Path(os.environ.get("SHADOW_CONFIG", "shadow.yaml"))
            _config = ShadowConfig.from_yaml(config_path)
        return copy.deepcopy(_config)