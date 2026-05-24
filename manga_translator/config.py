from __future__ import annotations

import os
from pathlib import Path


ENV_FILE = ".env"
MIN_LLAMA_THREADS = 1
MAX_LLAMA_THREADS = 32
FALLBACK_LLAMA_THREADS = 4


def load_env_file(path: str | Path = ENV_FILE) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def get_config_value(key: str, default: str = "", env_path: str | Path = ENV_FILE) -> str:
    return os.environ.get(key) or load_env_file(env_path).get(key, default)


def get_int_config_value(key: str, default: int, env_path: str | Path = ENV_FILE) -> int:
    raw_value = get_config_value(key, "", env_path)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def clamp_llama_threads(value: int) -> int:
    return max(MIN_LLAMA_THREADS, min(MAX_LLAMA_THREADS, value))


def detected_cpu_count() -> int:
    return clamp_llama_threads(os.cpu_count() or FALLBACK_LLAMA_THREADS)


def default_llama_threads(env_path: str | Path = ENV_FILE) -> int:
    return clamp_llama_threads(get_int_config_value("LLAMA_THREADS", detected_cpu_count(), env_path))
