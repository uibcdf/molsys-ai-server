from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


def _default_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base).expanduser().resolve() / "molsys-ai"
    return Path.home() / ".config" / "molsys-ai"


DEFAULT_PUBLIC_BASE_URL = "https://api.uibcdf.org"

# This default is only used for u1_ keys (LAN-first). u2_ keys never try it.
DEFAULT_LAN_BASE_URL = "http://192.168.0.100:8001"


@dataclass(frozen=True)
class CLIConfig:
    public_base_url: str = DEFAULT_PUBLIC_BASE_URL
    lan_base_url: str = DEFAULT_LAN_BASE_URL
    api_key: str | None = None

    # For LAN-first, we want failures to be quick.
    connect_timeout_s: float = 0.6
    read_timeout_s: float = 1800.0

    @property
    def timeout(self) -> tuple[float, float]:
        return (float(self.connect_timeout_s), float(self.read_timeout_s))


def config_path() -> Path:
    return _default_config_dir() / "config.json"


def load_config() -> CLIConfig:
    path = config_path()
    if not path.exists():
        return CLIConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return CLIConfig()

    if not isinstance(data, dict):
        return CLIConfig()

    kwargs: Dict[str, Any] = {}
    for field in asdict(CLIConfig()).keys():
        if field in data:
            kwargs[field] = data[field]

    try:
        return CLIConfig(**kwargs)
    except TypeError:
        return CLIConfig()


def save_config(cfg: CLIConfig) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(cfg), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

