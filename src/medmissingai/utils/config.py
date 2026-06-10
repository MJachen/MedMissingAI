from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_DATA_CONFIG = "configs/data_local.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train_local.yaml"
DEFAULT_MODEL_CONFIG = "configs/model.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        config = yaml.safe_load(text)
    except ImportError:
        config = _parse_simple_yaml(text)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return config


def load_experiment_config(
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Load either one legacy config or split data/train/model configs."""

    if config_path is not None:
        path = Path(config_path)
        return load_config(path), [path]

    data_path = Path(data_config_path or DEFAULT_DATA_CONFIG)
    train_path = Path(train_config_path or DEFAULT_TRAIN_CONFIG)
    model_path = Path(model_config_path or DEFAULT_MODEL_CONFIG)

    data_config = load_config(data_path)
    train_config = load_config(train_path)
    model_config = load_config(model_path)

    config: dict[str, Any] = {}
    config.update({k: v for k, v in train_config.items() if k not in {"data", "training", "model"}})
    config["data"] = _section(data_config, "data")
    config["training"] = _section(train_config, "training")
    config["model"] = _section(model_config, "model")
    return config, [data_path, train_path, model_path]


def get_save_dir(config: dict[str, Any]) -> Path:
    training = config["training"]
    save_dir = training.get("save_dir")
    if not save_dir:
        raise ValueError("Config must define training.save_dir")
    return Path(save_dir)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, config)
    if not isinstance(section, dict):
        raise ValueError(f"Config section `{name}` must be a mapping")
    return section


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by project configs.

    This fallback avoids making smoke runs depend on PyYAML being installed in
    an existing research environment. It supports top-level sections, one level
    of indented keys, inline lists, booleans, ints, floats, and strings.
    """

    config: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith(" "):
            key, value = _split_key_value(line)
            if value == "":
                current_section = {}
                config[key] = current_section
            else:
                config[key] = _parse_scalar(value)
                current_section = None
        else:
            if current_section is None:
                raise ValueError(f"Invalid simple YAML indentation near: {raw_line}")
            key, value = _split_key_value(line.strip())
            current_section[key] = _parse_scalar(value)

    return config


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Expected key/value line, got: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")
