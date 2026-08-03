from copy import deepcopy

import yaml

from deepm._paths import BACKTEST_SETTINGS_DIR, TRAIN_SETTINGS_DIR, SWEEP_SETTINGS_DIR

CORRELATION_SPAN_DEFAULT = 252

ARCHITECTURES = [
    "LSTM",
    "LSTM_SIMPLE",
    "MOM_TRANS",
    "MT_DEEPM",
    "AdvancedTemporalBaseline",
    "DeePM",
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge config dictionaries.

    Lists and scalar values are replaced by the child config. Nested dicts are
    merged so inherited configs can override a small subset of settings without
    copying the full universe/ticker metadata.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_with_inheritance(
    directory,
    file_name: str,
    *,
    seen: tuple[str, ...] = (),
) -> dict:
    """Load YAML config and apply optional ``inherits`` parent config."""
    if file_name in seen:
        chain = " -> ".join((*seen, file_name))
        raise ValueError(f"Cyclic config inheritance detected: {chain}")

    with open(directory / f"{file_name}.yaml", "r", encoding="UTF-8") as f:
        configs = yaml.safe_load(f) or {}

    parent = configs.pop("inherits", None)
    if parent is None:
        return configs

    base = _load_yaml_with_inheritance(directory, parent, seen=(*seen, file_name))
    return _deep_merge(base, configs)


def load_train_settings(file_name: str) -> dict:
    """Load the train settings from YAML."""
    configs = _load_yaml_with_inheritance(TRAIN_SETTINGS_DIR, file_name)
    configs["description"] = file_name
    return configs


def load_backtest_settings(file_name: str) -> dict:
    """Load backtest settings from YAML with optional inheritance support."""
    return _load_yaml_with_inheritance(BACKTEST_SETTINGS_DIR, file_name)


def load_settings_for_architecture(file_name: str, architecture: str) -> dict:
    """Load settings and apply architecture-specific defaults."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Must be one of {ARCHITECTURES}")
    settings = load_train_settings(file_name)

    run_name = architecture
    if settings["test_run"]:
        run_name = f"TEST/{run_name}"
    settings["run_name"] = run_name

    settings["use_contexts"] = False
    settings["cross_section"] = architecture in ["DeePM", "MT_DEEPM"]
    settings["num_context"] = 0

    settings.setdefault("correlation_span", CORRELATION_SPAN_DEFAULT)
    settings.setdefault("local_time_embedding", False)
    settings.setdefault("train_target_override", None)
    settings.setdefault("valid_target_override", None)
    settings.setdefault("extra_data_pre_steps", 0)
    settings.setdefault("tcost_inputs", False)

    return settings


def load_sweep_settings(file_name: str) -> dict:
    """Load the sweep settings from YAML."""
    with open(
        SWEEP_SETTINGS_DIR / f"{file_name}.yaml",
        "r",
        encoding="UTF-8",
    ) as f:
        configs = yaml.safe_load(f)
    return configs
