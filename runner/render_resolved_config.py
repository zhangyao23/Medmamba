from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_override(raw_item: str) -> tuple[str, Any]:
    if "=" not in raw_item:
        raise ValueError(f"Override must use key=value form: {raw_item}")
    key, raw_value = raw_item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key cannot be empty: {raw_item}")
    return key, yaml.safe_load(raw_value)


def set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        elif not isinstance(child, dict):
            raise TypeError(f"Cannot set nested key under non-mapping path: {dotted_key}")
        current = child
    current[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a derived YAML config with dotted-path overrides.")
    parser.add_argument("--base-config", required=True, help="Path to the base YAML config.")
    parser.add_argument("--output", required=True, help="Path to the rendered YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in dotted.path=value form. May be provided multiple times.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    output_path = Path(args.output).resolve()

    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Base config must be a mapping: {base_path}")

    for raw_item in args.overrides:
        key, value = parse_override(raw_item)
        set_nested(config, key, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    print(str(output_path))


if __name__ == "__main__":
    main()
