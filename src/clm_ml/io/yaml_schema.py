"""
Load YAML into frozen dataclasses, with errors that name the file and key.

Every config file (``run.yaml``, ``model.yaml``, ``params.yaml``, ``towers.yaml``) is
parsed by :func:`from_dict`, which walks a dataclass's type annotations and checks each
value. Unknown keys and invalid choices are errors, not silent defaults, so a typo in a
config file can never be ignored.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, Literal, TypeVar, Union

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Invalid configuration, e.g.
    ``model.yaml: numerics.integrator: 'rk5' is not one of [euler, ..., rk4]``."""


def read_yaml(path: Path) -> dict:
    """Read a YAML file that must contain a mapping (or be empty)."""
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of keys to values")
    return data


def from_dict(cls: type[T], data: Any, *, source: str, key: str = "") -> T:
    """Build dataclass ``cls`` from ``data``, checking every field against its annotation.

    ``source`` (usually a file name) and ``key`` (the dotted path so far) are used only to
    make error messages point at the offending entry.
    """
    where = f"{source}: {key}" if key else source
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(fields)}")

    values = {}
    for name, field in fields.items():
        sub_key = f"{key}.{name}" if key else name
        if name in data:
            values[name] = _convert(hints[name], data[name], source=source, key=sub_key)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{source}: {sub_key}: required key is missing")
    return cls(**values)


def _convert(annotation: Any, value: Any, *, source: str, key: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    where = f"{source}: {key}"

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in (a for a in args if a is not type(None)):
            try:
                return _convert(option, value, source=source, key=key)
            except ConfigError as exc:
                errors.append(str(exc).split(": ", 2)[-1])
        raise ConfigError(f"{where}: {value!r} is not valid ({'; or '.join(errors)})")

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of [{', '.join(map(str, args))}]")
        return value

    if origin is tuple:
        items = value if isinstance(value, (list, tuple)) else [value]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _convert(args[0], v, source=source, key=f"{key}[{i}]") for i, v in enumerate(items)
            )
        if len(items) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} values, got {len(items)}")
        return tuple(
            _convert(a, v, source=source, key=f"{key}[{i}]")
            for i, (a, v) in enumerate(zip(args, items))
        )

    if dataclasses.is_dataclass(annotation):
        return from_dict(annotation, value, source=source, key=key)

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        # YAML parses unquoted ISO dates/times into date objects; keep them as text.
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return value
    if annotation is Any:
        return value
    raise TypeError(f"unsupported annotation {annotation!r} for {where}")


def to_dict(obj: Any) -> Any:
    """Inverse of :func:`from_dict`: dataclasses to plain dicts/lists for YAML output."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return [to_dict(v) for v in obj]
    return obj


def apply_overrides(data: dict, overrides: list[str], *, source: str) -> dict:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML) to a nested dict, in place."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"{source}: override {item!r} must look like key.path=value")
        path, raw = item.split("=", 1)
        parts = path.strip().split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{source}: override {item!r}: {part} is not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return data
