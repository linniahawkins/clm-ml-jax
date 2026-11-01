"""
Load YAML into frozen dataclasses, with errors that name the file and key, and write
complete, commented YAML files from them.

The experiment files (``run.yaml``, ``domain.yaml``, ``model.yaml``) are parsed by
:func:`from_dict`, which walks a dataclass's type annotations and checks each value.
Unknown keys and invalid choices are errors, not silent defaults, so a typo in a config
file can never be ignored. With ``strict=True`` every key must be present (experiment
folders, decision 91): missing keys are listed together.

Each setting is declared with :func:`option`, which records what it means (written into
the generated files by :func:`render`), the Fortran name it replaces, the kind of run it
applies to (``applies``: ``"towers"`` or ``"grid"``), whether it is hidden from users
(test-only settings) and whether it is a file path (rewritten when a folder is copied).
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


def option(
    default: Any = dataclasses.MISSING,
    doc: str = "",
    fortran: str = "",
    *,
    applies: str | None = None,
    hidden: bool = False,
    path: bool = False,
):
    """A setting of an experiment file: its default (none: required), what it means
    (``doc``, written into generated files), the Fortran variable it replaces, the kind of
    run it applies to (``"towers"``/``"grid"``; None: every run), ``hidden`` (test-only: not
    written, may be left out) and ``path`` (a file or directory, relative to the folder)."""
    meta = {"doc": doc, "fortran": fortran, "applies": applies, "hidden": hidden, "path": path}
    return dataclasses.field(default=default, metadata=meta)


def _applies(field: dataclasses.Field, context: str | None) -> bool:
    want = field.metadata.get("applies")
    return want is None or context is None or want == context


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


class MissingKeys(ConfigError):
    """Keys missing from a strict (complete) file; ``keys`` lists them all."""

    def __init__(self, source: str, keys: list[str]):
        self.source, self.keys = source, keys
        super().__init__(f"{source}: missing key(s) {', '.join(keys)}")


def from_dict(
    cls: type[T],
    data: Any,
    *,
    source: str,
    key: str = "",
    strict: bool = False,
    context: str | None = None,
) -> T:
    """Build dataclass ``cls`` from ``data``, checking every field against its annotation.

    ``source`` (usually a file name) and ``key`` (the dotted path so far) are used only to
    make error messages point at the offending entry. ``strict``: every key that applies
    (``context``: the kind of run, ``"towers"`` or ``"grid"``) must be present, except hidden
    ones; keys for the other kind of run are errors. A strict load raises
    :class:`MissingKeys` naming every missing key.
    """
    missing: list[str] = []
    obj = _from_dict(cls, data, source=source, key=key, strict=strict, context=context,
                     missing=missing)  # fmt: skip
    if missing:
        raise MissingKeys(source, missing)
    return obj


def _from_dict(cls, data, *, source, key, strict, context, missing):
    where = f"{source}: {key}" if key else source
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    shown = [n for n, f in fields.items() if not f.metadata.get("hidden")]
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(shown)}")

    values = {}
    for name, field in fields.items():
        sub_key = f"{key}.{name}" if key else name
        applies = _applies(field, context)
        if name in data:
            if strict and not applies:
                raise ConfigError(
                    f"{source}: {sub_key}: only for {field.metadata['applies']} runs "
                    f"(this is a {context} run); remove it"
                )
            values[name] = _convert(hints[name], data[name], source=source, key=sub_key,
                                    strict=strict, context=context, missing=missing)  # fmt: skip
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            if not applies:
                values[name] = None
                continue
            missing.append(sub_key)
            raise (
                MissingKeys(source, missing)
                if strict
                else ConfigError(f"{source}: {sub_key}: required key is missing")
            )
        elif strict and applies and not field.metadata.get("hidden"):
            missing.append(sub_key)
    return cls(**values)


def _convert(annotation: Any, value: Any, *, source: str, key: str, strict: bool = False,
             context: str | None = None, missing: list | None = None) -> Any:  # fmt: skip
    sub = dict(strict=strict, context=context, missing=missing if missing is not None else [])
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    where = f"{source}: {key}"

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in (a for a in args if a is not type(None)):
            try:
                return _convert(option, value, source=source, key=key, **sub)
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
                _convert(args[0], v, source=source, key=f"{key}[{i}]", **sub)
                for i, v in enumerate(items)
            )
        if len(items) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} values, got {len(items)}")
        return tuple(
            _convert(a, v, source=source, key=f"{key}[{i}]", **sub)
            for i, (a, v) in enumerate(zip(args, items))
        )

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping of names to entries")
        return {
            str(k): _convert(args[1], v, source=source, key=f"{key}.{k}", **sub)
            for k, v in value.items()
        }

    if dataclasses.is_dataclass(annotation):
        return _from_dict(annotation, value, source=source, key=key, **sub)

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
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------------------
# Complete, commented files
# ---------------------------------------------------------------------------------------


def render(obj: Any, header: str, *, context: str | None = None) -> str:
    """A complete YAML file of dataclass ``obj``: ``header`` as comment lines, then every
    setting that applies to ``context`` (hidden ones left out), each with its explanation,
    its choices, the Fortran variable it replaces and its default (or REQUIRED) as comments.
    Loading the text (:func:`from_dict`, strict) gives ``obj`` back."""
    import textwrap

    lines = []
    for para in header.splitlines():
        lines += [f"# {w}".rstrip() for w in (textwrap.wrap(para, 94) or [""])]
    lines += _render_fields(obj, context, "")
    return "\n".join(lines) + "\n"


def _comment(text: str, indent: str) -> list[str]:
    import textwrap

    return [f"{indent}# {line}" for line in textwrap.wrap(" ".join(text.split()), 96 - len(indent))]


def flow(value: Any) -> str:
    """One value as YAML on one line."""
    if isinstance(value, tuple):
        value = list(value)
    return (
        yaml.safe_dump(value, default_flow_style=True, width=10**6)
        .strip()
        .removesuffix("...")
        .strip()
    )


def _render_fields(obj: Any, context: str | None, indent: str) -> list[str]:
    hints = typing.get_type_hints(type(obj))
    lines: list[str] = []
    top = indent == ""
    for f in dataclasses.fields(obj):
        if f.metadata.get("hidden") or not _applies(f, context):
            continue
        value = getattr(obj, f.name)
        doc = f.metadata.get("doc", "")
        section = (
            dataclasses.is_dataclass(value)
            or (
                isinstance(value, dict)
                and value
                and dataclasses.is_dataclass(next(iter(value.values())))
            )
            or (isinstance(value, tuple) and value and dataclasses.is_dataclass(value[0]))
        )
        lines.append("")
        if section:
            if doc:
                lines += _comment(doc, indent)
            lines.append(f"{indent}{f.name}:")
            if dataclasses.is_dataclass(value):
                cls_doc = type(value).__doc__ if type(value).__doc__ and top and not doc else ""
                sub = _render_fields(value, context, indent + "  ")
                if cls_doc and not cls_doc.startswith(type(value).__name__ + "("):
                    lines += _comment(cls_doc, indent + "  ")
                else:
                    sub = _strip_blank(sub)
                lines += sub
            elif isinstance(value, dict):
                for name, item in value.items():
                    lines.append(f"{indent}  {name}:")
                    lines += _strip_blank(_render_fields(item, context, indent + "    "))
            else:
                for k, item in enumerate(value):
                    item_lines = _strip_blank(_render_fields(item, context, indent + "  "))
                    first = next(i for i, ln in enumerate(item_lines)
                                 if ln.strip() and not ln.strip().startswith("#"))  # fmt: skip
                    item_lines = (
                        [f"{indent}{ln.strip()}" for ln in item_lines[:first]]
                        + [f"{indent}- {item_lines[first].strip()}"]
                        + item_lines[first + 1 :]
                    )
                    lines += ([""] if k else []) + item_lines
            continue
        lines += _comment(doc, indent) if doc else []
        notes = []
        choices = _choices(hints[f.name])
        if choices:
            notes.append("choices: " + " | ".join(map(str, choices)))
        if f.metadata.get("fortran"):
            notes.append("Fortran: " + f.metadata["fortran"])
        if f.default is dataclasses.MISSING:
            notes.append("REQUIRED")
        elif not dataclasses.is_dataclass(f.default):
            notes.append(f"default: {flow(f.default)}")
        if notes:
            lines += _comment("; ".join(notes) + ".", indent)
        lines.append(f"{indent}{f.name}: {flow(to_dict(value))}")
    return lines


def _strip_blank(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    return lines


def _choices(hint: Any) -> tuple:
    """The allowed values of a ``Literal[...]`` (or ``Optional[Literal]``) annotation."""
    if typing.get_origin(hint) is Literal:
        return typing.get_args(hint)
    if typing.get_origin(hint) in (Union, types.UnionType):
        for a in typing.get_args(hint):
            if typing.get_origin(a) is Literal:
                return typing.get_args(a)
    return ()


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
