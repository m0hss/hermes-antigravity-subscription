"""Model catalog mapping and reasoning effort resolution for Antigravity."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Efforts that `agy --effort` accepts, per base model. A model that agy lists
# as `<base>-low|medium|high` takes --effort; a model it lists by bare name
# does not (agy 1.2.x rejects --effort for those, and rejects a missing
# --effort for the others).
#
# `agy models` is the source of truth. This table only lets known models
# resolve without spawning a process, and covers the case where `agy models`
# fails. An empty tuple means "bare name, no --effort".
_KNOWN_EFFORTS: dict[str, tuple[str, ...]] = {
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("low", "medium", "high"),
    "gemini-3.1-pro": ("low", "high"),
    "claude-opus-5-5": ("low", "medium", "high"),
    "claude-sonnet-5-5": ("low", "medium", "high"),
    "gpt-oss-120b": ("medium",),
}

_FALLBACK_MODELS = list(_KNOWN_EFFORTS)

_MODEL_ALIASES = {
    "default": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "gemini-flash": "gemini-3.8-flash",
    "gemini-3.8": "gemini-3.8-flash",
    "pro": "gemini-3.1-pro",
    "gemini-pro": "gemini-3.1-pro",
    "gemini-3.1": "gemini-3.1-pro",
    "sonnet": "claude-sonnet-5-5",
    "claude-sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "claude-opus": "claude-opus-5-5",
}

_EFFORT_SUFFIXES = (("-high", "high"), ("-medium", "medium"), ("-low", "low"))
_EFFORT_RANK = {"low": 0, "medium": 1, "high": 2}

# Substrings that mark a line of `agy models` as a model id (the command also
# prints a "Fetching available models..." header).
_MODEL_ID_MARKERS = ("gemini", "claude", "gpt", "model")

_CATALOG_TTL_SECONDS = 3600.0
_CATALOG_FAILURE_TTL_SECONDS = 60.0
_catalog_lock = threading.Lock()
_catalog_cache: tuple[float, dict[str, tuple[str, ...]]] | None = None


def _normalize_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    e = str(effort).strip().lower()
    if e in ("none", "off", "minimal"):
        return "low"
    if e in ("xhigh", "max", "ultra"):
        return "high"
    if e in ("low", "medium", "high"):
        return e
    return "medium"


def split_effort_suffix(model: str) -> tuple[str, str | None]:
    """Split `gemini-3.8-flash-high` into (`gemini-3.8-flash`, `high`)."""
    for suffix, effort in _EFFORT_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)], effort
    return model, None


def parse_catalog(text: str) -> dict[str, tuple[str, ...]]:
    """Group `agy models` output as {base model: supported efforts}.

    `gemini-3.8-flash-{low,medium,high}` becomes one `gemini-3.8-flash` entry
    with three efforts; an id with no effort suffix maps to an empty tuple.
    Insertion order follows the order agy printed the models in.
    """
    grouped: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "fetching" in line.lower():
            continue
        model_id = line.split()[0]
        if not any(marker in model_id.lower() for marker in _MODEL_ID_MARKERS):
            continue
        base, effort = split_effort_suffix(model_id)
        efforts = grouped.setdefault(base, [])
        if effort and effort not in efforts:
            efforts.append(effort)
    return {
        base: tuple(sorted(efforts, key=_EFFORT_RANK.__getitem__))
        for base, efforts in grouped.items()
    }


def load_catalog(
    timeout: float = 15.0, *, only_if_stale: bool = False
) -> dict[str, tuple[str, ...]]:
    """Run `agy models`, cache the parsed result, and return it.

    Returns an empty dict when agy is missing, slow, or prints nothing usable.
    Failures are cached briefly so an absent agy does not cost a process
    spawn per request. With `only_if_stale`, a thread that waited on the lock
    reuses the result the previous holder just cached instead of spawning agy
    again.
    """
    global _catalog_cache
    try:
        from .process import resolve_agy_command
    except ImportError:
        from process import resolve_agy_command

    with _catalog_lock:
        if only_if_stale:
            fresh = _cached_catalog()
            if fresh is not None:
                return fresh
        try:
            res = subprocess.run(
                [resolve_agy_command(), "models"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            catalog = parse_catalog(res.stdout)
        except Exception as exc:
            logger.debug("Antigravity model catalog unavailable: %s", exc)
            catalog = {}
        ttl = _CATALOG_TTL_SECONDS if catalog else _CATALOG_FAILURE_TTL_SECONDS
        _catalog_cache = (time.monotonic() + ttl, catalog)
        return catalog


def _cached_catalog() -> dict[str, tuple[str, ...]] | None:
    cache = _catalog_cache
    if cache is not None and time.monotonic() < cache[0]:
        return cache[1]
    return None


def model_efforts(base_model: str, *, refresh: bool = True) -> tuple[str, ...] | None:
    """Efforts agy accepts for `base_model`; () means bare name, None means unknown.

    Order: a fresh `agy models` result, then the built-in table, then (when
    `refresh` is true) a new `agy models` call. While a cached result is
    fresh, a model it does not list is not looked up again until the cache
    expires, so an unknown name costs no process spawn per request.
    """
    cached = _cached_catalog()
    if cached is not None and base_model in cached:
        return cached[base_model]
    if base_model in _KNOWN_EFFORTS:
        return _KNOWN_EFFORTS[base_model]
    if refresh and cached is None:
        return load_catalog(only_if_stale=True).get(base_model)
    return None


def _heuristic_efforts(base_model: str, has_suffix: bool) -> tuple[str, ...]:
    """Efforts for a model that neither agy nor the built-in table lists."""
    if base_model.startswith("gemini-") and "pro" in base_model:
        return ("low", "high")
    if base_model.startswith("gemini-") and "flash" in base_model:
        return ("low", "medium", "high")
    # A suffixed id is an effort variant; a bare unknown id takes no --effort.
    return ("low", "medium", "high") if has_suffix else ()


def _nearest_effort(requested: str, supported: tuple[str, ...]) -> str:
    """Closest supported effort; on a tie the stronger one wins."""
    target = _EFFORT_RANK[requested]
    return min(
        supported,
        key=lambda e: (abs(_EFFORT_RANK[e] - target), -_EFFORT_RANK[e]),
    )


def resolve_model_and_effort(
    model: str | None,
    reasoning_effort: str | None = None,
) -> tuple[str, str | None]:
    """Map user/hermes model request to concrete CLI model ID and effort level.

    Returns (model id, effort). The effort is None for models that agy selects
    by bare name; the client then omits --effort.
    """
    m = str(model or "gemini-3.8-flash").strip()
    m = _MODEL_ALIASES.get(m.lower(), m)

    base_model, suffix_effort = split_effort_suffix(m)
    supported = model_efforts(base_model)
    if supported is None:
        supported = _heuristic_efforts(base_model, suffix_effort is not None)
    if not supported:
        return base_model, None

    # Resolve effort: explicit argument > model suffix > config setting > default
    effort = _normalize_effort(reasoning_effort)
    if not effort and suffix_effort:
        effort = suffix_effort
    if not effort:
        try:
            from hermes_cli.config import load_config_readonly
            cfg_effort = load_config_readonly().get("agent", {}).get("reasoning_effort")
            effort = _normalize_effort(cfg_effort)
        except Exception:
            pass
    if not effort:
        effort = "medium"

    concrete_effort = _nearest_effort(effort, supported)
    return f"{base_model}-{concrete_effort}", concrete_effort
