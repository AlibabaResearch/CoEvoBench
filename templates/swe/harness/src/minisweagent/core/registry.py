"""Name -> component object.

## The one rule: nothing here may run at import time

The platform refuses to seal a snapshot whose `minisweagent.agents.default` cannot be
imported. If resolving components happened while that module loaded, a single typo in a
single component file would make the whole snapshot unscorable — the generation's work
would be thrown away for a syntax error in an experiment you abandoned. So:

  * components are imported lazily, on first use, one at a time;
  * a component that fails to import is logged and skipped, never re-raised;
  * a slot whose configured component is unavailable falls back to the gen0 default;
  * if even the gen0 default is unavailable, `resolve` returns None and the caller turns
    that into a clearly-labelled *infrastructure* failure at run time — never a silent
    zero that would read as "the model could not do it".

There is deliberately no scan of the components directory (`pkgutil.walk_packages` and
friends import everything they find, which is the failure above wearing a hat), and no
on-disk cache of any kind — a sealed snapshot is hash-checked file by file, so writing a
cache next to the code would mark the snapshot as tampered with.
"""

from __future__ import annotations

import importlib

from minisweagent.utils.log import logger


class ComponentUnavailable(Exception):
    """A slot could not be filled, not even by its gen0 default. Infrastructure failure."""


#: Every resolution attempt that did not use the requested name, for the trajectory.
#: Purely diagnostic; reset per agent run via `reset_failures()`.
_failures: list[dict] = []


def reset_failures() -> None:
    _failures.clear()


def failures() -> list[dict]:
    return list(_failures)


def _import_component(name: str):
    """Import one component by registered name. Returns its module, or None on any failure."""
    from minisweagent.components import COMPONENTS

    target = COMPONENTS.get(name)
    if target is None:
        logger.warning("registry: no component registered under %r", name)
        _failures.append({"name": name, "reason": "unregistered"})
        return None
    try:
        return importlib.import_module(target)
    except Exception as exc:  # noqa: BLE001 — a broken component must not break the agent
        logger.warning("registry: component %r (%s) failed to import: %s", name, target, exc)
        _failures.append({"name": name, "reason": f"{type(exc).__name__}: {exc}"})
        return None


def _build(name: str, **kwargs):
    """Import `name` and call its `build(**kwargs)`. Returns the component, or None."""
    module = _import_component(name)
    if module is None:
        return None
    factory = getattr(module, "build", None)
    if factory is None:
        logger.warning("registry: component %r has no build() factory", name)
        _failures.append({"name": name, "reason": "no build()"})
        return None
    try:
        return factory(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry: component %r build() raised: %s", name, exc)
        _failures.append({"name": name, "reason": f"build(): {type(exc).__name__}: {exc}"})
        return None


def resolve(slot: str, name: str, **kwargs) -> tuple[object | None, str]:
    """Fill one slot. Returns (component, name_actually_used).

    Falls back to the slot's gen0 default if `name` is unavailable. Returns (None, name)
    only when the default is unavailable too — the caller must treat that as infrastructure
    failure, not as a weak episode.
    """
    from minisweagent.core.pipeline import DEFAULT_PIPELINE

    component = _build(name, **kwargs)
    if component is not None:
        return component, name

    fallback = DEFAULT_PIPELINE.get(slot)
    if isinstance(fallback, str) and fallback != name:
        logger.warning("registry: slot %r falling back to gen0 default %r", slot, fallback)
        component = _build(fallback, **kwargs)
        if component is not None:
            return component, fallback

    return None, name


def resolve_hooks(names, **kwargs) -> list:
    """Build the hook chain. Unavailable hooks are skipped (they are observers, not logic)."""
    built = []
    for name in names or []:
        hook = _build(name, **kwargs)
        if hook is None:
            continue
        if not getattr(hook, "name", ""):
            try:
                hook.name = name
            except Exception:  # noqa: BLE001 — a hook that rejects a name is still usable
                pass
        built.append(hook)
    return built
