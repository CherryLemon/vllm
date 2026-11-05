# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional model-owned runtime extensions selected before resource allocation.

Native execution remains the default. General plugins may register a factory
for their own architecture names; factories must decline unsupported configs
before allocating state. Execution errors are never retried with a new runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import RLock
from typing import Any

_runtime_factories: dict[str, Callable[[Any], Any]] = {}
_registry_lock = RLock()


def register_model_runtime(architectures: Iterable[str], factory: Callable) -> None:
    """Publish a registration atomically; repeated same-owner calls are harmless."""
    global _runtime_factories
    names = tuple(architectures)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("runtime architectures must be nonempty names")
    if len(set(names)) != len(names) or not callable(factory):
        raise ValueError("runtime registration requires unique names and a factory")
    with _registry_lock:
        for name in names:
            current = _runtime_factories.get(name)
            if current is not None and current is not factory:
                raise RuntimeError(f"a runtime is already registered for {name}")
        updated = dict(_runtime_factories)
        updated.update(dict.fromkeys(names, factory))
        _runtime_factories = updated


def create_model_runtime(config):
    """Resolve one architecture owner without importing any optional plugin."""
    model = config.model_config
    architecture = getattr(model, "architecture", None)
    names = (architecture,) if architecture else ()
    hf = getattr(model, "hf_config", None)
    names += tuple(getattr(hf, "architectures", None) or ())
    snapshot = _runtime_factories
    factories = []
    for name in names:
        factory = snapshot.get(name)
        if factory is not None and all(factory is not current for current in factories):
            factories.append(factory)
    if len(factories) > 1:
        raise ValueError("model architecture names resolve to different runtimes")
    if not factories:
        return None
    return factories[0](config)
