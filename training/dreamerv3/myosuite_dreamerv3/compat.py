# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Runtime compatibility helpers for DreamerV3 on modern JAX and NumPy."""

from __future__ import annotations

import functools
import importlib
from pathlib import Path
import sys
import types
from typing import Any


def import_myosuite() -> Any:
    """Import ``myosuite`` while avoiding PEP 420 namespace shadowing on ``sys.path``.

    Returns:
        Imported ``myosuite`` module.
    """
    import myosuite

    if getattr(myosuite, "__file__", None) is not None:
        return myosuite
    bad_parents = {str(Path(p).parent) for p in getattr(myosuite, "__path__", [])}
    saved_path = sys.path[:]
    try:
        sys.path = [p for p in sys.path if p not in bad_parents]
        for key in [
            k for k in sys.modules if k == "myosuite" or k.startswith("myosuite.")
        ]:
            del sys.modules[key]
        return importlib.import_module("myosuite")
    finally:
        sys.path = saved_path


@functools.cache
def apply_dreamer_jax_compat() -> None:
    """Apply JAX >= 0.8 shims so unmodified ``danijar/dreamerv3`` runs on modern JAX."""
    import jax

    orig_jit = jax.jit
    try:
        orig_jit(lambda x: x, None, None)
        needs_jit_shim = False
    except TypeError:
        needs_jit_shim = True

    if needs_jit_shim:
        pos_names = (
            "in_shardings",
            "out_shardings",
            "static_argnums",
            "static_argnames",
            "donate_argnums",
        )

        @functools.wraps(orig_jit)
        def _positional_compat_jit(fun: Any, *args: Any, **kwargs: Any) -> Any:
            if not args:
                return orig_jit(fun, **kwargs)
            mapped = dict(kwargs)
            for name, val in zip(pos_names, args):
                if val is not None or name in ("in_shardings", "out_shardings"):
                    mapped.setdefault(name, val)
            return orig_jit(fun, **mapped)

        jax.jit = _positional_compat_jit  # type: ignore[assignment]

    if hasattr(jax, "shard_map") and "jax.experimental.shard_map" not in sys.modules:
        try:
            import jax.experimental.shard_map  # noqa: F401
        except ModuleNotFoundError:
            mod = types.ModuleType("jax.experimental.shard_map")

            def _shard_map(
                f: Any,
                mesh: Any,
                in_specs: Any,
                out_specs: Any,
                check_rep: bool = False,
                **kw: Any,
            ) -> Any:
                return jax.shard_map(
                    f,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_vma=check_rep,
                    **kw,
                )

            mod.shard_map = _shard_map  # type: ignore[attr-defined]
            sys.modules["jax.experimental.shard_map"] = mod
            if hasattr(jax, "experimental"):
                jax.experimental.shard_map = mod  # type: ignore[attr-defined]

    src_cfg = getattr(getattr(jax, "_src", None), "config", None)
    if src_cfg is not None:
        if not hasattr(src_cfg, "explicit_device_get_scope"):
            src_cfg.explicit_device_get_scope = lambda: jax.transfer_guard("allow")
        if not hasattr(src_cfg, "explicit_device_put_scope"):
            src_cfg.explicit_device_put_scope = lambda: jax.transfer_guard("allow")

    try:
        import dreamerv3  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import 'dreamerv3'. Clone https://github.com/danijar/dreamerv3 "
            "and add it to PYTHONPATH (or install a wheel-enabled fork)."
        ) from exc
