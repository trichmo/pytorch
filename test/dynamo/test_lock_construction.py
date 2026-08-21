# Owner(s): ["module: dynamo"]

"""
Tests for constructing threading lock primitives during Dynamo tracing.

The pure-Python ``functools.lru_cache`` wrapper allocates an ``_thread.RLock``
(via ``RLock()``) when it is built. Dynamo must be able to construct these
C-level lock types so that building such a wrapper inside a compiled region does
not graph break. Because ``_thread.RLock``/``_thread.lock`` reject
``object.__new__`` ("not safe") but expose their own ``tp_new``, their
``__new__`` methods are whitelisted so the generic user-defined-object
construction path can allocate an example (and reconstruct one on escape).
"""

import _thread
import copy
import importlib
import sys
import threading

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _load_py_functools():
    """Return the pure-Python functools (``_functools`` C accelerator blocked)."""
    import functools  # noqa: F401  ensure it is importable at all

    saved_functools = sys.modules.pop("functools", None)
    saved_c = sys.modules.get("_functools", None)
    sys.modules["_functools"] = None  # force ImportError in `from _functools ...`
    try:
        return importlib.import_module("functools")
    finally:
        sys.modules.pop("functools", None)
        if saved_functools is not None:
            sys.modules["functools"] = saved_functools
        if saved_c is not None:
            sys.modules["_functools"] = saved_c
        else:
            sys.modules.pop("_functools", None)


class LockConstructionTests(torch._dynamo.test_case.TestCase):
    @parametrize(
        "kind",
        ["thread_rlock", "thread_lock", "threading_rlock", "threading_lock"],
    )
    def test_lock_construction_no_graph_break(self, kind):
        ctor = {
            "thread_rlock": _thread.RLock,
            "thread_lock": _thread.LockType,
            "threading_rlock": threading.RLock,
            "threading_lock": threading.Lock,
        }[kind]

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            lock = ctor()
            return x + 1, lock

        out, lock = fn(torch.ones(3))
        self.assertEqual(out, torch.ones(3) + 1)
        # The constructed lock escapes the graph and is reconstructed via its
        # own __new__, so it must be a usable, unlocked instance.
        self.assertTrue(hasattr(lock, "acquire"))
        with lock:
            pass

    def test_lru_cache_wrapper_copy_is_identity(self):
        py_functools = _load_py_functools()
        # Sanity: this is the pure-Python implementation, whose wrapper is a
        # plain function that allocates an RLock at construction time.
        self.assertFalse(hasattr(py_functools.lru_cache, "__wrapped__"))

        def orig(x, y):
            return 3 * x + y

        @torch.compile(backend="eager", fullgraph=True)
        def build_and_copy():
            part = py_functools.partial(orig, 2)
            wrapper = py_functools.lru_cache(2)(part)
            return copy.copy(wrapper) is wrapper

        self.assertTrue(build_and_copy())


instantiate_parametrized_tests(LockConstructionTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
