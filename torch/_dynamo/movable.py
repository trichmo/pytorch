import functools
import sys
import warnings
from collections.abc import Iterator

import torch.compiler


def _baseline_refcount() -> int:
    rc = [0]

    class _Probe:
        def __init__(self, value: object) -> None:
            rc[0] = sys.getrefcount(value)

    _Probe(object())
    return rc[0]


cached_baseline_refcount = _baseline_refcount()


class Movable:
    """A box that holds a Tensor and supports move semantics via .take()."""

    def __init__(self, value: object) -> None:
        if sys.getrefcount(value) > cached_baseline_refcount:
            warnings.warn(
                "Movable wraps a value that has external references. "
                "Without exclusive ownership, the value may persist through "
                "the entire compile region. "
                "Pass a temporary expression (e.g. Movable(torch.randn(4))) "
                "or use movable_iter() to wrap an iterator's output.",
                stacklevel=2,
            )
        self._value: object | None = value

    def take(self) -> object:
        val = self._value
        if val is None:
            raise RuntimeError(
                "Movable has already been taken. "
                "Create a fresh Movable for each call to the compiled function."
            )
        self._value = None
        return val

    @property
    def value(self) -> object:
        if self._value is None:
            raise RuntimeError("value has been moved")
        return self._value


def _is_compiled(fn: object) -> bool:
    from torch._dynamo.eval_frame import _TorchDynamoContext, OptimizedModule

    if isinstance(fn, (OptimizedModule, _TorchDynamoContext)):
        return True
    # torch.compile() on a plain function returns a wrapper function that carries
    # _torchdynamo_orig_callable; calling it drives Dynamo. The wrapper_id check
    # mirrors eval_frame.innermost_fn so functools.wraps copies of the marker
    # onto unrelated callables are not misdetected as compiled.
    return getattr(fn, "_torchdynamo_orig_callable", None) is not None and getattr(
        fn, "_torchdynamo_wrapper_id", None
    ) == id(fn)


def movable_call(fn):
    """Decorator that unwraps Movable arguments for eager/compiled compatibility.

    When the inner fn is compiled, or when Dynamo is tracing this wrapper,
    passes Movable through so Dynamo handles ownership via MovableSource.
    In pure eager mode, calls .take() on each Movable arg to extract the value.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _is_compiled(fn) or torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        args = tuple(a.take() if isinstance(a, Movable) else a for a in args)
        kwargs = {
            k: v.take() if isinstance(v, Movable) else v for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper


def movable_iter(iterable: object) -> Iterator[Movable]:
    it = iter(iterable)  # type: ignore[call-overload]
    while True:
        try:
            yield Movable(next(it))
        except StopIteration:
            return
