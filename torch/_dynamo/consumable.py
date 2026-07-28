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


def _box_for_resume_forwarding(value: object) -> "Consumable":
    """Box a value for Dynamo-generated resume-forwarding codegen.

    Skips the external-reference warning: the caller's own local is loaded to
    pass into this call and then immediately deleted by the very next
    instruction (see OutputGraph.compile_subgraph), so the refcount heuristic
    in Consumable.__init__ would spuriously fire on every occurrence of this
    pattern. Not for user code -- use Consumable(value) directly instead.
    """
    return Consumable(value, _skip_refcount_check=True)


def _unwrap_for_resume(value: object) -> object:
    """Unwrap a value boxed by _box_for_resume_forwarding, in a resume function's prologue.

    Dynamo does not guarantee it retraces every resume function -- some run as
    plain, uncompiled Python. When Dynamo *does* retrace this frame, the
    generic frame-entry unwrap (unwrap_consumable_for_tracing) has already
    replaced this argument's symbolic value with the unboxed one before any
    bytecode (including this call) is stepped through, so `value` is no
    longer a Consumable and this is a no-op. When Dynamo does not retrace the
    frame, `value` is still the raw box at runtime, so it must be taken here
    or every subsequent real operation on it would fail.
    """
    if isinstance(value, Consumable):
        return value.take()
    return value


class Consumable:
    """A box that holds a Tensor and supports move semantics via .take()."""

    def __init__(self, value: object, *, _skip_refcount_check: bool = False) -> None:
        if not _skip_refcount_check and sys.getrefcount(value) > cached_baseline_refcount:
            warnings.warn(
                "Consumable wraps a value that has external references. "
                "Without exclusive ownership, the value may persist through "
                "the entire compile region. "
                "Pass a temporary expression (e.g. Consumable(torch.randn(4))) "
                "or use consumable_iter() to wrap an iterator's output.",
                stacklevel=2,
            )
        self._value: object | None = value

    def take(self) -> object:
        val = self._value
        if val is None:
            raise RuntimeError(
                "Consumable has already been taken. "
                "Create a fresh Consumable for each call to the compiled function."
            )
        self._value = None
        return val

    @property
    def value(self) -> object:
        if self._value is None:
            raise RuntimeError("value has been consumed")
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


def consumable_call(fn):
    """Decorator that unwraps Consumable arguments for eager/compiled compatibility.

    When the inner fn is compiled, or when Dynamo is tracing this wrapper,
    passes Consumable through so Dynamo handles ownership via ConsumableSource.
    In pure eager mode, calls .take() on each Consumable arg to extract the value.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _is_compiled(fn) or torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        args = tuple(a.take() if isinstance(a, Consumable) else a for a in args)
        kwargs = {
            k: v.take() if isinstance(v, Consumable) else v for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper


def consumable_iter(iterable: object) -> Iterator[Consumable]:
    it = iter(iterable)  # type: ignore[call-overload]
    while True:
        try:
            yield Consumable(next(it))
        except StopIteration:
            return
