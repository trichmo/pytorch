import sys
import warnings
from typing import Iterator


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
        self._value = None
        assert val is not None
        return val

    @property
    def value(self) -> object:
        if self._value is None:
            raise RuntimeError("value has been moved")
        return self._value


def movable_iter(iterable: object) -> Iterator[Movable]:
    it = iter(iterable)  # type: ignore[call-overload]
    while True:
        try:
            yield Movable(next(it))
        except StopIteration:
            return
