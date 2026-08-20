# Owner(s): ["module: dynamo"]

import collections
import contextlib
import os
import re
import sys
import unittest
import warnings
import weakref

import torch
from torch._dynamo.consumable import (
    _is_compiled,
    Consumable,
    consumable_call,
    consumable_iter,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _mk(consumable_cls, value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return consumable_cls(value)


# Liveness probe for the eager consumable path. A custom op placed after the consumed
# input's last use records, at runtime, whether a weakref to that input is
# already dead -- i.e. whether the input was freed mid-forward.
_probe_lib = torch.library.Library("consumable_test_probe", "DEF")  # noqa: SCOPED_LIBRARY
_probe_state = {"wref": None, "alive_at_mark": None}


def _mark_impl(x):
    wref = _probe_state["wref"]
    _probe_state["alive_at_mark"] = wref is not None and wref() is not None
    return x.clone()


_probe_lib.define("mark(Tensor x) -> Tensor")
_probe_lib.impl("mark", _mark_impl, "CompositeExplicitAutograd")


_Pair = collections.namedtuple("_Pair", ["a", "b"])


@instantiate_parametrized_tests
class TestConsumable(TestCase):
    def test_basic(self):
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        t = torch.tensor([1, 2, 3, 4])
        expected = t + 1
        m = Consumable(torch.tensor([1, 2, 3, 4]))
        result = fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_consumable_can_del(self):
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x + 1
            del x
            return y

        m = Consumable(torch.tensor([1, 2, 3]))
        result = fn(m)
        self.assertEqual(result, torch.tensor([1, 2, 3]) + 1)
        self.assertIsNone(m._value)

    def test_multiple_args(self):
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = Consumable(t1)
            m2 = Consumable(t2)
        result = fn(m1, m2)
        self.assertEqual(result, expected)
        self.assertIsNone(m1._value)
        self.assertIsNone(m2._value)

    def test_mixed_consumable_and_tensor(self):
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = Consumable(t1)
        result = fn(m1, t2)
        self.assertEqual(result, expected)
        self.assertIsNone(m1._value)

    def test_second_call_with_fresh_consumable(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @consumable_call
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x * 2

        t1 = torch.randn(4)
        expected1 = t1 * 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = Consumable(t1)
        result1 = fn(m1)
        self.assertEqual(result1, expected1)
        self.assertEqual(cnt.frame_count, 1)

        t2 = torch.randn(4)
        expected2 = t2 * 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = Consumable(t2)
        result2 = fn(m2)
        self.assertEqual(result2, expected2)
        self.assertEqual(cnt.frame_count, 1)

    def test_recompile_on_shape_change(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @consumable_call
        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        m1 = Consumable(torch.randn(4))
        fn(m1)
        self.assertEqual(cnt.frame_count, 1)

        m2 = Consumable(torch.randn(8))
        fn(m2)
        self.assertEqual(cnt.frame_count, 2)

    def test_refcount_warning(self):
        t = torch.randn(4)
        with self.assertWarnsRegex(UserWarning, "external references"):
            Consumable(t)

    def test_no_warning_for_temporary(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Consumable(torch.randn(4))

    def test_consumable_iter(self):
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        tensors = [torch.randn(4) for _ in range(3)]
        expected = [t + 1 for t in tensors]
        results = [fn(m) for m in consumable_iter(tensors)]
        for r, e in zip(results, expected):
            self.assertEqual(r, e)

    def test_consumable_survives_graph_break(self):
        @consumable_call
        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + x

        t = torch.randn(4)
        expected = t + 1 + t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        result = fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_taken_consumable_value_raises(self):
        m = Consumable(torch.randn(4))
        m.take()
        with self.assertRaises(RuntimeError, msg="value has been consumed"):
            _ = m.value

    def test_taken_consumable_recompile_error(self):
        @consumable_call
        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(torch.randn(4))
        fn(m)
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            fn(m)

    def test_consumable_with_module(self):
        class TestModule(torch.nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.lin = torch.nn.Linear(in_c, out_c)

            def forward(self, x):
                return self.lin(x)

        in_c = 5
        out_c = 2
        tm = TestModule(in_c, out_c)

        @consumable_call
        @torch.compile(backend="inductor", fullgraph=True)
        def call_module(x):
            return tm(x)

        m = Consumable(torch.randn(in_c))
        call_module(m)
        self.assertIsNone(m._value)

    def test_eager_basic(self):
        @consumable_call
        def fn(x):
            return x + 1

        t = torch.randn(4)
        expected = t + 1
        m = Consumable(t)
        result = fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_eager_multiple_args(self):
        @consumable_call
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        result = fn(Consumable(t1), Consumable(t2))
        self.assertEqual(result, expected)

    def test_eager_mixed_consumable_and_tensor(self):
        @consumable_call
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        result = fn(Consumable(t1), t2)
        self.assertEqual(result, expected)

    def test_eager_kwargs(self):
        @consumable_call
        def fn(x, scale=None):
            if scale is not None:
                return x * scale
            return x

        t = torch.randn(4)
        s = torch.tensor(2.0)
        expected = t * s
        result = fn(Consumable(t), scale=Consumable(s))
        self.assertEqual(result, expected)

    def test_eager_and_compiled_same_signature(self):
        """The same decorated function works identically in eager and compiled modes."""

        def add_one(x):
            return x + 1

        eager_fn = consumable_call(add_one)
        compiled_fn = consumable_call(
            torch.compile(add_one, backend="eager", fullgraph=True)
        )

        t = torch.randn(4)
        expected = t + 1

        eager_result = eager_fn(Consumable(t.clone()))
        compiled_result = compiled_fn(Consumable(t.clone()))
        self.assertEqual(eager_result, expected)
        self.assertEqual(compiled_result, expected)

    def test_compile_wraps_consumable_call(self):
        """torch.compile applied after consumable_call (case 2)."""

        @consumable_call
        def fn(x):
            return x + 1

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        expected = t + 1
        m = Consumable(t)
        result = compiled_fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_compile_wraps_consumable_call_multiple_args(self):
        """torch.compile wrapping consumable_call with multiple Consumable args."""

        @consumable_call
        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        result = compiled_fn(Consumable(t1), Consumable(t2))
        self.assertEqual(result, expected)

    def test_primary_order_drives_dynamo(self):
        # M1: with @consumable_call over @torch.compile, the Consumable must reach
        # Dynamo (the compiled prologue does the .take()), not be taken by the
        # eager Python wrapper.
        cnt = torch._dynamo.testing.CompileCounter()

        @consumable_call
        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        self.assertTrue(_is_compiled(fn.__wrapped__))

        import torch._dynamo.consumable as consumable_mod

        take_callers = []
        orig_take = Consumable.take

        def spy_take(self):
            take_callers.append(sys._getframe(1).f_code.co_filename)
            return orig_take(self)

        t = torch.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        Consumable.take = spy_take
        try:
            result = fn(m)
        finally:
            Consumable.take = orig_take

        self.assertEqual(result, t + 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsNone(m._value)
        # The take() must NOT have been driven from the eager wrapper in
        # consumable.py; the compiled frame's generated bytecode does it.
        self.assertTrue(all(c != consumable_mod.__file__ for c in take_callers))
        self.assertTrue(len(take_callers) >= 1)

    def test_mis_specialization_recompiles(self):
        # M2: a non-Consumable arg that merely has a _value attribute must not
        # reuse the Consumable-specialized code (which would call .take()).
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(Consumable(torch.randn(4)))
        self.assertEqual(cnt.frame_count, 1)

        class HasValueNoTake:
            def __init__(self):
                self._value = torch.randn(4)

            def __add__(self, other):
                return self._value + other

        result = fn(HasValueNoTake())
        self.assertEqual(result.shape, torch.Size([4]))
        # The box TYPE_MATCH guard rejects the new type and forces a recompile.
        self.assertEqual(cnt.frame_count, 2)

    def test_double_take_raises_runtime_error(self):
        # M4 eager path: take() on an already-taken Consumable raises RuntimeError.
        m = Consumable(torch.randn(4))
        m.take()
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            m.take()

        # M4 compiled path: re-calling with a taken Consumable surfaces the
        # symbolic_convert already-taken RuntimeError.
        @consumable_call
        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = Consumable(torch.randn(4))
        fn(m2)
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            fn(m2)

    def test_no_external_ref_warning_on_graph_break(self):
        # M5: a graph break on a Consumable input must not construct a new Consumable in
        # Dynamo's generated code (values flow across the break as raw tensors),
        # so the ownership warning in Consumable.__init__ never fires.
        @torch.compile(backend="eager")
        def gb(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + x

        t = torch.randn(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = gb(m)
        self.assertEqual(result, t + 1 + t)
        self.assertIsNone(m._value)
        external = [w for w in caught if "external references" in str(w.message)]
        self.assertEqual(external, [])

    def test_grad_matches_eager(self):
        # A requires_grad Consumable input must match eager forward AND backward
        # under the compiled primary order.
        def f(x):
            return (x * x + x).sum()

        te = torch.randn(4, requires_grad=True)
        ye = f(te)
        ye.backward()

        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def cf(x):
            return (x * x + x).sum()

        tc = te.detach().clone().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(tc)
        yc = cf(m)
        yc.backward()

        self.assertEqual(ye, yc)
        self.assertEqual(te.grad, tc.grad)
        self.assertIsNone(m._value)

    @parametrize("shape", ["scalar", "tuple", "list", "dict"])
    def test_consumable_input_returned_across_graph_break(self, shape):
        # The compiled output must NEVER contain a Consumable: a consumable-sourced
        # input that survives a graph break and appears in the return value must
        # come back as a raw Tensor (matching eager), not reboxed.
        def eager(x):
            y = x + 1
            if shape == "scalar":
                return x
            if shape == "tuple":
                return (x, y)
            if shape == "list":
                return [x, y]
            return {"a": x, "b": y}

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            if shape == "scalar":
                return x
            if shape == "tuple":
                return (x, y)
            if shape == "list":
                return [x, y]
            return {"a": x, "b": y}

        t = torch.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        result = fn(m)
        expected = eager(t)

        self.assertEqual(result, expected)
        self.assertIsNone(m._value)
        # No Consumable anywhere in the returned structure.
        if isinstance(result, dict):
            elems = list(result.values())
        elif isinstance(result, (tuple, list)):
            elems = list(result)
        else:
            elems = [result]
        for e in elems:
            self.assertNotIsInstance(e, Consumable)
            self.assertIsInstance(e, torch.Tensor)

    def test_consumable_input_returned_break_before_use(self):
        # Break before the input is ever used, then return it: the destructive
        # take() still happens exactly once (in the initial frame) and the output
        # is a raw Tensor.
        @torch.compile(backend="eager")
        def fn(x):
            torch._dynamo.graph_break()
            return x

        t = torch.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        result = fn(m)
        self.assertEqual(result, t)
        self.assertNotIsInstance(result, Consumable)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsNone(m._value)

    def test_wrap_consumable_nested_in_list(self):
        # Nested path: a Consumable reached as a list element goes through
        # VariableBuilder.wrap_consumable, not the symbolic_convert root-arg path.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(lst):
            return lst[0] + 1

        t = torch.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        result = fn([m])
        self.assertEqual(result, t + 1)
        self.assertNotIsInstance(result, Consumable)
        self.assertIsNone(m._value)

    def test_wrap_consumable_nested_in_dict(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(d):
            return d["x"] * 2

        t = torch.arange(4.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Consumable(t)
        result = fn({"x": m})
        self.assertEqual(result, t * 2)
        self.assertNotIsInstance(result, Consumable)
        self.assertIsNone(m._value)

    def test_no_recompile_calling_consumable_call_twice(self):
        # Two calls with fresh Consumables of the same tensor metadata must reuse the
        # first compilation: the box TYPE_MATCH + inner TENSOR_MATCH guards hold.
        cnt = torch._dynamo.testing.CompileCounter()

        @consumable_call
        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        t1 = torch.randn(4)
        r1 = fn(_mk(Consumable, t1))
        self.assertEqual(r1, t1 + 1)
        self.assertEqual(cnt.frame_count, 1)

        t2 = torch.randn(4)
        r2 = fn(_mk(Consumable, t2))
        self.assertEqual(r2, t2 + 1)
        self.assertEqual(cnt.frame_count, 1)

    def test_dict_value_guards_match_plain(self):
        # A Consumable holding a dict must produce the same guards as passing the
        # plain dict: no recompile for same structure, recompile when the accessed
        # element's tensor metadata changes.
        def build(counter_backend):
            @torch.compile(backend=counter_backend)
            def fn(d):
                return d["a"] * 2

            return fn

        plain_cnt = torch._dynamo.testing.CompileCounter()
        plain = build(plain_cnt)
        plain({"a": torch.ones(4)})
        plain({"a": torch.ones(4)})
        plain({"a": torch.ones(8)})

        torch._dynamo.reset()
        mov_cnt = torch._dynamo.testing.CompileCounter()
        mov = build(mov_cnt)
        r1 = mov(_mk(Consumable, {"a": torch.ones(4)}))
        self.assertEqual(r1, torch.ones(4) * 2)
        self.assertEqual(mov_cnt.frame_count, 1)
        mov(_mk(Consumable, {"a": torch.ones(4)}))
        self.assertEqual(mov_cnt.frame_count, 1)
        r3 = mov(_mk(Consumable, {"a": torch.ones(8)}))
        self.assertEqual(r3, torch.ones(8) * 2)
        self.assertEqual(mov_cnt.frame_count, 2)

        # The Consumable path must not over- or under-guard relative to the plain path.
        self.assertEqual(mov_cnt.frame_count, plain_cnt.frame_count)

    def test_dynamic_shape_no_recompile(self):
        # A tensor marked dynamic inside a Consumable must not recompile when the
        # dynamic dim changes size (the inner TENSOR_MATCH honors the symbolic dim).
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        t1 = torch.randn(4)
        torch._dynamo.mark_dynamic(t1, 0)
        fn(_mk(Consumable, t1))
        self.assertEqual(cnt.frame_count, 1)

        t2 = torch.randn(8)
        torch._dynamo.mark_dynamic(t2, 0)
        r = fn(_mk(Consumable, t2))
        self.assertEqual(r, t2 + 1)
        self.assertEqual(cnt.frame_count, 1)

    @parametrize("container", ["dict", "list"])
    def test_consumable_container_survives_graph_break(self, container):
        # A Consumable holding a container (dict/list) that survives a graph break
        # must reconstruct the container for the resume region from graph outputs
        # -- the box is consumed by a single take(), so there is no aliasable
        # original object. Each element crosses the break as a raw Tensor.
        def eager(c):
            if container == "dict":
                y = c["a"] + 1
                return c["a"] + y
            y = c[0] + 1
            return c[0] + y

        @torch.compile(backend="eager")
        def fn(c):
            if container == "dict":
                y = c["a"] + 1
                torch._dynamo.graph_break()
                return c["a"] + y
            y = c[0] + 1
            torch._dynamo.graph_break()
            return c[0] + y

        t = torch.ones(4)
        raw = {"a": t} if container == "dict" else [t]
        m = _mk(Consumable, {"a": t} if container == "dict" else [t])
        result = fn(m)
        self.assertEqual(result, eager(raw))
        self.assertNotIsInstance(result, Consumable)
        self.assertIsNone(m._value)

    def test_consumable_dict_extra_key_survives_graph_break(self):
        # A key not used before the break must still be present after it: the
        # container is fully rebuilt, and the unused element is materialized as a
        # graph output on demand.
        def eager(d):
            y = d["a"] + 1
            return d["a"] + d["b"] + y

        @torch.compile(backend="eager")
        def fn(d):
            y = d["a"] + 1
            torch._dynamo.graph_break()
            return d["a"] + d["b"] + y

        raw = {"a": torch.ones(4), "b": torch.full((4,), 2.0)}
        m = _mk(Consumable, {"a": torch.ones(4), "b": torch.full((4,), 2.0)})
        result = fn(m)
        self.assertEqual(result, eager(raw))
        self.assertIsNone(m._value)

    @parametrize("container", ["dict", "list"])
    def test_consumable_container_returned_across_graph_break(self, container):
        # The container itself is returned after the break; it must come back as a
        # plain dict/list of raw Tensors, never wrapped in a Consumable.
        @torch.compile(backend="eager")
        def fn(c):
            if container == "dict":
                _ = c["a"] + 1
                torch._dynamo.graph_break()
                return c
            _ = c[0] + 1
            torch._dynamo.graph_break()
            return c

        t = torch.ones(4)
        m = _mk(Consumable, {"a": t} if container == "dict" else [t])
        result = fn(m)
        self.assertNotIsInstance(result, Consumable)
        elems = list(result.values()) if container == "dict" else list(result)
        for e in elems:
            self.assertNotIsInstance(e, Consumable)
            self.assertIsInstance(e, torch.Tensor)
        self.assertEqual(elems[0], t)
        self.assertIsNone(m._value)

    @parametrize("container", ["dict", "list"])
    def test_consumable_container_mutated_in_region(self, container):
        # Finding 1: mutating a Consumable-held container in-region makes side-effect
        # replay reconstruct the mutation target via cg(var.source), which calls
        # ConsumableSource.reconstruct() -> take() a SECOND time (the prologue already
        # consumed the box). Eager consumable_call takes once and mutates the raw
        # container fine, so compiled must not diverge with a crash.
        def body(c):
            if container == "dict":
                c["b"] = c["a"] + 1
                return c["a"] * 2
            c.append(c[0] + 1)
            return c[0] * 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(c):
            return body(c)

        raw = {"a": torch.ones(4)} if container == "dict" else [torch.ones(4)]
        m = _mk(
            Consumable, {"a": torch.ones(4)} if container == "dict" else [torch.ones(4)]
        )
        self.assertEqual(fn(m), body(raw))
        self.assertIsNone(m._value)

    def test_consumable_namedtuple_fullgraph(self):
        # A Consumable holding a namedtuple: each accessed field feeds the graph
        # via an AttrSource rooted at the ConsumableSource, and the box is taken
        # exactly once even though both fields become graph inputs.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(p):
            return p.a + p.b

        a, b = torch.randn(4), torch.randn(4)
        m = _mk(Consumable, _Pair(a, b))
        self.assertEqual(fn(m), a + b)
        self.assertIsNone(m._value)

    def test_consumable_namedtuple_survives_graph_break(self):
        # Fields used on both sides of a break: the namedtuple must be rebuilt for
        # the resume region from graph outputs (the box is consumed by a single
        # take(), so there is no aliasable original).
        def eager(p):
            y = p.a + 1
            return p.a + p.b + y

        @torch.compile(backend="eager")
        def fn(p):
            y = p.a + 1
            torch._dynamo.graph_break()
            return p.a + p.b + y

        a, b = torch.ones(4), torch.full((4,), 2.0)
        m = _mk(Consumable, _Pair(a, b))
        self.assertEqual(fn(m), eager(_Pair(a, b)))
        self.assertIsNone(m._value)

    def test_consumable_namedtuple_returned_across_graph_break(self):
        # The namedtuple itself is returned after a break. It must come back as
        # the same namedtuple type holding raw Tensors, never a Consumable. Before
        # the fix this raised AssertionError in UserDefinedTupleVariable.reconstruct,
        # which rejected any sourced tuple; consumable-rooted sources are now
        # rebuilt from items instead of aliasing the (already taken) box.
        @torch.compile(backend="eager")
        def fn(p):
            _ = p.a + 1
            torch._dynamo.graph_break()
            return p

        a, b = torch.ones(4), torch.full((4,), 2.0)
        m = _mk(Consumable, _Pair(a, b))
        result = fn(m)
        self.assertNotIsInstance(result, Consumable)
        self.assertIsInstance(result, _Pair)
        self.assertEqual(result.a, a)
        self.assertEqual(result.b, b)
        self.assertIsNone(m._value)

    def test_consumable_namedtuple_inner_freed_during_forward(self):
        # A used field of a Consumable-held namedtuple is moved into the graph and
        # freed mid-forward, exactly like a bare consumable tensor: it lands in the
        # boxed input list the compiled forward clears in place.
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(p, y):
            a = p.a + 1
            b = a * 2
            c = b + a
            d = c * 3
            return torch.ops.consumable_test_probe.mark(d) + y.sum()

        t_a = torch.randn(64, 64)
        t_b = torch.randn(64, 64)
        t_y = torch.randn(64, 64)
        _probe_state["wref"] = weakref.ref(t_a)
        _probe_state["alive_at_mark"] = None
        m = _mk(Consumable, _Pair(t_a, t_b))
        del t_a, t_b  # the box holds the only strong refs now
        out = fn(m, t_y)
        # The used field was dead by the late mark op (freed mid-forward).
        self.assertIs(_probe_state["alive_at_mark"], False)
        self.assertIsNone(m._value)
        self.assertIsNone(_probe_state["wref"]())
        self.assertEqual(out.shape, torch.Size([64, 64]))

    def test_consumable_tensor_subclass_fullgraph(self):
        # A traceable wrapper subclass flattens to several inner tensors, each a
        # graph input sourced via AttrSource off the ConsumableSource. All of them
        # root at one box, so the existing take-once dedup consumes it exactly once.
        from torch.testing._internal.two_tensor import TwoTensor

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        a, b = torch.randn(4), torch.randn(4)
        m = _mk(Consumable, TwoTensor(a, b))
        result = fn(m)
        self.assertIsInstance(result, TwoTensor)
        self.assertEqual(result.a, a + 1)
        self.assertEqual(result.b, b + 1)
        self.assertIsNone(m._value)

    def test_consumable_tensor_subclass_returned_across_graph_break(self):
        # A Consumable-held subclass returned after a break is rebuilt from graph
        # outputs via TensorWithTFOverrideVariable (to_subclass), never reboxed.
        from torch.testing._internal.two_tensor import TwoTensor

        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return x + y

        a, b = torch.ones(4), torch.full((4,), 2.0)
        m = _mk(Consumable, TwoTensor(a, b))
        result = fn(m)
        self.assertNotIsInstance(result, Consumable)
        self.assertIsInstance(result, TwoTensor)
        self.assertEqual(result.a, a + (a + 1))
        self.assertEqual(result.b, b + (b + 1))
        self.assertIsNone(m._value)

    def test_consumable_tensor_subclass_survives_graph_break(self):
        # Subclass used on both sides of a break, matching eager numerics.
        from torch.testing._internal.two_tensor import TwoTensor

        def eager(x):
            y = x * 2
            return x + y

        @torch.compile(backend="eager")
        def fn(x):
            y = x * 2
            torch._dynamo.graph_break()
            return x + y

        a, b = torch.randn(4), torch.randn(4)
        m = _mk(Consumable, TwoTensor(a, b))
        result = fn(m)
        expected = eager(TwoTensor(a, b))
        self.assertEqual(result.a, expected.a)
        self.assertEqual(result.b, expected.b)
        self.assertIsNone(m._value)

    @unittest.expectedFailure
    def test_taken_box_as_unused_arg_matches_eager(self):
        # Finding 2: an unused Consumable arg only installs a box TYPE_MATCH guard (the
        # inner TENSOR_MATCH is never armed), so an already-taken box (_value is
        # None) still hits the compiled cache line and returns -- while eager
        # consumable_call calls take() and raises. Compiled must match eager.
        @consumable_call
        @torch.compile(backend="eager")
        def compiled(x, y):  # y unused
            return x + 1

        @consumable_call
        def eager(x, y):  # y unused
            return x + 1

        taken = _mk(Consumable, torch.randn(4))
        taken.take()
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            eager(_mk(Consumable, torch.randn(4)), taken)

        taken2 = _mk(Consumable, torch.randn(4))
        taken2.take()
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            compiled(_mk(Consumable, torch.randn(4)), taken2)

    @unittest.expectedFailure
    def test_unused_consumable_arg_is_still_consumed(self):
        # Finding 3: take() is only emitted where the arg is loaded as a grapharg,
        # so an unused Consumable arg's box is never emptied and the tensor is never
        # released -- the whole point of Consumable silently no-ops with no warning.
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):  # y unused
            return x + 1

        m_used = _mk(Consumable, torch.randn(4))
        m_unused = _mk(Consumable, torch.ones(4))
        fn(m_used, m_unused)
        self.assertIsNone(m_used._value)
        self.assertIsNone(m_unused._value)

    def test_aliased_consumables_both_consumed(self):
        # Finding 3 (aliasing variant): two Consumables wrapping the same tensor get
        # deduped to one grapharg, so only one box is emptied. Both boxes should be
        # consumed so neither silently retains the tensor.
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return x + y

        t = torch.randn(4)
        a = _mk(Consumable, t)
        b = _mk(Consumable, t)
        result = fn(a, b)
        self.assertEqual(result, t + t)
        self.assertIsNone(a._value)
        self.assertIsNone(b._value)

    def test_consumable_consumed_inside_hop(self):
        # A Consumable used only inside a HOP body still auto-lifts to a root
        # grapharg, so the pre-graph prologue takes its box. Guards the grapharg
        # take() path for boxes first realized in a subtracer.
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(pred, x):
            return torch.cond(pred, lambda: x + 1, lambda: x - 1, ())

        t = torch.randn(4)
        m = _mk(Consumable, t)
        result = fn(torch.tensor(True), m)
        self.assertEqual(result, t + 1)
        self.assertIsNone(m._value)

    def test_aliased_consumables_consumed_inside_hop(self):
        # Aliasing variant of the HOP case: two Consumables wrapping the same tensor
        # dedupe to one grapharg, and the deduped alias is realized inside the HOP
        # subtracer. That subtracer's traced_sources is gone by codegen time, so
        # the alias box is never taken and silently retains the tensor.
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(pred, x, y):
            return torch.cond(pred, lambda: x + y, lambda: x - y, ())

        t = torch.randn(4)
        a = _mk(Consumable, t)
        b = _mk(Consumable, t)
        result = fn(torch.tensor(True), a, b)
        self.assertEqual(result, t + t)
        self.assertIsNone(a._value)
        self.assertIsNone(b._value)

    @torch._dynamo.config.patch(generate_pycode=True)
    def test_consumable_pycode_path(self):
        # Finding 4: the use_python_codegen / pycode wrapper path (used by AOT
        # compile and functional export, not by torch.compile) must handle a
        # Consumable input like the bytecode path. Today the generated pycode passes
        # the raw box as the grapharg (no take()), so the graph receives a Consumable
        # and raises TypeError. Reached only via fullgraph_capture +
        # forward_callable(use_python_codegen=True), as in test_pycode.py.
        from torch._dynamo.convert_frame import fullgraph_capture
        from torch._dynamo.utils import get_metrics_context

        def fn(x):
            return x + 1

        with get_metrics_context():
            cap = fullgraph_capture(fn, (_mk(Consumable, torch.ones(4)),), {})
        runtime_env = cap.graph_capture_output.get_runtime_env()
        backend_id = cap.backend_input.backend_id
        compiled = runtime_env.forward_callable(
            backend_id, cap.backend_input.graph_module, use_python_codegen=True
        )
        m = _mk(Consumable, torch.ones(4))
        self.assertEqual(compiled(m), torch.ones(4) + 1)
        self.assertIsNone(m._value)

    def test_eager_consumable_uses_boxed_codegen_not_interpreter(self):
        # The eager consumable path must free the consumed input mid-forward WITHOUT
        # torch.fx.Interpreter. It installs _BoxedCodeGen on the backend
        # GraphModule so the generated forward(self, args_list) clears the list
        # in place. Assert both (M1) Interpreter.boxed_run is never called and
        # (M2) the consumed input is already dead when a custom op placed after its
        # last use runs.
        calls = {"boxed_run": 0}
        orig_boxed_run = torch.fx.Interpreter.boxed_run

        def spy_boxed_run(self, *args, **kwargs):
            calls["boxed_run"] += 1
            return orig_boxed_run(self, *args, **kwargs)

        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            # x (consumable) is used only up to `c`; the rest of the chain does not
            # touch x, so it can be freed well before mark() runs.
            a = x + 1
            b = a * 2
            c = b + a
            d = c * 3
            return torch.ops.consumable_test_probe.mark(d) + y.sum()

        t_x = torch.randn(64, 64)
        t_y = torch.randn(64, 64)
        _probe_state["wref"] = weakref.ref(t_x)
        _probe_state["alive_at_mark"] = None
        m = _mk(Consumable, t_x)
        del t_x  # the box holds the only strong ref now

        torch.fx.Interpreter.boxed_run = spy_boxed_run
        try:
            out = fn(m, t_y)
        finally:
            torch.fx.Interpreter.boxed_run = orig_boxed_run

        # M1: no Interpreter on the eager consumable path.
        self.assertEqual(calls["boxed_run"], 0)
        # M2: the consumed input was freed mid-forward (dead at the late mark op).
        # assertIs(..., False) not assertFalse: if mark never ran, alive_at_mark
        # stays None and assertFalse(None) would pass vacuously.
        self.assertIs(_probe_state["alive_at_mark"], False)
        self.assertIsNone(m._value)
        self.assertIsNone(_probe_state["wref"]())
        self.assertEqual(out.shape, torch.Size([64, 64]))

    def test_eager_consumable_non_lazy_graph_module(self):
        # Covers the non-lazy elif branch in compile_subgraph: with lazy graph
        # modules disabled the backend GM is a plain GraphModule, so the boxed
        # codegen is regenerated via eager_gm.recompile() (not force_recompile).
        @consumable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return x + y

        t = torch.randn(4)
        s = torch.randn(4)
        expected = t + s
        with torch._dynamo.config.patch(use_lazy_graph_module=False):
            m = _mk(Consumable, t.clone())
            out = fn(m, s)
        self.assertEqual(out, expected)
        self.assertIsNone(m._value)


class TestConsumablePgo(TestCase):
    """PGO regression guards for boxed graph inputs.

    PR #192868 reverted resume-frame boxing (#185561) because boxing rewrote a
    value's PGO source from its logical local (e.g. L['x']) to a boxed-slot
    access, and that renamed identity leaked into PGO's dynamic-shape whitelist,
    breaking cross-run matching and forcing recompiles (issue #153701). These
    tests lock in the invariant that boxing -- Consumable inputs today, and any
    future unified auto-boxing -- must never change the logical PGO source name:
    boxing stays a codegen/runtime concern, invisible to guards and PGO.
    """

    def setUp(self):
        super().setUp()
        self._pgo_stack = contextlib.ExitStack()
        self._pgo_stack.enter_context(torch.compiler.config.patch(job_id=self.id()))
        self._pgo_stack.enter_context(
            torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
        )
        if os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1":
            self._pgo_stack.enter_context(fresh_cache())

    def tearDown(self):
        torch._dynamo.reset()
        self._pgo_stack.close()
        super().tearDown()

    @staticmethod
    def _all_dynamic_source_keys():
        from torch._dynamo.pgo import get_code_state

        keys = set()
        for code_state in get_code_state().values():
            keys |= set(code_state.automatic_dynamic.keys())
        return keys

    def _assert_no_boxed_slot_names(self, names):
        # Names Dynamo uses for the boxed calling convention; none may ever
        # surface as a PGO source (that is precisely the #192868 regression).
        for name in names:
            for bad in (
                "__resume_frame_values",
                "consumable_inputs",
                "frame_values",
                "__boxed",
            ):
                self.assertNotIn(
                    bad, name, f"boxed-slot name leaked into PGO source: {name!r}"
                )

    def test_resume_del_source_names_are_logical(self):
        # The #192868 failure mode without Consumable: a graph break followed by a
        # reachable `del` is exactly what #185561 boxed. The resume frame's PGO
        # sources must stay logical locals, not boxed-slot accesses.
        @torch.compile(backend="eager")
        def f(x, y):
            a = x + 1
            torch._dynamo.graph_break()
            b = y * 2
            del x
            return a.sum() + b.sum()

        f(torch.randn(4), torch.randn(4))
        f(torch.randn(8), torch.randn(8))

        keys = self._all_dynamic_source_keys()
        self.assertIn("L['x']", keys)
        self.assertIn("L['y']", keys)
        self._assert_no_boxed_slot_names(keys)

    def test_consumable_source_is_logical_across_break(self):
        # A Consumable input crossing a graph break records its fully logical
        # source (L['x'], with the box elided by consumable_logical_name), never
        # the synthesized boxed-list-index source and never a stray ._value
        # suffix -- both PGO and the user-facing dynamic_sources config key on
        # the same name the value would have had unboxed.
        @torch.compile(backend="eager")
        def f(x):
            a = x + 1
            torch._dynamo.graph_break()
            return a * x

        f(_mk(Consumable, torch.randn(4)))
        f(_mk(Consumable, torch.randn(8)))

        keys = self._all_dynamic_source_keys()
        self.assertIn("L['x']", keys)
        self._assert_no_boxed_slot_names(keys)

    def test_consumable_whitelist_reused_no_recompile(self):
        # The exact property #192868 lost: the PGO source recorded for a boxed
        # (Consumable) input must be stable enough that replaying the collected
        # dynamic-source whitelist in a FRESH pgo session (different job_id, so no
        # local profile carryover) makes the dim dynamic from the first call and
        # avoids a recompile. A boxed-slot rename would not match and recompile.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return x + 1

        f(_mk(Consumable, torch.randn(4)))
        f(_mk(Consumable, torch.randn(8)))  # second size marks the dim dynamic
        self.assertEqual(cnt.frame_count, 2)

        from torch._dynamo.pgo import get_code_state, render_code_state

        state = render_code_state(get_code_state())
        match = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state)
        self.assertIsNotNone(match)
        whitelist = match.group(1)
        self.assertIn("L['x']", whitelist)

        torch._dynamo.reset()
        cnt.clear()
        with torch.compiler.config.patch(
            job_id=self.id() + "_reuse", dynamic_sources=whitelist
        ):
            f(_mk(Consumable, torch.randn(4)))
            f(_mk(Consumable, torch.randn(6)))
            self.assertEqual(cnt.frame_count, 1)


if __name__ == "__main__":
    run_tests()
