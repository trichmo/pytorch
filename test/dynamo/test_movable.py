# Owner(s): ["module: dynamo"]

import sys
import warnings

import torch
from torch._dynamo.movable import _is_compiled, Movable, movable_call, movable_iter
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class TestMovable(TestCase):
    def test_basic(self):
        @movable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        t = torch.randn(4)
        expected = t + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(t)
        result = fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_movable_can_del(self):
        @movable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x + 1
            del x
            return y

        m = Movable(torch.tensor([1, 2, 3]))
        result = fn(m)
        self.assertEqual(result, torch.tensor([1, 2, 3]) + 1)
        self.assertIsNone(m._value)

    def test_multiple_args(self):
        @movable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = Movable(t1)
            m2 = Movable(t2)
        result = fn(m1, m2)
        self.assertEqual(result, expected)
        self.assertIsNone(m1._value)
        self.assertIsNone(m2._value)

    def test_mixed_movable_and_tensor(self):
        @movable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = Movable(t1)
        result = fn(m1, t2)
        self.assertEqual(result, expected)
        self.assertIsNone(m1._value)

    def test_second_call_with_fresh_movable(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @movable_call
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x * 2

        t1 = torch.randn(4)
        expected1 = t1 * 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = Movable(t1)
        result1 = fn(m1)
        self.assertEqual(result1, expected1)
        self.assertEqual(cnt.frame_count, 1)

        t2 = torch.randn(4)
        expected2 = t2 * 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = Movable(t2)
        result2 = fn(m2)
        self.assertEqual(result2, expected2)
        self.assertEqual(cnt.frame_count, 1)

    def test_recompile_on_shape_change(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @movable_call
        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        m1 = Movable(torch.randn(4))
        fn(m1)
        self.assertEqual(cnt.frame_count, 1)

        m2 = Movable(torch.randn(8))
        fn(m2)
        self.assertEqual(cnt.frame_count, 2)

    def test_refcount_warning(self):
        t = torch.randn(4)
        with self.assertWarnsRegex(UserWarning, "external references"):
            Movable(t)

    def test_no_warning_for_temporary(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Movable(torch.randn(4))

    def test_movable_iter(self):
        @movable_call
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        tensors = [torch.randn(4) for _ in range(3)]
        expected = [t + 1 for t in tensors]
        results = [fn(m) for m in movable_iter(tensors)]
        for r, e in zip(results, expected):
            self.assertEqual(r, e)

    def test_movable_survives_graph_break(self):
        @movable_call
        @torch.compile(backend="eager")
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + x

        t = torch.randn(4)
        expected = t + 1 + t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(t)
        result = fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_taken_movable_value_raises(self):
        m = Movable(torch.randn(4))
        m.take()
        with self.assertRaises(RuntimeError, msg="value has been moved"):
            _ = m.value

    def test_taken_movable_recompile_error(self):
        @movable_call
        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(torch.randn(4))
        fn(m)
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            fn(m)

    def test_movable_with_module(self):
        class TestModule(torch.nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.lin = torch.nn.Linear(in_c, out_c)

            def forward(self, x):
                return self.lin(x)

        in_c = 5
        out_c = 2
        tm = TestModule(in_c, out_c)

        @movable_call
        @torch.compile(backend="inductor", fullgraph=True)
        def call_module(x):
            return tm(x)

        m = Movable(torch.randn(in_c))
        call_module(m)
        self.assertIsNone(m._value)

    def test_eager_basic(self):
        @movable_call
        def fn(x):
            return x + 1

        t = torch.randn(4)
        expected = t + 1
        m = Movable(t)
        result = fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_eager_multiple_args(self):
        @movable_call
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        result = fn(Movable(t1), Movable(t2))
        self.assertEqual(result, expected)

    def test_eager_mixed_movable_and_tensor(self):
        @movable_call
        def fn(x, y):
            return x + y

        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        result = fn(Movable(t1), t2)
        self.assertEqual(result, expected)

    def test_eager_kwargs(self):
        @movable_call
        def fn(x, scale=None):
            if scale is not None:
                return x * scale
            return x

        t = torch.randn(4)
        s = torch.tensor(2.0)
        expected = t * s
        result = fn(Movable(t), scale=Movable(s))
        self.assertEqual(result, expected)

    def test_eager_and_compiled_same_signature(self):
        """The same decorated function works identically in eager and compiled modes."""

        def add_one(x):
            return x + 1

        eager_fn = movable_call(add_one)
        compiled_fn = movable_call(
            torch.compile(add_one, backend="eager", fullgraph=True)
        )

        t = torch.randn(4)
        expected = t + 1

        eager_result = eager_fn(Movable(t.clone()))
        compiled_result = compiled_fn(Movable(t.clone()))
        self.assertEqual(eager_result, expected)
        self.assertEqual(compiled_result, expected)

    def test_compile_wraps_movable_call(self):
        """torch.compile applied after movable_call (case 2)."""

        @movable_call
        def fn(x):
            return x + 1

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        expected = t + 1
        m = Movable(t)
        result = compiled_fn(m)
        self.assertEqual(result, expected)
        self.assertIsNone(m._value)

    def test_compile_wraps_movable_call_multiple_args(self):
        """torch.compile wrapping movable_call with multiple Movable args."""

        @movable_call
        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t1 = torch.randn(4)
        t2 = torch.randn(4)
        expected = t1 + t2
        result = compiled_fn(Movable(t1), Movable(t2))
        self.assertEqual(result, expected)

    def test_primary_order_drives_dynamo(self):
        # M1: with @movable_call over @torch.compile, the Movable must reach
        # Dynamo (the compiled prologue does the .take()), not be taken by the
        # eager Python wrapper.
        cnt = torch._dynamo.testing.CompileCounter()

        @movable_call
        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        self.assertTrue(_is_compiled(fn.__wrapped__))

        import torch._dynamo.movable as movable_mod

        take_callers = []
        orig_take = Movable.take

        def spy_take(self):
            take_callers.append(sys._getframe(1).f_code.co_filename)
            return orig_take(self)

        t = torch.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(t)
        Movable.take = spy_take
        try:
            result = fn(m)
        finally:
            Movable.take = orig_take

        self.assertEqual(result, t + 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsNone(m._value)
        # The take() must NOT have been driven from the eager wrapper in
        # movable.py; the compiled frame's generated bytecode does it.
        self.assertTrue(all(c != movable_mod.__file__ for c in take_callers))
        self.assertTrue(len(take_callers) >= 1)

    def test_mis_specialization_recompiles(self):
        # M2: a non-Movable arg that merely has a _value attribute must not
        # reuse the Movable-specialized code (which would call .take()).
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(Movable(torch.randn(4)))
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
        # M4 eager path: take() on an already-taken Movable raises RuntimeError.
        m = Movable(torch.randn(4))
        m.take()
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            m.take()

        # M4 compiled path: re-calling with a taken Movable surfaces the
        # symbolic_convert already-taken RuntimeError.
        @movable_call
        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = Movable(torch.randn(4))
        fn(m2)
        with self.assertRaisesRegex(RuntimeError, "already been taken"):
            fn(m2)

    def test_no_external_ref_warning_on_graph_break(self):
        # M5: a graph break on a Movable input must not construct a new Movable in
        # Dynamo's generated code (values flow across the break as raw tensors),
        # so the ownership warning in Movable.__init__ never fires.
        @torch.compile(backend="eager")
        def gb(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y + x

        t = torch.randn(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(t)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = gb(m)
        self.assertEqual(result, t + 1 + t)
        self.assertIsNone(m._value)
        external = [w for w in caught if "external references" in str(w.message)]
        self.assertEqual(external, [])

    def test_grad_matches_eager(self):
        # A requires_grad Movable input must match eager forward AND backward
        # under the compiled primary order.
        def f(x):
            return (x * x + x).sum()

        te = torch.randn(4, requires_grad=True)
        ye = f(te)
        ye.backward()

        @movable_call
        @torch.compile(backend="eager", fullgraph=True)
        def cf(x):
            return (x * x + x).sum()

        tc = te.detach().clone().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(tc)
        yc = cf(m)
        yc.backward()

        self.assertEqual(ye, yc)
        self.assertEqual(te.grad, tc.grad)
        self.assertIsNone(m._value)

    @parametrize("shape", ["scalar", "tuple", "list", "dict"])
    def test_movable_input_returned_across_graph_break(self, shape):
        # The compiled output must NEVER contain a Movable: a movable-sourced
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
            m = Movable(t)
        result = fn(m)
        expected = eager(t)

        self.assertEqual(result, expected)
        self.assertIsNone(m._value)
        # No Movable anywhere in the returned structure.
        if isinstance(result, dict):
            elems = list(result.values())
        elif isinstance(result, (tuple, list)):
            elems = list(result)
        else:
            elems = [result]
        for e in elems:
            self.assertNotIsInstance(e, Movable)
            self.assertIsInstance(e, torch.Tensor)

    def test_movable_input_returned_break_before_use(self):
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
            m = Movable(t)
        result = fn(m)
        self.assertEqual(result, t)
        self.assertNotIsInstance(result, Movable)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsNone(m._value)

    def test_wrap_movable_nested_in_list(self):
        # Nested path: a Movable reached as a list element goes through
        # VariableBuilder.wrap_movable, not the symbolic_convert root-arg path.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(lst):
            return lst[0] + 1

        t = torch.ones(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(t)
        result = fn([m])
        self.assertEqual(result, t + 1)
        self.assertNotIsInstance(result, Movable)
        self.assertIsNone(m._value)

    def test_wrap_movable_nested_in_dict(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(d):
            return d["x"] * 2

        t = torch.arange(4.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Movable(t)
        result = fn({"x": m})
        self.assertEqual(result, t * 2)
        self.assertNotIsInstance(result, Movable)
        self.assertIsNone(m._value)


if __name__ == "__main__":
    run_tests()
