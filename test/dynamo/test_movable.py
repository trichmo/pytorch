# Owner(s): ["module: dynamo"]

import warnings

import torch
from torch._dynamo.movable import Movable, movable_iter
from torch._dynamo.test_case import run_tests, TestCase


class TestMovable(TestCase):
    def test_basic(self):
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
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = x + 1
            del x
            return y

        m = Movable(torch.tensor([1,2,3]))
        result = fn(m)
        self.assertEqual(result, torch.tensor([1,2,3]) + 1)
        self.assertIsNone(m._value)

    def test_multiple_args(self):
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
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x + 1

        tensors = [torch.randn(4) for _ in range(3)]
        expected = [t + 1 for t in tensors]
        results = [fn(m) for m in movable_iter(tensors)]
        for r, e in zip(results, expected):
            self.assertEqual(r, e)

    def test_movable_survives_graph_break(self):
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
        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

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

        @torch.compile(backend="inductor", fullgraph=True)
        def call_module(x):
            return tm(x)

        m = Movable(torch.randn(in_c))
        call_module(m)
        self.assertIsNone(m._value)



if __name__ == "__main__":
    run_tests()
