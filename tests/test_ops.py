"""前向对拍: ops.* vs mx.* (逐算子全等)。"""

from __future__ import annotations

import mlx.core as mx

mx.random.seed(0)

from tinyautograd import Tensor, ops, random


def _pair(seed: int = 0) -> tuple[mx.array, mx.array]:
    a = mx.random.normal((3, 4), key=mx.random.key(seed))
    b = mx.random.normal((3, 4), key=mx.random.key(seed + 1))
    return a, b


def _check(t: Tensor, ref: mx.array) -> None:
    from tinyautograd.autograd import eval

    eval(t)
    assert t._array.dtype == ref.dtype
    assert mx.allclose(t._array, ref).item(), f"{t._array} != {ref}"


def test_binary_ops() -> None:
    a, b = _pair()
    ta, tb = Tensor(a), Tensor(b)
    _check(ops.add(ta, tb), mx.add(a, b))
    _check(ops.sub(ta, tb), mx.subtract(a, b))
    _check(ops.mul(ta, tb), mx.multiply(a, b))
    _check(ops.div(ta, tb), mx.divide(a, b))
    _check(ta + tb, mx.add(a, b))
    _check(ta - tb, mx.subtract(a, b))
    _check(ta * tb, mx.multiply(a, b))
    _check(ta / tb, mx.divide(a, b))
    _check(1.0 + ta, mx.add(a, 1.0))  # 反射 + 标量包装


def test_unary_ops() -> None:
    a, _ = _pair()
    ta = Tensor(a)
    _check(ops.neg(ta), mx.negative(a))
    _check(ops.exp(ta), mx.exp(a))
    _check(ops.log(ta * ta), mx.log(a * a))
    _check(ops.sqrt(ta * ta), mx.sqrt(a * a))
    _check(ops.square(ta), mx.square(a))
    _check(ops.sigmoid(ta), mx.sigmoid(a))
    _check(ops.tanh(ta), mx.tanh(a))
    _check(ops.relu(ta), mx.maximum(a, 0.0))
    _check(-ta, mx.negative(a))
    _check(ta**2, mx.power(a, 2.0))


def test_matmul() -> None:
    a = mx.random.normal((3, 4), key=mx.random.key(0))
    b = mx.random.normal((4, 5), key=mx.random.key(1))
    _check(Tensor(a) @ Tensor(b), a @ b)


def test_reductions() -> None:
    a, _ = _pair()
    ta = Tensor(a)
    _check(ops.sum(ta), mx.sum(a))
    _check(ops.sum(ta, axis=0), mx.sum(a, axis=0))
    _check(ops.mean(ta, axis=1, keepdims=True), mx.mean(a, axis=1, keepdims=True))
    _check(ops.max(ta, axis=0), mx.max(a, axis=0))


def test_shape_ops() -> None:
    a, _ = _pair()
    ta = Tensor(a)
    _check(ops.transpose(ta), mx.transpose(a))
    _check(ta.T, mx.transpose(a))
    _check(ops.reshape(ta, (4, 3)), mx.reshape(a, (4, 3)))


def test_getitem() -> None:
    a, _ = _pair()
    ta = Tensor(a)
    _check(ta[1], a[1])
    _check(ta[:, 2], a[:, 2])
    _check(ta[1:3, ::2], a[1:3, ::2])
    _check(ta[mx.array([0, 2])], a[mx.array([0, 2])])  # 花式索引 (D4)
    _check(ta[None, ..., 0], a[None, ..., 0])  # None + Ellipsis


def test_constructors() -> None:
    _check(ops.zeros((2, 3)), mx.zeros((2, 3)))
    _check(ops.ones((2,)), mx.ones((2,)))
    _check(ops.arange(5), mx.arange(5))
    t = random.normal((2, 2))
    from tinyautograd.autograd import eval

    eval(t)
    assert t.shape == (2, 2)
    a, _ = _pair()
    _check(ops.zeros_like(Tensor(a)), mx.zeros_like(a))
    _check(ops.ones_like(Tensor(a)), mx.ones_like(a))


def test_compose() -> None:
    a, _ = _pair()
    f = ops.compose(ops.square, ops.exp)  # f = sq∘exp
    _check(f(Tensor(a)), mx.square(mx.exp(a)))

def test_softmax() -> None:
    a = Tensor([1.0, 2.0, 3.0])
    assert ops.softmax(a).tolist() == mx.softmax(mx.array([1.0, 2.0, 3.0])).tolist()

