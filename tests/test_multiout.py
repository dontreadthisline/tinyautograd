"""多输出算子: stack / split / broadcast_to 前向 + 反向对拍 (mx 为 oracle)。

stack       : 多 Tensor 进、单 Tensor 出 (lift 的 *args 签名)
split       : 单 Tensor 进、多 Tensor 出 (lift_multi, 共享 op + index)
broadcast_to: 单进单出 (梯度由 mx.vjp 自动归约)
"""

from __future__ import annotations

import mlx.core as mx

mx.random.seed(0)

from tinyautograd import Tensor, eval, ops, value_and_grad, vjp
from tinyautograd.pytree import flatten


def _close(a: mx.array, b: mx.array) -> None:
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    assert mx.allclose(a, b).item(), f"{a} != {b}"


# ---- stack: 多进单出 ----


def test_stack_fwd() -> None:
    arrays = [mx.random.normal((3, 4), key=mx.random.key(i)) for i in range(3)]
    out = ops.stack([Tensor(a) for a in arrays])
    eval(out)
    _close(out._array, mx.stack(arrays))


def test_stack_vjp() -> None:
    arrays = [mx.random.normal((4,), key=mx.random.key(i)) for i in range(3)]
    ts = [Tensor(a) for a in arrays]
    c = mx.random.normal((3, 4))
    out, gs = vjp(lambda *xs: ops.stack(list(xs)), ts, [Tensor(c)])
    ref_out, ref_gs = mx.vjp(lambda *xs: mx.stack(list(xs)), arrays, [c])
    eval(out, *flatten(gs))
    _close(out._array, ref_out[0])
    for g, rg in zip(flatten(gs), ref_gs):
        _close(g._array, rg)


# ---- split: 单进多出 ----


def test_split_fwd() -> None:
    a = mx.random.normal((6,))
    parts = ops.split(Tensor(a), [2, 4])
    eval(*parts)
    for p, r in zip(parts, mx.split(a, [2, 4])):
        _close(p._array, r)


def test_split_grad() -> None:
    a = mx.random.normal((5,))

    def f_t(x: Tensor) -> Tensor:
        x0, x1, x2 = ops.split(x, [2, 4])
        return ops.sum(x0) + 2 * ops.sum(x1) + 3 * ops.sum(x2)

    def f_m(x: mx.array) -> mx.array:
        x0, x1, x2 = mx.split(x, [2, 4])
        return mx.sum(x0) + 2 * mx.sum(x1) + 3 * mx.sum(x2)

    loss_t, g_t = value_and_grad(f_t)(Tensor(a))
    loss_m, g_m = mx.value_and_grad(f_m)(a)
    eval(loss_t, g_t)
    _close(loss_t._array, loss_m)
    _close(g_t._array, g_m)


def test_split_partial_grad() -> None:
    # 部分输出被丢弃: 反传补零
    a = mx.random.normal((4,))

    def f_t(x: Tensor) -> Tensor:
        x0, _ = ops.split(x, [2])
        return ops.sum(x0)

    def f_m(x: mx.array) -> mx.array:
        x0, _ = mx.split(x, [2])
        return mx.sum(x0)

    loss_t, g_t = value_and_grad(f_t)(Tensor(a))
    loss_m, g_m = mx.value_and_grad(f_m)(a)
    eval(loss_t, g_t)
    _close(loss_t._array, loss_m)
    _close(g_t._array, g_m)


# ---- broadcast_to: 单进单出 ----


def test_broadcast_fwd() -> None:
    a = mx.random.normal((3,))
    out = ops.broadcast_to(Tensor(a), (2, 3))
    eval(out)
    _close(out._array, mx.broadcast_to(a, (2, 3)))


def test_broadcast_grad() -> None:
    a = mx.random.normal((3,))

    def f_t(x: Tensor) -> Tensor:
        return ops.sum(ops.broadcast_to(x, (2, 3)))

    def f_m(x: mx.array) -> mx.array:
        return mx.sum(mx.broadcast_to(x, (2, 3)))

    loss_t, g_t = value_and_grad(f_t)(Tensor(a))
    loss_m, g_m = mx.value_and_grad(f_m)(a)
    eval(loss_t, g_t)
    _close(loss_t._array, loss_m)
    _close(g_t._array, g_m)
