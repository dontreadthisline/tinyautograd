"""反向对拍: tinyautograd.vjp/grad vs mx.vjp / mx.value_and_grad (天然 oracle)。"""

from __future__ import annotations

import mlx.core as mx

mx.random.seed(0)

from tinyautograd import Tensor, detach, eval, grad, no_grad, ops, value_and_grad, vjp
from typing import Any

from tinyautograd.pytree import flatten


def _close(a: mx.array, b: mx.array) -> None:
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    assert mx.allclose(a, b).item(), f"{a} != {b}"


def test_vjp_matches_array_vjp() -> None:
    cases: list[tuple[str, Any, int]] = [
        # (算子名, mx 对应函数, 参数个数)
        ("add", mx.add, 2), ("sub", mx.subtract, 2), ("mul", mx.multiply, 2),
        ("div", mx.divide, 2), ("matmul", mx.matmul, 2),
        ("exp", mx.exp, 1), ("square", mx.square, 1), ("tanh", mx.tanh, 1),
    ]
    for name, fn_m, n in cases:
        shapes = [(3, 4), (4, 5)] if name == "matmul" else [(3, 4)] * n
        arrays = [mx.random.normal(s) for s in shapes]
        c = mx.random.normal(arrays[0].shape if name != "matmul" else (3, 5))
        out_shape = (3, 5) if name == "matmul" else (3, 4)
        tensors = [Tensor(a) for a in arrays]
        out, gs = vjp(getattr(ops, name), tensors, [Tensor(c)])
        ref_out, ref_gs = mx.vjp(fn_m, arrays, [c])
        eval(out, gs)
        _close(out._array, ref_out[0])
        assert out._array.shape == out_shape
        for g, rg in zip(flatten(gs), ref_gs):
            _close(g._array, rg)


def test_grad_matches_array() -> None:
    a = mx.random.normal((3, 4))

    def f_t(x: Tensor) -> Tensor:
        return ops.mean(ops.square(x) + ops.exp(x) * x)

    def f_m(x: mx.array) -> mx.array:
        return mx.mean(mx.square(x) + mx.exp(x) * x)

    _, g = value_and_grad(f_t)(Tensor(a))
    _, g_m = mx.value_and_grad(f_m)(a)
    eval(g)
    _close(g._array, g_m)
    assert mx.allclose(g._array, g_m).item()


def test_multi_path_accumulation() -> None:
    # x 同时出现在两条路径: (x*x).sum() + (x*3).sum() —— 梯度应相加
    a = mx.random.normal((4,))

    def f_t(x: Tensor) -> Tensor:
        return ops.sum(x * x) + ops.sum(x * 3.0)

    def f_m(x: mx.array) -> mx.array:
        return mx.sum(x * x) + mx.sum(x * 3.0)

    g = grad(f_t)(Tensor(a))
    _, g_m = mx.value_and_grad(f_m)(a)
    eval(g)
    _close(g._array, g_m)


def test_broadcast_gradient_reduced() -> None:
    # b (4,) 广播加到 x (3,4): 对 b 的梯度应归约回 (4,)
    x = mx.random.normal((3, 4))
    b = mx.random.normal((4,))

    def f_t(params: Tensor, xx: Tensor) -> Tensor:
        return ops.sum((xx + params) ** 2)

    def f_m(p: mx.array, xx: mx.array) -> mx.array:
        return mx.sum((xx + p) ** 2)

    _, g = value_and_grad(f_t)(Tensor(b), Tensor(x))
    _, g_m = mx.value_and_grad(f_m)(b, x)
    eval(g)
    assert g.shape == (4,)
    _close(g._array, g_m)


def test_tree_params() -> None:
    x = mx.random.normal((8, 4))
    params = {
        "W1": Tensor(mx.random.normal((4, 6)) * 0.5),
        "b1": Tensor(mx.zeros((6,))),
        "W2": Tensor(mx.random.normal((6, 2)) * 0.5),
        "b2": Tensor(mx.zeros((2,))),
    }

    def loss_t(p: dict[str, Tensor], xx: Tensor) -> Tensor:
        h = ops.relu(xx @ p["W1"] + p["b1"])
        o = h @ p["W2"] + p["b2"]
        return ops.mean(o * o)

    def loss_m(p: dict[str, mx.array], xx: mx.array) -> mx.array:
        h = mx.maximum(xx @ p["W1"] + p["b1"], 0.0)
        o = h @ p["W2"] + p["b2"]
        return mx.mean(o * o)

    _, g = value_and_grad(loss_t)(params, Tensor(x))
    _, g_m = mx.value_and_grad(loss_m)(
        {k: v._array for k, v in params.items()}, x
    )
    assert isinstance(g, dict) and set(g) == {"W1", "b1", "W2", "b2"}
    eval(g)
    for k in params:
        _close(g[k]._array, g_m[k])


def test_multi_output_vjp() -> None:
    a = mx.random.normal((3, 2))

    def f_t(v: Tensor) -> tuple[Tensor, Tensor]:
        return v * v, ops.exp(v)

    def f_m(v: mx.array) -> tuple[mx.array, mx.array]:
        return v * v, mx.exp(v)

    out, g = vjp(f_t, Tensor(a), [Tensor(mx.ones((3, 2))), Tensor(mx.zeros((3, 2)))])
    ref_out, ref_g = mx.vjp(f_m, [a], [mx.ones((3, 2)), mx.zeros((3, 2))])
    eval(g)
    _close(g._array, ref_g[0])  # 余切 [1,0]: 梯度 = 2v (来自 v*v 分支)


def test_getitem_grad() -> None:
    a = mx.random.normal((4, 5))

    def f_t(x: Tensor) -> Tensor:
        return ops.sum(x[1:3, ::2] ** 2)

    def f_m(x: mx.array) -> mx.array:
        return mx.sum(x[1:3, ::2] ** 2)

    g = grad(f_t)(Tensor(a))
    _, g_m = mx.value_and_grad(f_m)(a)
    eval(g)
    _close(g._array, g_m)


def test_no_grad_and_detach() -> None:
    a = Tensor(mx.random.normal((3,)))
    with no_grad():
        t = ops.exp(a)
    assert t.inputs == () and t.op is None  # 不建边
    g = grad(lambda x: ops.sum(ops.exp(detach(x)) * x))(a)
    eval(g)
    # detach 部分断链: d/dx [exp(detach(x)) * x] = exp(detach(x))
    _close(g._array, mx.exp(a._array))


def test_reach_pruning() -> None:
    # 常量分支 (2.0) 的子图不应被回放: 梯度只来自 x 路径
    x = Tensor(mx.random.normal((3,)))
    y_leaf = ops.exp(Tensor(mx.array([1.0, 2.0, 3.0])))  # 不依赖 x
    g = grad(lambda t: ops.sum(t * y_leaf))(x)
    eval(g)
    _close(g._array, mx.exp(mx.array([1.0, 2.0, 3.0])))
