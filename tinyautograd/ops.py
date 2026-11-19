"""ops: 全部算子的门面 (静态定义, 绑定 mlx)。

词汇三层:
  lift 系  — 可导算子: 有输入边, 反向外包 mx.vjp
  const 系 — 常量源: 无输入边, 调用时刻定值 (随机语义由此编码)
  派生词   — 由上述两类组合出的便利词 (relu)

两个提升门 (函子对偶) 同时是用户的算子扩展点:
  lift(fn)       — 新的可导算子 (有输入边)
  lift_multi(fn) — 新的多输出可导算子 (输出共享 op)
  const(fn)      — 新的常量/随机源 (叶子, 不参与求导)
"""

from __future__ import annotations

from functools import reduce
from typing import Any, Callable

import mlx.core as mx

__all__ = [
    # 提升门 (扩展点)
    "lift",
    "lift_multi",
    "const",
    "compose",
    # lift 系: 可导算子
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "square",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "matmul",
    "transpose",
    "sum",
    "mean",
    "max",
    "reshape",
    "stack",
    "split",
    "broadcast_to",
    # const 系: 常量源
    "zeros",
    "ones",
    "full",
    "arange",
    "zeros_like",
    "ones_like",
    # 派生词
    "relu",
]

ArrayFn = Callable[..., Any]
TensorFn = Callable[..., Any]

_no_grad = False


def lift(fn: ArrayFn) -> TensorFn:
    """把 mlx 算子提升为可导算子: 前向 = fn, 反向 = mx.vjp(fn, ...)。

    no_grad 上下文中不建边 (纯前向)。
    """

    def wrapped(*args: Any) -> Any:
        from tinyautograd.tensor import Op, Tensor, wrap

        ts = tuple(wrap(a) for a in args)
        arrays = tuple(a._array for a in ts)
        if _no_grad:
            return Tensor(fn(*arrays))
        return Tensor(fn(*arrays), inputs=ts, op=Op(fn, arrays))

    wrapped.__name__ = getattr(fn, "__name__", "lifted")
    return wrapped


def lift_multi(fn: ArrayFn) -> Callable[..., tuple[Any, ...]]:
    """把"返回 tuple/list"的 mlx 函数提升为多输出可导算子。

    多个输出共享同一个 op (index 区分), 反传时每个输出独立贡献自己的
    cotan, 梯度累加到输入。
    """

    def wrapped(*args: Any) -> tuple[Any, ...]:
        from tinyautograd.tensor import Op, Tensor, wrap

        ts = tuple(wrap(a) for a in args)
        arrays = tuple(a._array for a in ts)
        if _no_grad:
            return tuple(Tensor(x) for x in fn(*arrays))
        outs = tuple(fn(*arrays))
        op = Op(fn, arrays, outs)
        return tuple(
            Tensor(x, inputs=ts, op=op, index=i) for i, x in enumerate(outs)
        )

    wrapped.__name__ = getattr(fn, "__name__", "lifted_multi")
    return wrapped


def const(fn: ArrayFn) -> TensorFn:
    """常量源: 无输入边, 产出调用时刻定值的叶子 (op=None)。"""

    def wrapped(*args: Any) -> Any:
        from tinyautograd.tensor import Tensor

        return Tensor(fn(*args))

    wrapped.__name__ = getattr(fn, "__name__", "const")
    return wrapped


def compose(*fns: Callable[..., Any]) -> Callable[..., Any]:
    """函数组合子: compose(f, g)(x) = f(g(x)), 即数学上的 f∘g。"""
    return reduce(lambda f, g: lambda *a, **k: f(g(*a, **k)), fns)


# ---------------------------------------------------------------------------
# lift 系: 一元/二元可导算子 (签名与 mx 对齐, 标量自动 wrap)
# ---------------------------------------------------------------------------
add = lift(mx.add)
sub = lift(mx.subtract)
mul = lift(mx.multiply)
div = lift(mx.divide)
pow = lift(mx.power)
maximum = lift(mx.maximum)
minimum = lift(mx.minimum)
neg = lift(mx.negative)
abs = lift(mx.abs)
exp = lift(mx.exp)
log = lift(mx.log)
sqrt = lift(mx.sqrt)
square = lift(mx.square)
sin = lift(mx.sin)
cos = lift(mx.cos)
tanh = lift(mx.tanh)
sigmoid = lift(mx.sigmoid)
matmul = lift(mx.matmul)
transpose = lift(mx.transpose)


def _rsqrt(a: Any) -> Any:
    return mx.reciprocal(mx.sqrt(a))


rsqrt = lift(_rsqrt)


# ---------------------------------------------------------------------------
# lift 系: 带元数据参数的算子 (闭包固定元数据后提升, op.args 可完整重放)
# ---------------------------------------------------------------------------
def sum(t: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """轴约简: sum(x) / sum(x, axis) / sum(x, axis, keepdims)。"""
    return lift(lambda a: mx.sum(a, axis=axis, keepdims=keepdims))(t)


def mean(t: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """轴平均: mean(x) / mean(x, axis) / mean(x, axis, keepdims)。"""
    return lift(lambda a: mx.mean(a, axis=axis, keepdims=keepdims))(t)


def max(t: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """轴最大: max(x) / max(x, axis) / max(x, axis, keepdims)。"""
    return lift(lambda a: mx.max(a, axis=axis, keepdims=keepdims))(t)


def reshape(t: Any, shape: Any) -> Any:
    """形状重排, shape 接受 int 或 tuple。"""
    return lift(
        lambda a: mx.reshape(a, (shape,) if isinstance(shape, int) else shape)
    )(t)


def stack(tensors: Any, axis: int = 0) -> Any:
    """多 Tensor 沿新轴堆叠: stack([a, b], axis)。"""
    ts = tuple(tensors)
    return lift(lambda *arrs: mx.stack(list(arrs), axis=axis))(*ts)


def split(a: Any, indices: Any, axis: int = 0) -> tuple[Any, ...]:
    """单 Tensor 沿轴切多份 (lift_multi: 多输出共享 op)。"""
    return lift_multi(lambda arr: mx.split(arr, indices, axis=axis))(a)


def broadcast_to(t: Any, shape: Any) -> Any:
    """广播到目标形状 (反向由 mx.vjp 自动归约)。"""
    return lift(lambda arr: mx.broadcast_to(arr, shape))(t)


# ---------------------------------------------------------------------------
# const 系: 常量源 (无输入边, 叶子)
# ---------------------------------------------------------------------------
zeros = const(mx.zeros)
ones = const(mx.ones)
full = const(mx.full)
arange = const(mx.arange)


def zeros_like(t: Any, dtype: Any = None) -> Any:
    return const(mx.zeros)(t.shape, dtype or t.dtype)


def ones_like(t: Any, dtype: Any = None) -> Any:
    return const(mx.ones)(t.shape, dtype or t.dtype)


# ---------------------------------------------------------------------------
# 派生词: 由基本词汇组合的便利词
# ---------------------------------------------------------------------------
def relu(t: Any) -> Any:
    return maximum(t, 0.0)
