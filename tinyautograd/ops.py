"""ops: 全部算子的门面 (静态定义, 绑定 mlx)。后端接触面收敛于此 (与 tensor),
数组层机制桥 (mx_*) 供变换层使用, 不进用户面。"""

from collections.abc import Sequence
from functools import reduce
from typing import Any, Callable

import mlx.core as mx

from tinyautograd import tensor
from tinyautograd.tensor import (
    Array,
    ArrayFn,
    MultiArrayFn,
    Tensor,
    TensorFn,
    TensorLike,
)

__all__ = [
    # 提升门 (扩展点)
    "lift",
    "lift_multi",
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
    "softmax",
    "matmul",
    "transpose",
    "sum",
    "mean",
    "max",
    "reshape",
    "stack",
    "split",
    "broadcast_to",
    # 源词汇: 叶子 (无输入边; 内部 _source 生成, 用户可直接 Tensor 构造)
    "zeros",
    "ones",
    "arange",
    "zeros_like",
    "ones_like",
    # 派生词
    "relu",
]


def lift(fn: ArrayFn) -> TensorFn:
    """把 mlx 算子提升为可导算子: 前向 = fn, 反向 = mx.vjp(fn, ...)。

    no_grad 上下文中不建边 (纯前向)。
    """

    def wrapped(*args: TensorLike) -> Tensor:
        ts = tuple(tensor._wrap(a) for a in args)
        if tensor._no_grad:
            arrays = tuple(t._ensure() for t in ts)
            return Tensor(fn(*arrays))
        node = fn(*(t._array for t in ts))
        ph = mx.zeros(node.shape, node.dtype)
        return Tensor(ph, inputs=ts, op=tensor.Op(fn), computed=False)

    wrapped.__name__ = getattr(fn, "__name__", "lifted")
    return wrapped


def lift_multi(fn: MultiArrayFn) -> Callable[..., tuple[Tensor, ...]]:
    """把"返回 tuple/list"的 mlx 函数提升为多输出可导算子。

    多个输出共享同一个 op (index 区分), 反传时每个输出独立贡献自己的
    cotan, 梯度累加到输入。
    """

    def wrapped(*args: TensorLike) -> tuple[Tensor, ...]:
        ts = tuple(tensor._wrap(a) for a in args)
        if tensor._no_grad:
            arrays = tuple(t._ensure() for t in ts)
            return tuple(Tensor(x) for x in fn(*arrays))
        nodes = tuple(fn(*(t._array for t in ts)))  # 临时, 借 shape
        placeholders = tuple(mx.zeros(n.shape, n.dtype) for n in nodes)
        op = tensor.Op(fn, placeholders)  # 多输出: outputs 即占位 (形状参照)
        return tuple(
            Tensor(ph, inputs=ts, op=op, index=i, computed=False)
            for i, ph in enumerate(placeholders)
        )

    wrapped.__name__ = getattr(fn, "__name__", "lifted_multi")
    return wrapped


def _source(fn: ArrayFn) -> TensorFn:
    """内部便利: 把 mx 构造函数包装为源词汇 (叶子, op=None)。"""

    def wrapped(*args: Any) -> Tensor:
        # 参数是透传给 mx 构造函数的 shape/dtype 等 (域由被包装函数定义), 不参与 wrap
        return tensor.tensor(fn(*args))

    wrapped.__name__ = getattr(fn, "__name__", "source")
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
softmax = lift(mx.softmax)
matmul = lift(mx.matmul)
transpose = lift(mx.transpose)


def _rsqrt(a: Array) -> Array:
    return mx.reciprocal(mx.sqrt(a))


rsqrt = lift(_rsqrt)


def sum(
    t: TensorLike, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """轴约简: sum(x) / sum(x, axis) / sum(x, axis, keepdims)。"""
    return lift(lambda a: mx.sum(a, axis=axis, keepdims=keepdims))(t)


def mean(
    t: TensorLike, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """轴平均: mean(x) / mean(x, axis) / mean(x, axis, keepdims)。"""
    return lift(lambda a: mx.mean(a, axis=axis, keepdims=keepdims))(t)


def max(
    t: TensorLike, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """轴最大: max(x) / max(x, axis) / max(x, axis, keepdims)。"""
    return lift(lambda a: mx.max(a, axis=axis, keepdims=keepdims))(t)


def reshape(t: TensorLike, shape: int | tuple[int, ...]) -> Tensor:
    """形状重排, shape 接受 int 或 tuple。"""
    return lift(lambda a: mx.reshape(a, (shape,) if isinstance(shape, int) else shape))(
        t
    )


def stack(tensors: Sequence[TensorLike], axis: int = 0) -> Tensor:
    """多 Tensor 沿新轴堆叠: stack([a, b], axis)。"""
    ts = tuple(tensors)
    return lift(lambda *arrs: mx.stack(list(arrs), axis=axis))(*ts)


def split(
    a: TensorLike, indices: int | Sequence[int], axis: int = 0
) -> tuple[Tensor, ...]:
    """单 Tensor 沿轴切多份 (lift_multi: 多输出共享 op)。"""
    # mlx stub 将模块级 mx.split 误标为 -> array (方法版为 list[array]), 实际返回 list
    return lift_multi(lambda arr: mx.split(arr, indices, axis=axis))(a)  # type: ignore[arg-type]


def broadcast_to(t: TensorLike, shape: int | tuple[int, ...]) -> Tensor:
    """广播到目标形状 (反向由 mx.vjp 自动归约)。"""
    return lift(
        lambda arr: mx.broadcast_to(arr, (shape,) if isinstance(shape, int) else shape)
    )(t)


zeros = _source(mx.zeros)
ones = _source(mx.ones)
arange = _source(mx.arange)


def zeros_like(t: Tensor, dtype: mx.Dtype | None = None) -> Tensor:
    return _source(mx.zeros)(t.shape, dtype or t.dtype)


def ones_like(t: Tensor, dtype: mx.Dtype | None = None) -> Tensor:
    return _source(mx.ones)(t.shape, dtype or t.dtype)


def relu(t: TensorLike) -> Tensor:
    return maximum(t, 0.0)
