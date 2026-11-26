"""Tensor: 值 = 图节点 = 线性映射载体 (值中心表示)。"""

from typing import Callable, NamedTuple

import mlx.core as mx

from tinyautograd import ops

__all__ = ["Tensor", "Op", "tensor"]

type Array = mx.array
type TensorLike = Tensor | Array | int | float | bool | list | tuple
type ArrayFn = Callable[..., Array]
type MultiArrayFn = Callable[..., list[Array] | tuple[Array, ...]]
type TensorFn = Callable[..., Tensor]

mx_vjp = mx.vjp
mx_eval = mx.eval
mx_zeros_like = mx.zeros_like
mx_ones_like = mx.ones_like

# 图构成开关: no_grad 上下文置位, lift 构造时读取
_no_grad = False


class Op(NamedTuple):
    fn: ArrayFn | MultiArrayFn
    outputs: tuple[Array, ...] = ()


class Tensor:
    __slots__ = ("_array", "inputs", "op", "index", "_computed")

    _array: Array
    inputs: tuple[Tensor, ...]
    op: Op | None
    index: int
    _computed: bool

    def __init__(
        self,
        value: Array | int | float | bool | list | tuple,
        inputs: tuple[Tensor, ...] = (),
        op: Op | None = None,
        index: int = 0,
        computed: bool = True,
    ) -> None:
        if not isinstance(value, mx.array):
            value = mx.array(value)
        self._array = value
        self.inputs = inputs
        self.op = op
        self.index = index
        self._computed = computed

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> mx.Dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return int(self._array.ndim)

    @property
    def size(self) -> int:
        return int(self._array.size)

    @property
    def T(self) -> Tensor:
        return ops.transpose(self)

    def _ensure(self) -> Array:
        """值读统一门:未求值则对该子图前向求值"""
        from tinyautograd.runtime import eval as _eval

        if not self._computed:
            _eval(self)
        return self._array

    def item(self) -> int | float | bool:
        return self._ensure().item()

    def tolist(self) -> mx.list_or_scalar:
        return self._ensure().tolist()

    def __repr__(self) -> str:
        if not self._computed:
            return f"Tensor(<lazy> shape={self.shape} dtype={self.dtype})"
        return f"Tensor({self._array!r})"

    def __len__(self) -> int:
        return len(self._array)

    def _bin(self, name: str, other: TensorLike, reflect: bool = False) -> Tensor:
        fn = getattr(ops, name)
        a, b = _wrap(other), self
        return fn(a, b) if reflect else fn(self, a)

    def __add__(self, other: TensorLike) -> Tensor:
        return self._bin("add", other)

    def __radd__(self, other: TensorLike) -> Tensor:
        return self._bin("add", other, reflect=True)

    def __sub__(self, other: TensorLike) -> Tensor:
        return self._bin("sub", other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        return self._bin("sub", other, reflect=True)

    def __mul__(self, other: TensorLike) -> Tensor:
        return self._bin("mul", other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        return self._bin("mul", other, reflect=True)

    def __truediv__(self, other: TensorLike) -> Tensor:
        return self._bin("div", other)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return self._bin("div", other, reflect=True)

    def __matmul__(self, other: TensorLike) -> Tensor:
        return self._bin("matmul", other)

    def __pow__(self, other: TensorLike) -> Tensor:
        return self._bin("pow", other)

    def __neg__(self) -> Tensor:
        return ops.neg(self)

    def __getitem__(
        self, idx: int | slice | tuple[int | slice | None, ...] | None
    ) -> Tensor:
        return ops.lift(lambda a, _idx=idx: a[_idx])(self)


def tensor(value: TensorLike) -> Tensor:
    """通用源构造器: 把标量/序列/mx 数组包装为叶子 Tensor。"""
    return _wrap(value)


def _wrap(value: TensorLike) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)
