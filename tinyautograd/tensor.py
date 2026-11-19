"""Tensor: 值 = 图节点 = 线性映射载体 (值中心表示)。"""

from typing import Any, NamedTuple

import mlx.core as mx

__all__ = ["Tensor", "Op"]

type Array = mx.array


class Op(NamedTuple):
    fn: Any
    args: tuple[Array, ...]
    outputs: tuple[Array, ...] = ()


class Tensor:
    __slots__ = ("_array", "inputs", "op", "index")

    _array: Array
    inputs: tuple[Tensor, ...]
    op: Op | None
    index: int

    def __init__(
        self,
        value: Array | int | float | bool | list | tuple,
        inputs: tuple[Tensor, ...] = (),
        op: Op | None = None,
        index: int = 0,
    ) -> None:
        if not isinstance(value, mx.array):
            value = mx.array(value)
        self._array = value
        self.inputs = inputs
        self.op = op
        self.index = index

    # ---- 值的元数据 ----
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> Any:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return int(self._array.ndim)

    @property
    def size(self) -> int:
        return int(self._array.size)

    @property
    def T(self) -> Tensor:
        """数学记法 X^T 的直接对应 (可导, 等价 ops.transpose)."""
        from tinyautograd.ops import transpose

        return transpose(self)

    # ---- DSL 出口: 读取触发求值 ----
    def item(self) -> Any:
        return self._array.item()

    def tolist(self) -> Any:
        return self._array.tolist()

    def numpy(self) -> object:
        import numpy as np

        return np.asarray(self._array)  # mx.array 实现 __array__ 协议

    def __repr__(self) -> str:
        return f"Tensor({self._array!r})"

    def __len__(self) -> int:
        return len(self._array)

    # ---- magic methods: 中缀数学记法 (算子本体在 ops.*, 这里只做转发) ----
    def _bin(self, name: str, other: object, reflect: bool = False) -> Tensor:
        from tinyautograd import ops
        from tinyautograd.tensor import wrap

        fn = getattr(ops, name)
        a, b = wrap(other), self
        return fn(a, b) if reflect else fn(self, a)

    def __add__(self, other: object) -> Tensor:
        return self._bin("add", other)

    def __radd__(self, other: object) -> Tensor:
        return self._bin("add", other, reflect=True)

    def __sub__(self, other: object) -> Tensor:
        return self._bin("sub", other)

    def __rsub__(self, other: object) -> Tensor:
        return self._bin("sub", other, reflect=True)

    def __mul__(self, other: object) -> Tensor:
        return self._bin("mul", other)

    def __rmul__(self, other: object) -> Tensor:
        return self._bin("mul", other, reflect=True)

    def __truediv__(self, other: object) -> Tensor:
        return self._bin("div", other)

    def __rtruediv__(self, other: object) -> Tensor:
        return self._bin("div", other, reflect=True)

    def __matmul__(self, other: object) -> Tensor:
        return self._bin("matmul", other)

    def __pow__(self, other: object) -> Tensor:
        return self._bin("pow", other)

    def __neg__(self) -> Tensor:
        from tinyautograd import ops

        return ops.neg(self)

    def __getitem__(self, idx: Any) -> Tensor:
        from tinyautograd.ops import lift

        return lift(lambda a, _idx=idx: a[_idx])(self)


def wrap(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)
