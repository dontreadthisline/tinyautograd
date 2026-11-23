"""runtime: 运行时 —— 前向求值引擎 + 进出 DSL 的门。

与 autograd (反向引擎) 对等:
  前向: _topo_all + eval             (拓扑序逐节点重放 op.fn, 单步求值)
  反向: _topo_pruned + _backward     (autograd 模块)
mlx 在此退化为无状态单步服务 (fn + mx_eval), 图与调度完全归属本框架。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from tinyautograd import ops
from tinyautograd.ops import mx_eval
from tinyautograd.pytree import PyTree, flatten
from tinyautograd.tensor import Array, Tensor

if TYPE_CHECKING:
    from tinyautograd.tensor import Op

__all__ = ["eval", "no_grad", "detach"]


def _topo_all(roots: list[Tensor]) -> list[Tensor]:
    """前向用: 后序遍历全部祖先 (无剪枝), 叶先根后 —— 前向拓扑序。"""
    order: list[Tensor] = []
    visited: set[int] = set()

    def dfs(t: Tensor) -> None:
        key = id(t)
        if key in visited:
            return
        visited.add(key)
        for p in t.inputs:
            dfs(p)
        order.append(t)

    for t in roots:
        dfs(t)
    return order


def eval(*args: PyTree[Tensor]) -> None:
    """显式离开 DSL 算账 (显式求值门)。

    前向调度由本图承担: 按拓扑序逐节点重放 op.fn 并单步求值,
    mlx 图深度恒为 1; 占位数组被真值替换。
    """
    order = _topo_all(flatten(args))
    replayed: dict[int, tuple[Array, ...]] = {}  # id(op) -> 重放结果 (多输出去重)
    for t in order:
        if t._computed:
            continue
        if t.op is None:
            mx_eval(t._array)
            t._computed = True
            continue
        outs = replayed.get(id(t.op))
        if outs is None:
            arrays = tuple(p._array for p in t.inputs)
            ret = t.op.fn(*arrays)
            outs = tuple(ret) if isinstance(ret, (list, tuple)) else (ret,)
            mx_eval(*outs)
            replayed[id(t.op)] = outs
        t._array = outs[t.index]
        t._computed = True


@contextmanager
def no_grad() -> Generator[None, None, None]:
    """上下文: 算子不建边 (纯前向 / 推理)。不建边就没有延迟的本钱 —— eager 执行。"""
    prev = ops._no_grad
    ops._no_grad = True
    try:
        yield
    finally:
        ops._no_grad = prev


def detach(t: Tensor) -> Tensor:
    """同值、无父节点的新 Tensor (等价 stop_gradient)。"""
    return Tensor(t._ensure())
