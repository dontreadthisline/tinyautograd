"""runtime: 运行时 —— 前向求值引擎 + 进出 DSL 的门。

与 autograd (反向引擎) 对等:
  前向: _topo_all + eval             (拓扑序逐节点重放 op.fn, 单步求值)
  反向: _topo_pruned + _backward     (autograd 模块)
mlx 在此退化为无状态单步服务 (fn + mx_eval), 图与调度完全归属本框架。
"""

from contextlib import contextmanager
from typing import Generator

from tinyautograd import tensor
from tinyautograd.pytree import PyTree, flatten
from tinyautograd.tensor import Array, Tensor, mx_eval

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
    """
    前向调度: 按拓扑序逐节点重放 op.fn 并单步求值;
    假占位数组被真值替换, _computed 标记幂等。
    """
    order = _topo_all(flatten(args))
    for t in order:
        if t._computed:
            continue
        if t.op is None:
            mx_eval(t._array)
        else:
            ret = t.op.fn(*(p._array for p in t.inputs))
            outs = ret if isinstance(ret, (list, tuple)) else (ret,)
            mx_eval(*outs)
            t._array = outs[t.index]
        t._computed = True


@contextmanager
def no_grad() -> Generator[None, None, None]:
    """上下文: 算子不建边 (纯前向 / 推理)。不建边就没有延迟 —— eager 执行。"""
    prev = tensor._no_grad
    tensor._no_grad = True
    try:
        yield
    finally:
        tensor._no_grad = prev


def detach(t: Tensor) -> Tensor:
    """同值、无父节点的新 Tensor (等价 stop_gradient)。"""
    return Tensor(t._ensure())
