"""autograd: 反向引擎 —— 拓扑剪枝、单步 vjp、余切接力。

与 runtime (前向引擎 + 门) 对等; 详见各模块 docstring。
"""

from collections.abc import Sequence
from typing import Any, Callable

from tinyautograd.pytree import PyTree, flatten, substitute
from tinyautograd.runtime import eval
from tinyautograd.tensor import (
    Array,
    ArrayFn,
    MultiArrayFn,
    Tensor,
    mx_vjp,
    mx_zeros_like,
)

__all__ = ["vjp", "value_and_grad", "grad"]


def _single_step_vjp(
    fn: ArrayFn | MultiArrayFn,
    args: list[Array],
    cotans: Array | list[Array],
) -> list[Array]:
    """单算子局部反传: 外包 mx_vjp (cotans 单个或 list, 支持多输出)。"""
    if not isinstance(cotans, (list, tuple)):
        cotans = [cotans]
    return mx_vjp(fn, args, cotans)[1]


def _topo_pruned(
    roots: list[Tensor], leaf_ids: set[int]
) -> tuple[list[Tensor], set[int]]:
    """反向用: 后序遍历, 只收可达目标叶的节点 (剪枝), 返回 (order, reach)。"""
    order: list[Tensor] = []
    visited: set[int] = set()
    reach: set[int] = set()

    def dfs(t: Tensor) -> bool:
        key = id(t)
        if key in visited:
            return key in reach
        visited.add(key)
        r = key in leaf_ids
        for p in t.inputs:
            if dfs(p):
                r = True
        if r:
            reach.add(key)
            order.append(t)
        return r

    for t in roots:
        dfs(t)
    return order, reach


def _backward(
    out: Tensor | PyTree[Tensor],
    init_cotans: list[Array],
    leaves: list[Tensor],
) -> dict[int, Array]:
    """从 out 出发反向传播, 返回 {id(leaf): 梯度}。叶子无梯度时缺项。"""
    out_leaves = flatten(out)
    if len(out_leaves) != len(init_cotans):
        raise ValueError(
            f"[vjp] 输出 Tensor 数 ({len(out_leaves)}) "
            f"与余切数量 ({len(init_cotans)}) 不匹配。"
        )
    leaf_ids = {id(t) for t in leaves}
    if any(not t._computed for t in out_leaves):
        eval(*out_leaves)
    order, reach = _topo_pruned(out_leaves, leaf_ids)
    cotans_map: dict[int, Array] = {
        id(t): cotan for t, cotan in zip(out_leaves, init_cotans)
    }
    for t in reversed(order):
        c = cotans_map.get(id(t))
        if t.op is None or c is None:
            continue
        outputs = t.op.outputs or (t._array,)
        cotans = [
            c if i == t.index else mx_zeros_like(a) for i, a in enumerate(outputs)
        ]
        vjps = _single_step_vjp(t.op.fn, [p._array for p in t.inputs], cotans)
        for p, g in zip(t.inputs, vjps):
            if id(p) in reach:
                prev = cotans_map.get(id(p))
                cotan = g if prev is None else g + prev
                cotans_map[id(p)] = cotan

    return {idx: cotans_map[idx] for idx in leaf_ids if idx in cotans_map}


def vjp(
    fun: Callable[..., Tensor | PyTree[Tensor]],
    primals: Tensor | PyTree[Tensor],
    cotangents: Tensor | PyTree[Tensor],
    **kwargs: Any,
) -> tuple[Tensor | PyTree[Tensor], PyTree[Tensor]]:
    """vector-Jacobian product (原子层, 求值语义: 带点带方向, 一次性)。"""
    args = primals if isinstance(primals, (list, tuple)) else (primals,)
    leaves = flatten(args)
    if not leaves:
        raise ValueError("[vjp] 目标参数中没有 Tensor")
    cotan_leaves = [t._array for t in flatten([cotangents])]
    out = fun(*args, **kwargs)
    grads = _backward(out, cotan_leaves, leaves)
    grad_tree = substitute(args, {i: Tensor(g) for i, g in grads.items()})
    if isinstance(primals, Tensor):
        return out, grad_tree[0]
    return out, grad_tree


def value_and_grad[**P](
    fun: Callable[P, Tensor],
    argnums: int | Sequence[int] = 0,
    argnames: str | Sequence[str] | None = None,
) -> Callable[P, tuple[Tensor, Any]]:
    """把 fun 提升为返回 (值, 梯度树) 的函数。fun 须返回标量 Tensor。

    目标经 vjp 的 argnums 贯穿到剪枝种子。
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[Tensor, Any]:
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        names = (
            ()
            if argnames is None
            else (argnames,)
            if isinstance(argnames, str)
            else tuple(argnames)
        )
        args1 = [arg for i in nums for arg in flatten(args[i])]
        args2 = [
            arg for name in names if name in kwargs for arg in flatten(kwargs[name])
        ]
        out = fun(*args, **kwargs)
        if not isinstance(out, Tensor) or out.size != 1:
            raise ValueError("输出是非标量Tensor")
        grads = _backward(
            out,
            [Tensor(1.0)._array],
            args1 + args2,
        )
        mapping = {i: Tensor(grad) for i, grad in grads.items()}
        grads = tuple(substitute(args[i], mapping) for i in nums) + tuple(
            substitute(kwargs[name], mapping) for name in names
        )
        return out, grads[0] if len(grads) == 1 else grads

    return wrapped


def grad[**P](
    fun: Callable[P, Tensor],
    argnums: int | Sequence[int] = 0,
    argnames: str | Sequence[str] | None = None,
) -> Callable[P, Any]:
    """value_and_grad 的丢值版。"""
    vg = value_and_grad(fun, argnums, argnames)

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
        return vg(*args, **kwargs)[1]

    return wrapped
