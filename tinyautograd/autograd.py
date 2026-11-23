"""autograd: 反向引擎 —— 拓扑剪枝、单步 vjp、余切接力。

与 runtime (前向引擎 + 门) 对等; 详见各模块 docstring。
"""

from __future__ import annotations

from typing import Any, Callable

from tinyautograd.ops import (
    mx_add,
    mx_ones_like,
    mx_vjp,
    mx_zeros_like,
)
from tinyautograd.pytree import PyTree, flatten, substitute
from tinyautograd.runtime import eval
from tinyautograd.tensor import Array, ArrayFn, MultiArrayFn, Tensor

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
        eval(*out_leaves)  # VJP 在前向点取值: 反向需要前向值, 隐式前向
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
                cotan = g if prev is None else mx_add(g, prev)
                cotans_map[id(p)] = cotan

    return {idx: cotans_map[idx] for idx in leaf_ids if idx in cotans_map}


def vjp(
    fun: Callable[..., Tensor | PyTree[Tensor]],
    primals: Tensor | PyTree[Tensor],
    cotangents: Tensor | PyTree[Tensor],
) -> tuple[Tensor | PyTree[Tensor], PyTree[Tensor]]:
    """vector-Jacobian product (原子层)。

    primals 作为 fun 的位置参数传入 (单个或元组), 全部 Tensor 叶均为求导目标;
    cotangents 与 fun 输出同构。返回 (输出, 与 primals 同构的梯度树)。
    """
    args = tuple(primals) if isinstance(primals, (list, tuple)) else (primals,)
    leaves = flatten(args)
    if not leaves:
        raise ValueError("[vjp] primals 中没有 Tensor。")
    cotan_leaves = [t._array for t in flatten([cotangents])]
    out = fun(*args)
    grads = _backward(out, cotan_leaves, leaves)
    grad_tree = substitute(args, {i: Tensor(g) for i, g in grads.items()})
    if isinstance(primals, Tensor):
        return out, grad_tree[0]
    return out, grad_tree


def value_and_grad[**P](
    fun: Callable[P, Tensor],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[P, tuple[Tensor, Any]]:
    """把 fun 提升为返回 (值, 梯度树) 的函数。fun 须返回标量 Tensor。
    currying 约定: 被求导参数 (默认第 0 位) 应为 Tensor 或 Tensor 树。
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[Tensor, Any]:
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        targets = flatten([args[i] for i in nums])
        if not targets:
            raise ValueError("[value_and_grad] argnums 未选中任何 Tensor 参数。")
        out = fun(*args, **kwargs)
        if not isinstance(out, Tensor) or out.size != 1:
            shape = getattr(out, "shape", type(out).__name__)
            raise ValueError(f"[value_and_grad] fun 须返回标量 Tensor, 得到: {shape}")
        grads = _backward(out, [mx_ones_like(out._array)], targets)
        mapping = {i: Tensor(g) for i, g in grads.items()}
        trees = tuple(substitute(args[n], mapping) for n in nums)
        grad_tree = trees[0] if len(trees) == 1 else trees
        return out, grad_tree

    return wrapped


def grad[**P](
    fun: Callable[P, Tensor],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[P, Any]:
    """value_and_grad 的丢值版。"""
    vg = value_and_grad(fun, argnums)

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
        return vg(*args, **kwargs)[1]

    return wrapped
