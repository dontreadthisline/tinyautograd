"""autograd:"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Generator

import mlx.core as mx

from tinyautograd import ops
from tinyautograd.pytree import PyTree, flatten, substitute
from tinyautograd.tensor import Tensor

__all__ = ["vjp", "value_and_grad", "grad", "eval", "no_grad", "detach"]


def _vjp(
    fn: Callable[..., Any], args: list[Any], cotans: Any
) -> list[Any]:
    """单算子局部反传: 外包 mx.vjp (cotans 单个或 list, 支持多输出)。"""
    if not isinstance(cotans, (list, tuple)):
        cotans = [cotans]
    return mx.vjp(fn, list(args), list(cotans))[1]


def _backward(
    out: Tensor | PyTree[Tensor],
    init_cotans: list[Any],
    leaves: list[Tensor],
) -> dict[int, Any]:
    """从 out 出发反向传播, 返回 {id(leaf): 梯度}。叶子无梯度时缺项。"""
    out_leaves = flatten(out)
    if len(out_leaves) != len(init_cotans):
        raise ValueError(
            f"[vjp] 输出 Tensor 数 ({len(out_leaves)}) "
            f"与余切数量 ({len(init_cotans)}) 不匹配。"
        )
    leaf_ids = {id(t) for t in leaves}

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

    for t in out_leaves:
        dfs(t)
    cotan_map: dict[int, Any] = {}
    for t, c in zip(out_leaves, init_cotans):
        cotan_map[id(t)] = c
    for t in reversed(order):
        if t.op is None:
            continue
        op = t.op
        if op.outputs:
            # 多输出: 每个输出独立贡献自己的 cotan, 其余补零
            c = cotan_map.get(id(t))
            if c is None:
                continue
            cotans = [mx.zeros_like(arr) for arr in op.outputs]
            cotans[t.index] = c
            vjps = _vjp(op.fn, list(op.args), cotans)
        else:
            # 单输出
            c = cotan_map.get(id(t))
            if c is None:
                continue
            vjps = _vjp(op.fn, list(op.args), c)
        for p, g in zip(t.inputs, vjps):
            if id(p) not in reach:  # 跳过不需要计算的节点
                continue
            if id(p) in cotan_map:  # 全微分路径的加和
                cotan_map[id(p)] = mx.add(cotan_map[id(p)], g)
            else:
                cotan_map[id(p)] = g  # 去更新输入节点的余切,供下一步使用
    return {i: cotan_map[i] for i in leaf_ids if i in cotan_map}


def vjp(
    fun: Callable[..., Tensor | PyTree[Tensor]],
    primals: Any,
    cotangents: Any,
) -> tuple[Tensor | Any, Any]:
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
        grads = _backward(out, [mx.ones_like(out._array)], targets)
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


def eval(*args: Any) -> None:
    """显式离开 DSL 算账 (惰性求值下的唯一显式门)。"""
    arrays = [t._array for a in args for t in flatten(a)]
    if arrays:
        mx.eval(*arrays)


@contextmanager
def no_grad() -> Generator[None, None, None]:
    """上下文: 算子不建边 (纯前向 / 推理)。"""
    prev = ops._no_grad
    ops._no_grad = True
    try:
        yield
    finally:
        ops._no_grad = prev


def detach(t: Tensor) -> Tensor:
    """同值、无父节点的新 Tensor (等价 stop_gradient)。"""
    return Tensor(t._array)
