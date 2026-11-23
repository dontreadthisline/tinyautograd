"""Tensor 树 (pytree) 工具: 摊平 / 同构替换。

树节点: Tensor | list | tuple | dict[str, ...]; 其他值为常量, 摊平时忽略。
类型层对应: PyTree[T] (T 为叶子域), 常量容忍体现在 PyTree[object]。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tinyautograd.tensor import Tensor

__all__ = ["PyTree", "flatten", "substitute"]


type PyTree[T] = T | list[PyTree[T]] | tuple[PyTree[T], ...] | dict[str, PyTree[T]]


def flatten(tree: PyTree[object]) -> list[Tensor]:
    from tinyautograd.tensor import Tensor

    out: list[Tensor] = []
    stack: list[PyTree[object]] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, Tensor):
            out.append(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(reversed(list(node)))
    return out


def substitute[T](template: T, mapping: dict[int, Tensor]) -> T:
    """按模板结构重建树, Tensor 叶被 mapping[id(t)] 替换 (缺省原样保留)。

    递归同构变换超出当前类型系统的表达力 (T 只能保持整树, 无法逐叶精化),
    分支处的 cast 是标准解。
    """
    from tinyautograd.tensor import Tensor

    if isinstance(template, Tensor):
        return cast("T", mapping.get(id(template), template))
    if isinstance(template, dict):
        return cast("T", {k: substitute(v, mapping) for k, v in template.items()})
    if isinstance(template, tuple):
        return cast("T", tuple(substitute(v, mapping) for v in template))
    if isinstance(template, list):
        return cast("T", [substitute(v, mapping) for v in template])
    return template
