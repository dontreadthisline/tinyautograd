"""Tensor 树 (pytree) 工具: 摊平 / 同构替换。

树节点: Tensor | list | tuple | dict[str, ...]; 其他值为常量, 摊平时忽略。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyautograd.tensor import Tensor

__all__ = ["PyTree", "flatten", "substitute"]


type PyTree[T] = T | list[PyTree[T]] | tuple[PyTree[T], ...] | dict[str, PyTree[T]]


def flatten(tree: object) -> list[Tensor]:
    from tinyautograd.tensor import Tensor

    out: list[Tensor] = []
    stack: list[object] = [tree]
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
    """按模板结构重建树, Tensor 叶被 mapping[id(t)] 替换 (缺省原样保留)。"""
    from tinyautograd.tensor import Tensor

    if isinstance(template, Tensor):
        return mapping.get(id(template), template)  # type: ignore[return-value]
    if isinstance(template, dict):
        return {k: substitute(v, mapping) for k, v in template.items()}  # type: ignore[return-value]
    if isinstance(template, tuple):
        return tuple(substitute(v, mapping) for v in template)  # type: ignore[return-value]
    if isinstance(template, list):
        return [substitute(v, mapping) for v in template]  # type: ignore[return-value]
    return template
