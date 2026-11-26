"""Tensor 树 (pytree) 工具: 摊平 / 同构替换。

树节点: Tensor | list | tuple | dict[str, ...]; 其他值为常量, 摊平时忽略。
类型层对应: PyTree[T] (T 为叶子域), 常量容忍体现在 PyTree[object]。
"""

from typing import cast

from tinyautograd.tensor import Tensor

__all__ = ["PyTree", "flatten", "substitute"]


type PyTree[T] = T | list[PyTree[T]] | tuple[PyTree[T], ...] | dict[str, PyTree[T]]


def flatten(tree: PyTree[object]) -> list[Tensor]:
    out = []
    stack = [tree]
    while stack:
        top = stack.pop()
        if isinstance(top, Tensor):
            out.append(cast("Tensor", top))
        elif isinstance(top, dict):
            stack.extend(top.values())
        elif isinstance(top, (tuple, list)):
            stack.extend(reversed(top))
    return out


def substitute[T](template: T, mapping: dict[int, Tensor]) -> T:
    """按模板结构重建树, Tensor 叶被 mapping[id(t)] 替换"""
    if isinstance(template, Tensor):
        return cast("T", mapping.get(id(template)))
    elif isinstance(template, dict):
        return cast("T", {k: substitute(v, mapping) for k, v in template.items()})
    elif isinstance(template, tuple):
        return cast("T", tuple(substitute(v, mapping) for v in template))
    elif isinstance(template, list):
        return cast("T", list([substitute(v, mapping) for v in template]))
    return template
