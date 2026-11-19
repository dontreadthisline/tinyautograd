"""random: 随机源 (与 ops 分离 —— 唯一带隐式状态的词汇)。

const 编码: 每次调用重新采样, 产出的 Tensor 是叶子 (op=None),
不参与求导 —— 随机量参与前向, 梯度绕它而行。
"""

from __future__ import annotations

import mlx.core as mx

from tinyautograd.ops import const

__all__ = ["seed", "normal", "uniform"]

seed = mx.random.seed
normal = const(mx.random.normal)
uniform = const(mx.random.uniform)
