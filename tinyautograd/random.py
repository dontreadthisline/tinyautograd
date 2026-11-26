"""random: 随机源 (与 ops 分离 —— 唯一带隐式状态的词汇)。

源词汇: 每次调用重新采样, 产出的 Tensor 是叶子 (op=None),
不参与求导 —— 随机量参与前向, 梯度绕它而行。
"""

import mlx.core as mx

from tinyautograd.ops import _source

from tinyautograd.tensor import TensorFn

__all__ = ["seed", "normal"]

seed = mx.random.seed
normal: TensorFn = _source(mx.random.normal)
