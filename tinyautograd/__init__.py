"""tinyautograd: MLX 之上的极简函数式 autograd 框架。"""

from tinyautograd import ops, random
from tinyautograd.autograd import detach, eval, grad, no_grad, value_and_grad, vjp
from tinyautograd.ops import const, lift, lift_multi
from tinyautograd.tensor import Op, Tensor

__all__ = [
    "Tensor",
    "Op",
    "ops",
    "random",
    "vjp",
    "value_and_grad",
    "grad",
    "eval",
    "no_grad",
    "detach",
    "lift",
    "lift_multi",
    "const",
]

__version__ = "0.2.0"
