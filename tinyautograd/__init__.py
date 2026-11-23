"""tinyautograd: MLX 之上的极简函数式 autograd 框架。"""

from tinyautograd import ops, random
from tinyautograd.autograd import grad, value_and_grad, vjp
from tinyautograd.ops import lift, lift_multi
from tinyautograd.runtime import detach, eval, no_grad
from tinyautograd.tensor import Tensor, tensor

__all__ = [
    "Tensor",
    "tensor",
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
]

__version__ = "0.2.0"
