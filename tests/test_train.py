"""端到端: 两层 MLP + MSE 训练收敛 (训练循环属 DSL 外, 手写 SGD)。"""

from __future__ import annotations

import mlx.core as mx

from tinyautograd import Tensor, eval, ops, random, value_and_grad


def test_mlp_converges() -> None:
    mx.random.seed(0)
    params: dict[str, Tensor] = {
        "W1": random.normal((6, 8)) * (6**-0.5),
        "b1": ops.zeros((8,)),
        "W2": random.normal((8, 3)) * (8**-0.5),
        "b2": ops.zeros((3,)),
    }
    x = random.normal((32, 6))
    w_true = mx.random.normal((6, 3))
    y = Tensor(mx.maximum(x._array, 0.0) @ w_true)

    def loss_fn(p: dict[str, Tensor], xx: Tensor, yy: Tensor) -> Tensor:
        h = ops.relu(xx @ p["W1"] + p["b1"])
        o = h @ p["W2"] + p["b2"]
        return ops.mean((o - yy) ** 2)

    loss_and_grad = value_and_grad(loss_fn)
    lr = 0.25
    losses: list[float] = []
    for _ in range(600):
        loss, grads = loss_and_grad(params, x, y)
        params = {k: v - lr * grads[k] for k, v in params.items()}
        eval(loss, params)
        losses.append(loss.item())
    assert losses[-1] < 0.05 * losses[0], f"未收敛: {losses[0]} -> {losses[-1]}"
    assert losses[-1] < 0.02
