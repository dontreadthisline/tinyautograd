"""两层 MLP + MSE 训练 —— 用户代码零 mx 依赖。

训练循环属于 DSL 之外: 框架只提供值的语言 (Tensor/ops)、函数的算子
(value_and_grad)、边界原语 (eval)。更新与调度是用户的普通 Python。

运行: uv run python examples/demo.py
"""

from __future__ import annotations

from tinyautograd import Tensor, eval, ops, random, value_and_grad, vjp


def demo1():
    random.seed(0)
    params: dict[str, Tensor] = {
        "W1": random.normal((6, 4)) * (6**-0.5),
        "b1": ops.zeros((4,)),
        "W2": random.normal((4, 3)) * (32**-0.5),
        "b2": ops.zeros((3,)),
    }
    x = random.normal((64, 6))
    w_true = random.normal((6, 3))  # 教师参数: 生成携带可恢复结构的目标
    y = ops.relu(x) @ w_true  # y = relu(x) @ w_true, 最优 loss -> 0

    # ---- 层 1+2: 用 DSL 书写可微函数 (currying 约定: 参数树占第 0 位) ----
    def loss_fn(p: dict[str, Tensor], xx: Tensor, yy: Tensor) -> Tensor:
        h = ops.relu(xx @ p["W1"] + p["b1"])
        o = h @ p["W2"] + p["b2"]
        return ops.mean((o - yy) ** 2)

    loss_and_grad = value_and_grad(loss_fn)

    # ---- DSL 外: 训练循环控制流; 更新表达式仍在 DSL 内 (继续建图) ----
    lr = 0.25
    for step in range(300):
        loss, grads = loss_and_grad(params, x, y)  # 一次 trace, 全惰性
        params = {k: v - lr * grads[k] for k, v in params.items()}
        if step % 60 == 0 or step == 299:
            eval(loss, params)
            print(f"step {step:3d}  loss = {loss.item():.5f}")


def demo3():
    # ---- 组合子与原子层演示: vjp ----
    def f(x: Tensor) -> Tensor:
        return ops.exp(2 * x)

    x = Tensor([0.0, 0.0])
    out, gv = vjp(f, x, [Tensor([1.0, 1.0])])
    eval(out, gv)
    print(
        f"exp(2*x) [0.0,0.0] = {out.tolist()} vjp(exp(2*x)), [1.0,1.0] [0.0,0.0] ) = {gv.tolist()}, "
    )


def main():
    demo1()
    demo3()


if __name__ == "__main__":
    main()
