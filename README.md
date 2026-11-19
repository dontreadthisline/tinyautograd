# tinyautograd

MLX 之上的极简函数式 autograd 框架。只在 `mx.array` 上做薄封装:
**自己实现计算图与反向传播**, 每个算子只复用 MLX 的单算子前向与
`mx.vjp` 局部反传。

配套有一本自包含的设计之书 (Quarto, 见下方"构建本书"), 从链式法则的
数学一路讲到 Python 接口与类型系统。

## 快速上手

```python
from tinyautograd import Tensor, ops, random, value_and_grad, eval

def loss_fn(params: dict[str, Tensor], x: Tensor, y: Tensor) -> Tensor:
    h = ops.relu(x @ params["W1"] + params["b1"])
    o = h @ params["W2"] + params["b2"]
    return ops.mean((o - y) ** 2)

random.seed(0)
params = {"W1": random.normal((6, 32)) * 0.1, ...}
loss_and_grad = value_and_grad(loss_fn)
loss, grads = loss_and_grad(params, x, y)   # 一次 trace, 全惰性
params = {k: v - 0.1 * g for k, (v, g) in ...}
eval(loss, params)                           # 唯一显式算账门
```

## API 一览

| 层 | 接口 | 说明 |
|---|---|---|
| 值的语言 | `Tensor` + `ops.*` + `random.*` | 运算符中缀记法 + 模块函数; **无方法链**; 随机源独立成模块 (每次调用重采样, 叶子不参与求导) |
| 函数的算子 | `vjp` / `value_and_grad` / `grad` | 自函子: Callable → Callable, 可组合 |
| 边界原语 | `eval` / `no_grad` / `detach` | DSL ↔ Python 世界的门 |
| 扩展门 | `lift` / `lift_multi` / `const` | mx 函数 → 可导算子 / 多输出算子 / 常量源, 框架与用户零特权 |

算子词汇表静态定义于 `tinyautograd/ops.py` (一行一个, 全部可见、
可补全、可跳转), 不经过任何动态代理。

## 结构

```
tinyautograd/
  tensor.py      # Tensor: 值 = 图节点 = 线性映射载体
  ops.py         # 算子门面 + lift/const 提升门
  autograd.py    # 反向引擎 (一次 DFS) + vjp/grad/value_and_grad + eval/no_grad/detach
  pytree.py      # Tensor 树摊平 / 同构还原
  random.py      # 随机源 (const 编码)
tests/           # 前向对拍 mx + 反向对拍 mx.value_and_grad + 端到端收敛
examples/demo.py # 两层 MLP 训练
docs-book/       # 设计之书 (Quarto, 自包含离线构建)
```

## 使用

```bash
uv sync                        # 或: uv run pytest tests/ -q
uv run pytest tests/ -q       # 测试
uv run python examples/demo.py
```

## 构建本书

```bash
just all    # test + 构建书 + 固定端口预览 (http://localhost:4866)
just serve  # 仅预览;  just docs 仅构建;  just stop 停止
```

书依赖 Quarto (`brew install quarto`); 数学渲染使用本地打包的 KaTeX
(取自本机 slidev 的 node_modules, 全程离线, 首次构建自动同步)。

## 设计决策速览

- 反向为函数式变换, Tensor 上无 `.backward()`/`.grad`/`requires_grad`
- 算子全部为模块函数, 无方法链 (仅中缀记法 + 元数据 property + DSL 出口)
- 图随 Python 引用存活, `no_grad` 控制建边, `detach` 为语义孔 (stop-gradient)
- 绑定 MLX 单后端 (曾有可插拔后端层, 为算子静态可见性而移除, 书中第 4 章记录了这次重构)
- 类型: PEP 695 (`type` 语句 / `ParamSpec`), `PyTree[T]` 递归泛型
- 不做: module 层、优化器、vmap/compile、多设备、高阶导
