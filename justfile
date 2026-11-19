# tinyautograd 任务入口
# 用法: cd tinyautograd && just docs / just serve / just preview / just test / just all

default: docs

# 构建书 + 组装落地页 (quarto + 离线 katex, 含产物验证)
docs:
    ./scripts/build-docs.sh

# 固定端口后台预览落地页+书: http://localhost:4866 (幂等, 已在跑则跳过)
serve:
    ./scripts/serve-docs.sh

# 前台预览落地页+书 (与线上同构: landing 为入口, Ctrl+C 停止)
preview:
    @lsof -tiTCP:4866 -sTCP:LISTEN | xargs kill 2>/dev/null || true
    python3 scripts/serve-docs.py 4866 docs-book

# 改 qmd 自动重渲染的交互式预览 (quarto 原生, 只覆盖书部分, 独立端口)
watch:
    cd docs-book && quarto preview --port 4867

# 停止后台预览
stop:
    lsof -tiTCP:4866 -sTCP:LISTEN | xargs kill 2>/dev/null || echo "无运行实例"

# 跑测试
test:
    .venv/bin/python -m pytest tests/ -q

# 测试 + 构建 + 预览 (修改书后的默认动作)
all: test docs serve
