# tinyautograd 任务入口
# 用法: cd tinyautograd && just docs / just serve / just test / just all

default: docs

# 构建书 (quarto + 离线 katex, 含产物验证)
docs:
    ./scripts/build-docs.sh

# 固定端口后台预览: http://localhost:4866 (幂等, 已在跑则跳过)
serve:
    ./scripts/serve-docs.sh

# 停止预览
stop:
    lsof -tiTCP:4866 -sTCP:LISTEN | xargs kill 2>/dev/null || echo "无运行实例"

# 前台交互式预览 (调试用; 正常场景用 serve)
preview:
    cd docs-book && quarto preview

# 跑测试
test:
    .venv/bin/python -m pytest tests/ -q

# 测试 + 构建 + 预览 (修改书后的默认动作)
all: test docs serve
