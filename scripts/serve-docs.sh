#!/usr/bin/env bash
# serve-docs.sh — 固定端口托管已构建的书 (幂等, 秒级)
#
# 静态托管 _book: 不重复渲染 (渲染是 build-docs.sh 的职责)。
# 需要自动重渲染的交互场景请用 `just preview` (quarto preview, 前台)。
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${DOCS_PORT:-4866}"
URL="http://localhost:$PORT"

if ! [ -f docs-book/_book/index.html ]; then
  echo "[serve] _book 不存在, 先构建 ..." >&2
  ./scripts/build-docs.sh
fi

# 已在跑 (响应我们的书): 直接返回
if curl -s --max-time 2 "$URL" | head -c 3000 | grep -q "tinyautograd"; then
  echo "[serve] 已在运行: $URL"
  exit 0
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n >/dev/null 2>&1; then
  echo "[serve] 错误: 端口 $PORT 被占用" >&2
  lsof -iTCP:"$PORT" -sTCP:LISTEN -P -n >&2 || true
  exit 1
fi

nohup python3 -m http.server "$PORT" --directory docs-book/_book \
  > /tmp/tinyautograd-docs-serve.log 2>&1 &

for _ in $(seq 1 10); do
  sleep 0.2
  curl -s -o /dev/null --max-time 1 "$URL" && break
done

if curl -s --max-time 2 "$URL" | head -c 3000 | grep -q "tinyautograd"; then
  echo "[serve] 已启动: $URL"
else
  echo "[serve] 启动异常, 日志: /tmp/tinyautograd-docs-serve.log" >&2
  exit 1
fi
