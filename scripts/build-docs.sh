#!/usr/bin/env bash
# build-docs.sh — tinyautograd 书构建 (quarto + 离线 katex)
#
# 流程: 检查 quarto -> 幂等同步本地 katex 资源 -> quarto render -> 验证产物
# 验证项: 页面数、零外网 CDN 引用、katex 资源随书打包、源码 include 已展开
set -euo pipefail
cd "$(dirname "$0")/.."

QUARTO_BIN="${QUARTO_BIN:-quarto}"
KATEX_SRC_CANDIDATES=(
  "/opt/homebrew/lib/node_modules/@slidev/cli/node_modules/katex/dist"
)
DEST="docs-book/resources/libs/katex"

# ---- 1. 工具检查 -----------------------------------------------------------
if ! command -v "$QUARTO_BIN" >/dev/null 2>&1; then
  echo "错误: 未找到 quarto。安装: brew install quarto" >&2
  exit 1
fi

# ---- 2. katex 资源幂等同步 (源: 本机 slidev 自带, 全程离线) ----------------
if [ ! -f "$DEST/katex.min.js" ]; then
  src=""
  for c in "${KATEX_SRC_CANDIDATES[@]}"; do
    [ -f "$c/katex.min.js" ] && src="$c" && break
  done
  if [ -z "$src" ]; then
    echo "错误: $DEST 缺失, 且找不到本机 katex 源 (slidev)。" >&2
    echo "  方案一: npm i -g @slidev/cli 后重跑 (脚本会从其 node_modules 复制)" >&2
    echo "  方案二: 从任何机器拷贝 katex/dist 到 $DEST" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$DEST")"
  cp -r "$src" "$DEST"
  echo "[katex] 已从 $src 同步到 $DEST"
else
  echo "[katex] 资源在位, 跳过同步"
fi

# ---- 3. 渲染 (内容未变则跳过; FORCE=1 强制) --------------------------------
newest_src=$( { stat -f '%m' docs-book/_quarto.yml 2>/dev/null; \
    find docs-book -maxdepth 1 \( -name '*.qmd' -o -name '*.html' -o -name '*.bib' \) -exec stat -f '%m' {} \; ; \
    find tinyautograd -name '*.py' -exec stat -f '%m' {} \; ; } | sort -rn | head -1 )
book_m=$(stat -f '%m' docs-book/_book/index.html 2>/dev/null || echo 0)
if [ "${FORCE:-0}" != "1" ] && [ "$book_m" -gt 0 ] && [ "$book_m" -ge "$newest_src" ]; then
  echo "[quarto] 产物已是最新, 跳过渲染 (FORCE=1 可强制)"
else
  echo "[quarto] render ..."
  (cd docs-book && "$QUARTO_BIN" render >/dev/null)
fi

# ---- 4. 产物验证 -----------------------------------------------------------
python3 - <<'EOF'
import glob, os, re, sys

pages = sorted(glob.glob("docs-book/_book/*.html"))
fail = []

if len(pages) < 9:
    fail.append(f"页面数 {len(pages)} < 9")

for f in pages:
    html = open(f).read()
    name = os.path.basename(f)
    ext = [u for u in re.findall(r"https?://[^\"'\s>]*", html)
           if not any(x in u for x in ("github.com", "w3.org", "purl.org", "namespace"))]
    if any("katex@" in u or "polyfill" in u or "mathjax" in u for u in ext):
        fail.append(f"{name}: 仍引用数学 CDN: {ext[:2]}")
    for m in set(re.findall(r'(?:src|href)="([^"]*katex[^"]*)"', html)):
        if not os.path.exists(os.path.join("docs-book/_book", m)):
            fail.append(f"{name}: katex 资源缺失 {m}")

# 源码 include 展开 (书与代码同构的关键断言)
for page, mark in [("02-value-centric", "__slots__"), ("03-dsl", "lifted_multi")]:
    if mark not in open(f"docs-book/_book/{page}.html").read():
        fail.append(f"{page}: 源码未展开 (include 失效?)")

if fail:
    print("验证失败:")
    for x in fail:
        print(" -", x)
    sys.exit(1)
print(f"[验证] {len(pages)} 页 | 零外网数学 CDN | katex 资源随书 | 源码 include 已展开")
EOF

echo "构建完成: $(pwd)/docs-book/_book/index.html"
