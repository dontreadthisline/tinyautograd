#!/usr/bin/env python3
"""serve-docs.py — 静态托管 _book, 所有响应带 no-cache 头。

替代 python -m http.server: 后者不发 Cache-Control, 浏览器对 HTML
启用启发式缓存, 书更新后 sidebar/目录显示旧内容。
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, *args):  # 静默访问日志
        pass


def main() -> None:
    port = int(sys.argv[1])
    directory = sys.argv[2]
    handler = partial(NoCacheHandler, directory=directory)
    server = ThreadingHTTPServer(("", port), handler)
    print(f"serving {Path(directory).resolve()} on :{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
