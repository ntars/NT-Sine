#!/usr/bin/env python3
"""官网移动端布局核验：在真实小屏宽度下量横向溢出。

为什么需要：headless Chrome 的 `--window-size` 有 500px 最小宽度，
给 390 实际按 500 渲染，再把图裁成 390——看起来像页面溢出，其实是工具假象。
这里改用 360px 宽的 iframe 加载首页：iframe 内部的媒体查询按 iframe 自身宽度
生效，因此能拿到真实的小屏布局，绕开窗口最小宽度限制。

用法：
    python scripts/check_viewport.py                     # 测 320/360/390/414/768 五档
    python scripts/check_viewport.py --widths 320 360
    python scripts/check_viewport.py --no-baseline       # 不跑"强制 nowrap"对照组

退出码：0 = 各档均无溢出；1 = 有溢出（会列出越界元素）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
]

# 测量页模板：iframe 载入首页，读它的 contentDocument 数越界元素。
# 对照组把行内 code 强制回 nowrap——如果现状零溢出而对照组也零溢出，
# 说明样式里没有隐藏的溢出隐患；两边不一致则能一眼看出是哪条规则在兜底。
PAGE = """<!DOCTYPE html>
<meta charset="utf-8">
<title>视口溢出测量</title>
<iframe id="f" src="/index.html" style="border:0"></iframe>
<pre id="out">measuring</pre>
<script>
const W = __WIDTH__;
const f = document.getElementById('f');
f.width = W;
f.height = 900;
f.addEventListener('load', () => {
  const d = f.contentDocument;
  const root = d.documentElement;

  function measure() {
    const over = [];
    for (const el of d.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      // 代码块自带 overflow-x:auto，内部超宽是设计内的滚动，不撑宽页面
      if (el.closest('.pre')) continue;
      if (r.right > W + 0.5 || r.left < -0.5) {
        over.push(el.tagName.toLowerCase() + (el.className ? '.' + el.className : '')
                  + '@' + Math.round(r.right));
      }
    }
    return { scrollWidth: root.scrollWidth, count: over.length, items: over.slice(0, 6) };
  }

  const shipped = measure();
  let baseline = null;
  if (__BASELINE__) {
    const probe = d.createElement('style');
    probe.textContent = 'code{white-space:nowrap!important;overflow-wrap:normal!important}';
    d.head.appendChild(probe);
    baseline = measure();
    probe.remove();
  }

  document.getElementById('out').textContent = 'RESULT '
    + JSON.stringify({ w: W, shipped, baseline });
});
</script>
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
        found = shutil.which(c)
        if found:
            return found
    raise SystemExit("找不到 Chrome/Chromium，无法做视口核验")


class _QuietHandler(SimpleHTTPRequestHandler):
    """静音版静态服务：只响应文件，不往终端刷访问日志。"""

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        pass


def serve(directory: Path) -> tuple[ThreadingHTTPServer, int]:
    """在后台线程起一个只读静态服务，返回 (服务器, 端口)。"""
    handler = partial(_QuietHandler, directory=str(directory))
    # 端口 0 让系统分配空闲端口，避免和正在跑的开发服务器撞车
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def measure(chrome: str, url: str, timeout: int = 40) -> dict:
    out = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=6000", "--dump-dom", url],
        capture_output=True, text=True, timeout=timeout,
    )
    for line in out.stdout.splitlines():
        if "RESULT " in line:
            return json.loads(line.split("RESULT ", 1)[1].split("</pre>")[0])
    raise SystemExit(f"没能从页面读到测量结果（Chrome 退出码 {out.returncode}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[320, 360, 390, 414, 768])
    ap.add_argument("--no-baseline", action="store_true", help="跳过强制 nowrap 对照组")
    args = ap.parse_args()

    chrome = find_chrome()
    httpd, port = serve(ROOT)
    bad = []
    try:
        print(f"{'宽度':>6}  {'scrollWidth':>11}  {'溢出元素':>8}  说明")
        print("-" * 52)
        for w in args.widths:
            page = PAGE.replace("__WIDTH__", str(w)).replace(
                "__BASELINE__", "false" if args.no_baseline else "true")
            # 测量页必须和 index.html 同源，才能读 iframe 的 contentDocument
            probe = ROOT / f".vp_probe_{w}.html"
            probe.write_text(page, encoding="utf-8")
            try:
                res = measure(chrome, f"http://127.0.0.1:{port}/{probe.name}")
            finally:
                probe.unlink(missing_ok=True)

            s = res["shipped"]
            note = "窗口宽度一致" if s["scrollWidth"] <= w else "页面被撑宽"
            if res.get("baseline") and res["baseline"]["count"] > s["count"]:
                note += f"；关掉断点样式会多 {res['baseline']['count'] - s['count']} 处溢出"
            print(f"{w:>6}  {s['scrollWidth']:>11}  {s['count']:>8}  {note}")
            if s["count"]:
                bad.append((w, s["items"]))
    finally:
        httpd.shutdown()

    if bad:
        print("\n有溢出，需要修：")
        for w, items in bad:
            print(f"  {w}px: {', '.join(items)}")
        return 1
    print("\n各档均无横向溢出，小屏布局正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
