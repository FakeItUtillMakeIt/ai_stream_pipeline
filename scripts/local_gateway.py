#!/usr/bin/env python3
"""AI Stream 单端口网关：Web/API + HLS 走同一个端口。

路由规则：
  /live/*   -> mediamtx HLS（默认 127.0.0.1:8888，默认保留 /live 前缀）
  /api/*    -> ai_stream_server（默认 127.0.0.1:18888）
  /health   -> ai_stream_server
  /metrics  -> ai_stream_server
  其他      -> Web 前端（默认静态托管 web/pipeline-editor/dist；
               找不到 dist 时回退到 127.0.0.1:5173；可用 --ui-root "" 强制回退）

注意：mediamtx 的 HLS path 就是 RTMP 推流路径里 host 之后的部分。
例如推流 rtmp://localhost/live/child_intrusion_fighting 对应
http://localhost:8888/live/child_intrusion_fighting/index.m3u8，
所以默认保留 /live；如果上游没有这个前缀，加 --strip-hls-prefix。

这样 Cloudflare Quick Tunnel 只需要暴露一个端口，登录页、API、证据文件和
HLS 预览同源，浏览器播放 HLS 不需要额外处理 CORS。

只依赖 Python 3 标准库，适合临时演示 / 内网访问；生产环境建议用
Caddy/Nginx 做同样的反代。
"""

from __future__ import annotations

import argparse
import http.client
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".woff2": "font/woff2",
}


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, app, ui, hls, hls_prefix, timeout):
        super().__init__(address, handler)
        self.app = app
        self.ui = ui
        self.ui_root = None
        self.hls = hls
        self.hls_prefix = hls_prefix.rstrip("/") or "/live"
        self.strip_hls_prefix = False
        self.upstream_timeout = timeout
        self.verbose = False


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ai-stream-gateway"
    sys_version = ""

    def do_GET(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_PATCH(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def _proxy(self):
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.send_error(501, "WebSocket is not supported by this gateway")
            return

        split = urlsplit(self.path)
        path = split.path or "/"
        prefix = self.server.hls_prefix
        if path == prefix or path.startswith(prefix + "/"):
            upstream = self.server.hls
            upstream_path = path
            if self.server.strip_hls_prefix:
                upstream_path = path[len(prefix):] or "/"
        elif path == "/api" or path.startswith("/api/") or path in ("/health", "/metrics"):
            upstream = self.server.app
            upstream_path = path
        elif self.server.ui_root is not None:
            self._serve_static(path)
            return
        else:
            upstream = self.server.ui
            upstream_path = path
        if split.query:
            upstream_path += "?" + split.query

        upstream_host, upstream_port = upstream
        body = self._read_request_body()

        headers = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP or lower in ("host", "content-length"):
                continue
            headers[key] = value
        headers["Host"] = f"{upstream_host}:{upstream_port}"
        headers["X-Forwarded-For"] = (
            f"{self.headers.get('X-Forwarded-For')}, {self.client_address[0]}"
            if self.headers.get("X-Forwarded-For")
            else self.client_address[0]
        )
        if self.headers.get("X-Forwarded-Proto"):
            headers["X-Forwarded-Proto"] = self.headers["X-Forwarded-Proto"]
        if self.headers.get("X-Forwarded-Host"):
            headers["X-Forwarded-Host"] = self.headers["X-Forwarded-Host"]
        if body is not None:
            headers["Content-Length"] = str(len(body))

        conn = http.client.HTTPConnection(
            upstream_host, upstream_port, timeout=self.server.upstream_timeout
        )
        try:
            conn.request(self.command, upstream_path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001
            conn.close()
            self.send_error(502, f"upstream {upstream_host}:{upstream_port} unreachable: {exc}")
            return

        try:
            self.send_response_only(resp.status, resp.reason)
            has_length = False
            for key, value in resp.getheaders():
                lower = key.lower()
                if lower in HOP_BY_HOP:
                    continue
                if lower == "content-length":
                    has_length = True
                elif lower == "location":
                    value = self._rewrite_location(value, upstream_host, upstream_port)
                self.send_header(key, value)

            no_body = self.command == "HEAD" or resp.status in (204, 304)
            if not has_length and not no_body:
                self.send_header("Connection", "close")
                self.close_connection = True
            else:
                self.send_header("Connection", "keep-alive")
            self.end_headers()

            if no_body:
                resp.read()
            else:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def _serve_static(self, path):
        root = self.server.ui_root
        rel = unquote(path).lstrip("/") or "index.html"
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            self.send_error(403, "forbidden")
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            if "." not in Path(rel).name:
                candidate = root / "index.html"
            else:
                self.send_error(404, "not found")
                return
        if not candidate.is_file():
            self.send_error(404, "not found")
            return

        try:
            stat = candidate.stat()
            with candidate.open("rb") as fh:
                self.send_response_only(200, "OK")
                self.send_header("Content-Type", self._mime_type(candidate))
                self.send_header("Content-Length", str(stat.st_size))
                self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
                cache = "no-cache" if candidate.name == "index.html" else "public, max-age=3600"
                self.send_header("Cache-Control", cache)
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                if self.command != "HEAD":
                    while True:
                        chunk = fh.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as exc:
            self.send_error(500, f"read {candidate}: {exc}")

    @staticmethod
    def _mime_type(path):
        suffix = path.suffix.lower()
        if suffix in MIME_TYPES:
            return MIME_TYPES[suffix]
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    def _read_request_body(self):
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            return self._read_chunked()
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else None

    def _read_chunked(self):
        chunks = []
        while True:
            line = self.rfile.readline(65536)
            if not line:
                break
            size_text = line.split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError:
                break
            if size == 0:
                while True:
                    trailer = self.rfile.readline(65536)
                    if trailer in (b"\r\n", b"\n", b""):
                        break
                break
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)
        return b"".join(chunks)

    @staticmethod
    def _rewrite_location(value, host, port):
        for scheme in ("http", "https"):
            origin = f"{scheme}://{host}:{port}"
            if value.startswith(origin):
                return value[len(origin):] or "/"
        return value

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)


def parse_address(value):
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        port = rest.lstrip(":")
    else:
        host, _, port = value.rpartition(":")
    if not host or not port:
        raise argparse.ArgumentTypeError("expected HOST:PORT")
    try:
        return host, int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid port") from exc


def main():
    parser = argparse.ArgumentParser(description="AI Stream single-port gateway")
    parser.add_argument("--listen", type=parse_address, default=("127.0.0.1", 8080))
    parser.add_argument("--ui", type=parse_address, default=("127.0.0.1", 5173))
    parser.add_argument(
        "--ui-root",
        default=None,
        help="静态前端目录；默认自动使用 web/pipeline-editor/dist，传空字符串则改用 --ui 上游",
    )
    parser.add_argument("--app", type=parse_address, default=("127.0.0.1", 18888))
    parser.add_argument("--hls", type=parse_address, default=("127.0.0.1", 8888))
    parser.add_argument("--hls-prefix", default="/live")
    parser.add_argument(
        "--strip-hls-prefix",
        action="store_true",
        help="转发 HLS 时去掉 /live 前缀（默认保留，适配 mediamtx 的 live/<stream> 路径）",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ui_root = None
    if args.ui_root is None:
        default_root = Path(__file__).resolve().parent.parent / "web" / "pipeline-editor" / "dist"
        if (default_root / "index.html").is_file():
            ui_root = default_root.resolve()
    elif args.ui_root:
        candidate = Path(args.ui_root).expanduser().resolve()
        if not (candidate / "index.html").is_file():
            parser.error(f"--ui-root 缺少 index.html: {candidate}")
        ui_root = candidate

    server = GatewayServer(
        args.listen,
        GatewayHandler,
        args.app,
        args.ui,
        args.hls,
        args.hls_prefix,
        args.timeout,
    )
    server.ui_root = ui_root
    server.strip_hls_prefix = args.strip_hls_prefix
    server.verbose = args.verbose

    host, port = args.listen
    print(f"gateway listening on http://{host}:{port}", file=sys.stderr)
    print(f"  /api, /health, /metrics -> {args.app[0]}:{args.app[1]}", file=sys.stderr)
    hls_mode = "strip prefix" if args.strip_hls_prefix else "preserve prefix"
    print(
        f"  {server.hls_prefix}/*               -> {args.hls[0]}:{args.hls[1]} ({hls_mode})",
        file=sys.stderr,
    )
    if ui_root is not None:
        print(f"  /*                      -> static {ui_root}", file=sys.stderr)
    else:
        print(f"  /*                      -> {args.ui[0]}:{args.ui[1]}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
