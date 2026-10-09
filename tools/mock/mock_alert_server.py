#!/usr/bin/env python3
# tools/mock/mock_alert_server.py
#
# 告警接收端（模拟）——接收 ai_stream_pipeline 里 report 节点 POST 过来的告警，
# 并顺带提供一个 OpenAI 兼容的 VLM 桩，方便整条 alert→evidence→vlm_gate→report
# 链路在没有真实大模型时也能本地跑通、观察 PUSH 结果。
#
# 只用标准库。直接运行即可：
#   python3 tools/mock/mock_alert_server.py
#
# 常用环境变量：
#   ALERT_PORT   告警接收端口，默认 8902   （对应 report.params.url 的端口）
#   VLM_PORT     VLM 桩端口，   默认 8901   （对应 vlm_gate.params.url 的端口）
#   MOCK_VERDICT VLM 返回真/假，默认 true   （false 可验证 gate 判假→DROP，不上报）
#   MOCK_CONF    VLM 置信度，   默认 0.9     （调低到 <tau 验证 HOLD）
#   MOCK_DELAY_MS VLM 响应延迟，默认 0      （调高验证超时/HOLD）
#   RECORD       告警落盘文件，默认 ./mock_alerts.jsonl（每行一条，便于回流/评测）
#
# 与 pipeline 对齐：
#   report 节点 POST 到 <report.url>            → 本服务 /alert
#   vlm_gate  POST 到 <vlm.url>/chat/completions → 本服务 /v1/chat/completions
import json
import os
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALERT_PORT = int(os.environ.get("ALERT_PORT", "8902"))
VLM_PORT = int(os.environ.get("VLM_PORT", "8901"))
MOCK_VERDICT = os.environ.get("MOCK_VERDICT", "true").lower() in ("1", "true", "yes")
MOCK_CONF = float(os.environ.get("MOCK_CONF", "0.9"))
MOCK_DELAY_MS = int(os.environ.get("MOCK_DELAY_MS", "0"))
RECORD = os.environ.get("RECORD", "./mock_alerts.jsonl")

_print_lock = threading.Lock()
_record_lock = threading.Lock()


def log_line(*parts):
    with _print_lock:
        print(" ".join(str(p) for p in parts), flush=True)


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def record_alert(payload):
    """把收到的告警按行写入 RECORD，供后续统计漏报/误报或做评测集。"""
    try:
        with _record_lock, open(RECORD, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:  # 落盘失败不应影响接收本身
        log_line(f"[{now_str()}] 写 RECORD 失败: {e}")


class AlertHandler(BaseHTTPRequestHandler):
    """接收 report 节点上报的告警。"""

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"ok": True, "service": "alert-receiver",
                     "endpoints": ["/alert"], "record": RECORD})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {"_raw": raw.decode("utf-8", "replace")}

        rec = payload.get("alert", payload) or {}
        extra = rec.get("extra_data", {}) or {}
        vlm = extra.get("vlm", {}) or {}

        # 报警图 base64：report 节点把 evidence 标注图编码在此字段随告警上传。
        img_b64 = payload.get("snapshot_base64") or payload.get("image_base64") or ""
        img_note = "-"
        if img_b64:
            img_note = f"img={len(img_b64)}B(b64)"
            try:
                import base64 as _b64
                raw = _b64.b64decode(img_b64)
                os.makedirs("mock_alert_images", exist_ok=True)
                fn = f"mock_alert_images/alert_{int(time.time()*1000)}.jpg"
                with open(fn, "wb") as imf:
                    imf.write(raw)
                img_note = f"img={len(img_b64)}B(b64)->{len(raw)}B 存 {fn}"
            except Exception as e:
                img_note = f"img解码失败:{e}"

        # 人可读摘要
        snap = extra.get("snapshot_path", "-")
        objs = rec.get("object_ids", [])
        log_line(
            f"[{now_str()}] 🚨 ALERT  name={rec.get('alert_name')} "
            f"type={rec.get('alert_type_name')} status={rec.get('status_name')} "
            f"zone={rec.get('zone_no')} tracks={objs} "
            f"| vlm={vlm.get('decision')}({vlm.get('confidence')}) '{vlm.get('reason')}' "
            f"| snap={snap} | {img_note}"
        )

        # 完整落盘（带接收时间）
        entry = {"received_at": datetime.now().isoformat(timespec="seconds"),
                 "payload": payload}
        record_alert(entry)

        self._reply({"ok": True, "received": True})

    def log_message(self, *a):  # 静音默认访问日志
        pass


class VlmHandler(BaseHTTPRequestHandler):
    """OpenAI 兼容的 VLM 桩：返回固定的告警核验结论。"""

    def do_GET(self):
        self._send({"ok": True, "service": "vlm-mock",
                    "model": "mock-vlm", "verdict": MOCK_VERDICT,
                    "confidence": MOCK_CONF})

    def do_POST(self):
        if MOCK_DELAY_MS > 0:
            time.sleep(MOCK_DELAY_MS / 1000.0)

        n = int(self.headers.get("content-length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}

        if MOCK_VERDICT:
            content = {"verdict": True, "confidence": MOCK_CONF,
                       "reason": "儿童在墙/门边"}
        else:
            content = {"verdict": False, "confidence": min(MOCK_CONF, 0.05),
                       "reason": "未看到儿童贴边"}

        body = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": req.get("model", "mock-vlm"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": json.dumps(content, ensure_ascii=False)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        log_line(f"[{now_str()}] 🔎 VLM  verdict={content['verdict']} "
                 f"conf={content['confidence']} (← {req.get('model', 'mock-vlm')})")
        self._send(body)

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def serve(port, handler, name):
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    log_line(f"[{now_str()}] ▶ {name} 监听 :{port}")
    httpd.serve_forever()


def main():
    log_line(f"[{now_str()}] mock 服务启动  VLM_PORT={VLM_PORT} ALERT_PORT={ALERT_PORT} "
             f"verdict={MOCK_VERDICT} conf={MOCK_CONF} delay={MOCK_DELAY_MS}ms record={RECORD}")
    if "--alert-only" in sys.argv:
        serve(ALERT_PORT, AlertHandler, "alert-receiver")
        return
    if "--vlm-only" in sys.argv:
        serve(VLM_PORT, VlmHandler, "vlm-mock")
        return
    threading.Thread(target=serve, args=(VLM_PORT, VlmHandler, "vlm-mock"), daemon=True).start()
    serve(ALERT_PORT, AlertHandler, "alert-receiver")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_line("\n已停止。")
