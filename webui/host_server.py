#!/usr/bin/env python3
"""Speaksy host bridge — serves the React UI and speaks its WebSocket protocol.

The UI (``webui/rd_ui_v1.1.html``) talks to a "Host" through a small message
bus. In the packaged desktop build that Host is WebView2; here it is a local
WebSocket server so the exact same UI runs in any browser on macOS/Linux/WSL.

This is the **bridge skeleton** (step 1): it serves the page, answers device
queries, and streams a *demo* JA→VI subtitle feed on start/stop so the whole
round-trip can be verified end-to-end. Wiring the real ``TranslationPipeline``
is step 2 — see ``_DemoEngine`` for the single seam to replace.

Zero third-party deps: a minimal RFC 6455 WebSocket sits on top of asyncio's
stream server, so ``python3 webui/host_server.py`` runs with nothing installed.

Protocol (mirrors the HostBridge in the HTML):
    UI → Host   ui/ready, engine/listDevices, engine/start, engine/stop,
                engine/testDevice, ui/textSnapshot
    Host → UI   engine/devices, engine/status, engine/subtitle,
                engine/error, engine/testResult
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import struct
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger("speaksy.host")

_ROOT = Path(__file__).resolve().parent.parent
_UI_FILE = Path(__file__).resolve().parent / "rd_ui_v1.1.html"
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Make `src` importable regardless of where the host is launched from, so real
# device probing works (mirrors webui/streamlit_app.py).
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env so ZT_WEBHOOK_URL and other settings are available even before
# config.py is imported (demo mode never imports config until _RealEngine).
try:
    import config as _config  # noqa: F401 — side-effect: loads .env into os.environ
except Exception:
    pass  # best-effort; env vars set in the shell still work

# Fallback device list when soundcard is unavailable (e.g. headless WSL). The UI
# only needs {id, name}; ids are opaque strings it round-trips back on start.
_DEMO_DEVICES = [
    {"id": "default", "name": "(demo) Default Device"},
    {"id": "loopback", "name": "(demo) System Loopback"},
]


def list_input_devices() -> tuple[list[dict[str, str]], str | None]:
    """Return (devices, default_id) from soundcard, or demo devices if absent.

    Best-effort by design: importing ``audio_capture`` pulls in numpy +
    soundcard, which may be missing on a headless box. Any failure degrades to
    the demo list so the UI still renders a working device picker.

    Note: we enumerate via ``soundcard`` directly rather than
    ``audio_capture.list_devices()`` — that helper ``print()``s every device,
    which spams the console each time the UI polls for devices.
    """
    try:
        import soundcard as sc

        from src import audio_capture

        default_id: str | None = None
        try:
            loop = audio_capture.find_loopback_device()
            if loop is not None:
                default_id = "loopback"
        except Exception:  # noqa: BLE001 - device probing is inherently flaky
            pass

        devices: list[dict[str, str]] = [{"id": "loopback", "name": "System Loopback (what you hear)"}]
        for index, mic in enumerate(sc.all_microphones(include_loopback=True)):
            name = str(getattr(mic, "name", mic))
            devices.append({"id": f"idx:{index}", "name": name})
        if default_id is None:
            default_id = devices[0]["id"]
        return devices, default_id
    except Exception as exc:  # noqa: BLE001 - headless/no-audio fallback
        logger.info("soundcard unavailable (%s); using demo devices", exc)
        return list(_DEMO_DEVICES), "default"


# --------------------------------------------------------------------------- #
# Translation — real JA→VI via the 9router backend (RouterTranslator)          #
# --------------------------------------------------------------------------- #

# Scripted Japanese utterances stand in for ASR output (this box has no audio);
# the Vietnamese is produced *live* by the real translator. ``fallback`` is only
# used if the 9router gateway is unreachable, so the demo never shows blanks.
_DEMO_SCRIPT = [
    {"src": "会議の後で資料を送ります。", "fallback": "Tôi sẽ gửi tài liệu sau cuộc họp."},
    {"src": "皆さん、ご意見はありますか？", "fallback": "Mọi người có ý kiến gì không?"},
    {"src": "この部分は締め切りが厳しいので優先しましょう。", "fallback": "Phần này hạn chót gấp nên hãy ưu tiên xử lý."},
    {"src": "では、次のアジェンダに移ります。", "fallback": "Vậy thì, chúng ta chuyển sang mục tiếp theo."},
]

_translator = None
_translator_lock = threading.Lock()
_translator_failed = False


def get_translator():
    """Lazily build the configured translator (RouterTranslator by default).

    Best-effort and cached: a failure (gateway down, deps missing) is remembered
    so the demo falls back to scripted Vietnamese instead of retrying on every
    segment. Returns None when no translator is available.
    """
    global _translator, _translator_failed
    if _translator is not None or _translator_failed:
        return _translator
    with _translator_lock:
        if _translator is not None or _translator_failed:
            return _translator
        try:
            from src.router_translator import RouterTranslator

            t = RouterTranslator()
            t.warmup()
            _translator = t
            logger.info("Host translator ready: RouterTranslator (9router)")
        except Exception as exc:  # noqa: BLE001 - degrade to scripted fallback
            logger.warning("Translator unavailable (%s); using scripted fallback", exc)
            _translator_failed = True
    return _translator


async def translate_ja_vi(text: str, fallback: str) -> str:
    """Translate Japanese→Vietnamese off the event loop, with a safe fallback."""
    t = get_translator()
    if t is None:
        return fallback
    loop = asyncio.get_event_loop()
    try:
        out = await loop.run_in_executor(None, t.translate, text)
        return (out or "").strip() or fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("translate failed (%s); using fallback", exc)
        return fallback


def _fire_webhook(session_id: str, seq: int, japanese: str, vietnamese: str) -> None:
    """POST one translated pair to the webhook URL configured in ZT_WEBHOOK_URL."""
    url = os.environ.get("ZT_WEBHOOK_URL", "")
    if not url:
        return
    payload = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": f"🎙️ [{seq}] {japanese}",
                "wrap": True,
                "size": "Small",
                "color": "Accent",
            },
            {
                "type": "TextBlock",
                "text": f"🇻🇳 {vietnamese}",
                "wrap": True,
                "size": "Default",
                "weight": "Bolder",
            },
        ],
    }

    timeout = float(os.environ.get("ZT_WEBHOOK_TIMEOUT", "5"))
    proxy_url = os.environ.get("ZT_WEBHOOK_PROXY", "")
    proxies = {"https": proxy_url, "http": proxy_url} if proxy_url else {}

    def _post() -> None:
        try:
            import requests as _req
            s = _req.Session()
            s.trust_env = not proxy_url  # ignore HTTPS_PROXY from Windows env
            if proxy_url:
                s.proxies = proxies
            s.post(url, json=payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webhook POST failed seq=%d: %s", seq, exc)

    threading.Thread(target=_post, daemon=True).start()


class _DemoEngine:
    """Streams a JA→VI subtitle feed: scripted Japanese, *live* translation.

    Stands in for the audio→ASR→translate pipeline on a box with no audio: it
    walks the scripted Japanese utterances, translates each through the real
    9router backend, and emits partial → final (``segmentEnd:true``) frames plus
    periodic ``engine/status`` so the UI's RTF/latency/EQ indicators animate.
    """

    def __init__(self, conn: "Connection", from_lang: str, to_lang: str):
        self._conn = conn
        self._from = from_lang
        self._to = to_lang
        self._task: asyncio.Task | None = None
        self._session_id = str(__import__("uuid").uuid4())
        self._seq = 0

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        await self._conn.send({
            "type": "engine/status",
            "payload": {"connected": True, "running": True, "device": "9router", "state": "Running"},
        })
        i = 0
        try:
            while True:
                item = _DEMO_SCRIPT[i % len(_DEMO_SCRIPT)]
                i += 1
                src = item["src"]

                # Show the recognized Japanese first (partial, VI pending), then
                # the committed segment once the real translation returns.
                await self._subtitle(src, "…", partial=True, segment_end=False, ts=_now_ms())
                t0 = time.time()
                dst = await translate_ja_vi(src, item["fallback"])
                latency_ms = int((time.time() - t0) * 1000)
                await self._subtitle(src, dst, partial=False, segment_end=True, ts=_now_ms())
                self._seq += 1
                if dst:
                    _fire_webhook(self._session_id, self._seq, src, dst)

                await self._conn.send({
                    "type": "engine/status",
                    "payload": {
                        "running": True,
                        "device": "9router",
                        "state": "Running",
                        "latencyMs": latency_ms,
                    },
                })
                await asyncio.sleep(1.4)
        except asyncio.CancelledError:
            raise

    async def _subtitle(self, src: str, dst: str, *, partial: bool, segment_end: bool, ts: int) -> None:
        await self._conn.send({
            "type": "engine/subtitle",
            "payload": {
                "tsMs": ts,
                "srcText": src,
                "dstText": dst,
                "partial": partial,
                "segmentEnd": segment_end,
            },
        })


class WsDisplay:
    """A ``SubtitleDisplay`` look-alike that forwards the pipeline to WebSocket.

    The real ``TranslationPipeline`` pushes results by calling display methods
    from its worker threads. This adapter implements that same surface and turns
    each call into an ``engine/subtitle`` / ``engine/status`` frame, marshalled
    back onto the server's event loop with ``run_coroutine_threadsafe`` (the
    pipeline threads are not the loop thread).

    Pairing model: ``show_source`` stashes the latest Japanese; ``show_target``
    emits the committed bilingual segment. ``show_pair`` does both at once.
    """

    def __init__(self, conn: "Connection", loop: asyncio.AbstractEventLoop):
        self._conn = conn
        self._loop = loop
        self._last_src = ""

    def _emit(self, coro) -> None:
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _send(self, obj: dict) -> None:
        self._emit(self._conn.send(obj))

    # ---- SubtitleDisplay surface used by the pipeline ---------------------- #
    def show_source(self, japanese: str, seq: int | None = None) -> None:
        self._last_src = japanese or ""
        self._send({
            "type": "engine/subtitle",
            "payload": {"tsMs": _now_ms(), "srcText": self._last_src, "dstText": "…",
                        "partial": True, "segmentEnd": False},
        })

    def show_source_partial(self, committed: str, tail: str = "") -> None:
        text = (committed or "") + (tail or "")
        self._send({
            "type": "engine/subtitle",
            "payload": {"tsMs": _now_ms(), "srcText": text, "dstText": "…",
                        "partial": True, "segmentEnd": False},
        })

    def finalize_source(self, japanese: str) -> None:
        self._last_src = japanese or self._last_src

    def show_target(self, vietnamese: str, japanese: str | None = None, seq: int | None = None) -> None:
        src = japanese if japanese is not None else self._last_src
        self._send({
            "type": "engine/subtitle",
            "payload": {"tsMs": _now_ms(), "srcText": src or "", "dstText": vietnamese or "",
                        "partial": False, "segmentEnd": True},
        })

    def show_pair(self, japanese: str, vietnamese: str) -> None:
        self._send({
            "type": "engine/subtitle",
            "payload": {"tsMs": _now_ms(), "srcText": japanese or "", "dstText": vietnamese or "",
                        "partial": False, "segmentEnd": True},
        })

    def show(self, japanese: str, vietnamese: str) -> None:
        self.show_pair(japanese, vietnamese)

    def info(self, message: str) -> None:
        logger.info("[pipeline] %s", message)
        self._send({
            "type": "engine/status",
            "payload": {"connected": True, "running": True, "device": "pipeline", "state": str(message)[:80]},
        })


class _RealEngine:
    """Runs the real ``TranslationPipeline`` (audio→ASR→translate) on a thread.

    Requires a capture device and the ASR/translate dependencies, so it only
    works on a machine with audio (Windows/macOS) — opt in with ``ZT_HOST_REAL=1``.
    On headless WSL the imports fail and the engine reports the error to the UI
    instead of crashing the host.
    """

    def __init__(self, conn: "Connection", from_lang: str, to_lang: str, device_id: str):
        self._conn = conn
        self._from = from_lang
        self._to = to_lang
        self._device_id = device_id
        self._loop = asyncio.get_event_loop()
        self._pipeline = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="RealEngine", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop_event.set()  # cooperative shutdown of all stages
            except Exception:  # noqa: BLE001
                pass
        # run_forever() returns once the stop_event propagates; join off-loop.
        if self._thread is not None:
            await self._loop.run_in_executor(None, self._thread.join, 5.0)
            self._thread = None
        self._pipeline = None

    def _run(self) -> None:
        try:
            import config
            from src import audio_capture
            from src.pipeline import TranslationPipeline
        except Exception as exc:  # noqa: BLE001 - missing audio/ASR deps (e.g. WSL)
            self._report_error(f"Real pipeline unavailable: {exc}")
            return
        try:
            device = self._resolve_device(audio_capture)
            display = WsDisplay(self._conn, self._loop)
            # Prefer the low-latency streaming recognizer, but only if its model
            # is present — otherwise fall back to the offline recognizer
            # (ReazonSpeech), which ships with the default model download. This
            # avoids hard-failing when only the offline ASR model is installed.
            streaming = Path(str(config.STREAMING_ASR_MODEL_DIR)).is_dir()
            if not streaming:
                display.info("Streaming ASR model not found; using offline recognizer.")
            self._pipeline = TranslationPipeline(
                device=device, display=display, streaming=streaming, backend="local",
            )
            self._pipeline.run_forever()
        except Exception as exc:  # noqa: BLE001
            self._report_error(f"Pipeline error: {exc}")
        finally:
            # Release the translator's pooled HTTP connections on shutdown so
            # repeated start/stop cycles don't leak sockets.
            translator = getattr(self._pipeline, "translator", None)
            close = getattr(translator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _resolve_device(audio_capture):
        # ZT_HOST_MIC=1 captures the default microphone instead of the system
        # loopback (what-you-hear). Default is loopback for meeting audio.
        if os.environ.get("ZT_HOST_MIC") == "1":
            return audio_capture.get_default_microphone()
        dev = audio_capture.find_loopback_device()
        return dev if dev is not None else audio_capture.get_default_microphone()

    def _report_error(self, message: str) -> None:
        logger.error(message)
        try:
            asyncio.run_coroutine_threadsafe(
                self._conn.send({"type": "engine/error", "payload": {"message": message}}),
                self._loop,
            )
        except RuntimeError:
            pass


def _now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- #
# WebSocket connection + protocol dispatch                                    #
# --------------------------------------------------------------------------- #


class Connection:
    """One browser tab: frames in, JSON messages out, protocol dispatch."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._engine: "_DemoEngine | _RealEngine | None" = None
        self._running = False

    # ---- outbound ---------------------------------------------------------- #
    async def send(self, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        frame = _encode_text_frame(data)
        async with self._write_lock:
            self._writer.write(frame)
            await self._writer.drain()

    # ---- lifecycle --------------------------------------------------------- #
    async def serve(self) -> None:
        try:
            while True:
                msg = await _read_message(self._reader)
                if msg is None:
                    break  # close frame or EOF
                await self._dispatch(msg)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            await self._teardown()

    async def _teardown(self) -> None:
        if self._engine is not None:
            await self._engine.stop()
            self._engine = None
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- protocol ---------------------------------------------------------- #
    async def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("dropping non-JSON frame: %r", raw[:120])
            return
        mtype = msg.get("type")
        payload = msg.get("payload") or {}

        if mtype == "ui/ready":
            await self._send_status(state="Idle")
            await self._send_devices()
        elif mtype == "engine/listDevices":
            await self._send_devices()
        elif mtype == "engine/start":
            await self._on_start(payload)
        elif mtype == "engine/stop":
            await self._on_stop()
        elif mtype == "engine/testDevice":
            await self.send({
                "type": "engine/testResult",
                "payload": {"ok": True, "message": "デバイスは利用可能です（demo host）。"},
            })
        elif mtype == "ui/textSnapshot":
            pass  # step 2: persist snapshot to disk for file-output trigger
        else:
            logger.debug("unhandled message type: %s", mtype)

    async def _send_status(self, *, state: str) -> None:
        await self.send({
            "type": "engine/status",
            "payload": {
                "connected": True,
                "running": self._running,
                "device": "CPU",
                "state": state,
            },
        })

    async def _send_devices(self) -> None:
        devices, default_id = list_input_devices()
        await self.send({
            "type": "engine/devices",
            "payload": {"inputDevices": devices, "defaultInputDeviceId": default_id},
        })

    async def _on_start(self, payload: dict) -> None:
        if self._running:
            return
        from_lang = str(payload.get("fromLang") or "auto")
        to_lang = str(payload.get("toLang") or "vi")
        device_id = str(payload.get("inputDeviceId") or "")
        self._running = True
        # ZT_HOST_REAL=1 opts into the real audio→ASR→translate pipeline (needs a
        # capture device + ASR deps — i.e. a Windows/macOS box, not headless WSL).
        # Anywhere else, the scripted engine drives real 9router translation.
        if os.environ.get("ZT_HOST_REAL") == "1":
            self._engine = _RealEngine(self, from_lang, to_lang, device_id)
        else:
            self._engine = _DemoEngine(self, from_lang, to_lang)
        self._engine.start()

    async def _on_stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()
            self._engine = None
        self._running = False
        await self._send_status(state="Idle")


# --------------------------------------------------------------------------- #
# Minimal RFC 6455 framing (server side: read masked, write unmasked)         #
# --------------------------------------------------------------------------- #


def _encode_text_frame(payload: bytes) -> bytes:
    header = bytearray([0x81])  # FIN + text opcode
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    return bytes(header) + payload


async def _read_message(reader: asyncio.StreamReader) -> str | None:
    """Read one text message; return None on close/ping-handled/EOF-control."""
    b0, b1 = await reader.readexactly(2)
    opcode = b0 & 0x0F
    masked = (b1 & 0x80) != 0
    length = b1 & 0x7F

    if length == 126:
        (length,) = struct.unpack(">H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack(">Q", await reader.readexactly(8))

    mask = await reader.readexactly(4) if masked else b"\x00\x00\x00\x00"
    payload = bytearray(await reader.readexactly(length))
    if masked:
        for i in range(length):
            payload[i] ^= mask[i % 4]

    if opcode == 0x8:  # close
        return None
    if opcode == 0x9:  # ping — caller doesn't need it; pong is best-effort skipped
        return ""
    if opcode == 0xA:  # pong
        return ""
    return payload.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# HTTP: serve the UI, then upgrade /ws to WebSocket                            #
# --------------------------------------------------------------------------- #


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request_line = await reader.readline()
        if not request_line:
            writer.close()
            return
        parts = request_line.decode("latin-1").split()
        method, path = (parts[0], parts[1]) if len(parts) >= 2 else ("GET", "/")

        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()

        if headers.get("upgrade", "").lower() == "websocket":
            await _do_handshake(writer, headers)
            await Connection(reader, writer).serve()
            return

        await _serve_http(writer, method, path)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def _do_handshake(writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
    key = headers.get("sec-websocket-key", "")
    accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    writer.write(resp.encode("latin-1"))
    await writer.drain()


async def _serve_http(writer: asyncio.StreamWriter, method: str, path: str) -> None:
    clean = path.split("?", 1)[0]
    if clean in ("/", "/index.html", "/rd_ui_v1.1.html"):
        body = _UI_FILE.read_bytes()
        _write_response(writer, 200, "text/html; charset=utf-8", body)
    elif clean == "/healthz":
        _write_response(writer, 200, "text/plain; charset=utf-8", b"ok")
    else:
        _write_response(writer, 404, "text/plain; charset=utf-8", b"not found")
    await writer.drain()


def _write_response(writer: asyncio.StreamWriter, status: int, content_type: str, body: bytes) -> None:
    reason = {200: "OK", 404: "Not Found"}.get(status, "OK")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n"
    )
    writer.write(head.encode("latin-1") + body)


async def _main(host: str, port: int) -> None:
    if not _UI_FILE.exists():
        raise SystemExit(f"UI file not found: {_UI_FILE}")
    server = await asyncio.start_server(_handle, host, port)
    addr = ", ".join(str(s.getsockname()) for s in server.sockets)
    logger.info("Speaksy host serving UI + WS on http://%s:%d  (sockets: %s)", host, port, addr)
    print(f"==> Speaksy host ready → http://{host}:{port}")
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaksy host bridge (serve UI + WebSocket).")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8770, help="port (default: 8770)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(_main(args.host, args.port))
    except KeyboardInterrupt:
        print("\n==> Speaksy host stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
