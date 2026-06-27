# Web UI + 9router translation backend

This document covers the browser UI ("Speaksy") and the **9router** translation
backend added on top of the original CLI translator. The original
audio → ASR → translate → subtitle pipeline is unchanged; these are two additive
layers:

1. A **host bridge** that serves the React UI in a browser and drives the real
   pipeline over WebSocket.
2. A **9router translator backend** that offloads JA→VI translation to a local
   OpenAI-compatible gateway, so no heavy local LLM (Qwen) is loaded.

ASR still runs **locally** in every mode — it must, to hear the meeting. Only the
translation step is offloaded.

---

## 1. Quick start

Start the 9router gateway first (it listens on `http://127.0.0.1:20128/v1`), then:

**macOS / Linux:**

```bash
# Terminal CLI, local ASR + 9router translation (no Qwen download):
./run.sh --router                 # bootstraps .env from .env.example on first use
./run.sh --router --streaming     # lower-latency streaming ASR
./run.sh --router --mic           # capture the microphone instead of system audio

# Web UI in a browser (set the backend explicitly):
ZT_TRANSLATOR=router ZT_HOST_REAL=1 ./run-host.sh   # http://127.0.0.1:8770
```

macOS system-audio capture needs [BlackHole](https://github.com/ExistentialAudio/BlackHole)
(or an Aggregate Device) as the loopback input; use `--mic` to skip that.

**Windows:**

```powershell
# Terminal CLI, local ASR + 9router translation (no Qwen download):
.\run.ps1 -Router
.\run.ps1 -Router -Streaming          # lower-latency streaming ASR
.\run.ps1 -Router -Model "gh/claude-opus-4.8"   # higher-quality model

# Web UI in a browser:
.\run-host.ps1                        # real audio -> ASR -> 9router, http://127.0.0.1:8770
.\run-host.ps1 -Demo                  # no audio: scripted Japanese, live 9router translation
.\run-host.ps1 -Mic                   # capture microphone instead of system loopback
```

Open the printed `http://127.0.0.1:8770` and click the start button (翻訳開始).

`-Demo` is the fastest way to confirm the UI + gateway work without installing the
ASR stack or capturing audio.

---

## 2. Architecture

```
Browser (webui/rd_ui_v1.1.html, React SPA "Speaksy")
   │   HostBridge transport: WebView2 → WebSocket → postMessage
   ▼
webui/host_server.py  (stdlib asyncio, hand-rolled RFC6455 WebSocket; zero deps)
   │   serves UI at /, WebSocket at /ws
   ├── _DemoEngine   — scripted Japanese, translated live by 9router (no audio)
   └── _RealEngine   — runs the real TranslationPipeline on a thread (ZT_HOST_REAL=1)
                       WsDisplay adapts pipeline display calls → engine/subtitle frames
   ▼
src/pipeline.py  (audio_capture → VAD/ASR → aggregator → translator → display)
   ▼
src/router_translator.py  RouterTranslator → 9router gateway (OpenAI /v1/chat/completions)
```

### WebSocket protocol (UI ⇄ host)

| Direction | Messages |
|-----------|----------|
| UI → Host | `ui/ready`, `engine/listDevices`, `engine/start{inputDeviceId,fromLang,toLang,...}`, `engine/stop`, `engine/testDevice`, `ui/textSnapshot` |
| Host → UI | `engine/devices`, `engine/status`, `engine/subtitle{srcText,dstText,partial,segmentEnd,tsMs}`, `engine/error`, `engine/testResult` |

The same UI file works unchanged in the packaged desktop build (WebView2) and in a
browser (WebSocket) — `HostBridge.hasHost()` picks the transport.

---

## 3. The 9router translator backend

`RouterTranslator` (`src/router_translator.py`) is a drop-in alongside the NLLB
and local-LLM backends (same `translate` / `translate_many` / `warmup` contract).
Select it with `ZT_TRANSLATOR=router`; the pipeline factory
(`pipeline._create_translator`) falls back to NLLB if init fails.

It calls the gateway with the standard OpenAI body and reads
`choices[0].message.content`.

### Domain quality layer

Even though a strong hosted model does the translating, a thin deterministic layer
runs first, tuned for this domain (IT engineering + Japanese emergency medical
dispatch, 救急搬送システム):

- **Person names → romaji** before sending (深瀬さん → `Fukase-san`), so names are
  never translated literally. Uses the shared `src/japanese_names.py` map.
- **Latin-only source substitution** — katakana loanwords and proper nouns whose
  mapping is ASCII (`マイクロサービス`→`microservice`, `DMAT`) are substituted into
  the still-Japanese source; Vietnamese-valued terms are **not** (that creates a
  half-translated hybrid the model misreads), they are pinned via the prompt.
- **System-prompt glossary** — the Vietnamese domain glossary
  (`src/domain_data.py`: 消防=cứu hỏa, 搬送=vận chuyển, 傷病者=nạn nhân, …) is
  appended to the system prompt.
- **Filler fast-path** — pure back-channel (はい, うん) is answered locally without
  a network round-trip.
- **post_correct on the source** — ASR misrecognition fixes
  (`src/post_correction.py`) run on the Japanese input, not the Vietnamese output.

### Performance

- **Batch parallelism**: `translate_many` fans a drained batch out over a small
  thread pool (`ROUTER_MAX_PARALLEL`), so N sentences cost ~1 round-trip instead
  of N. Measured ~3× faster than sequential on a 6-sentence batch.
- **Fail-fast**: a hard per-request timeout (`ROUTER_TIMEOUT_S`, default 6s) and a
  single retry; a stalled segment is dropped rather than blocking the meeting.

### Configuration (env vars)

| Env | Default | Meaning |
|-----|---------|---------|
| `ZT_TRANSLATOR` | `nllb` | set to `router` to use 9router |
| `ZT_ROUTER_BASE_URL` | `http://127.0.0.1:20128/v1` | gateway base URL |
| `ZT_ROUTER_KEY` | `sk_9router` | bearer token |
| `ZT_ROUTER_MODEL` | `gh/claude-sonnet-4.6` | model id (see `GET /v1/models`) |
| `ZT_ROUTER_TEMPERATURE` | `0.1` | sampling temperature |
| `ZT_ROUTER_MAX_TOKENS` | `180` | per-segment output cap |
| `ZT_ROUTER_TIMEOUT` | `6` | per-request seconds (live = fail fast) |
| `ZT_ROUTER_CONTEXT` | `3` | prior sentences for terminology coherence (0 disables) |
| `ZT_ROUTER_MAX_PARALLEL` | `4` | max concurrent requests per batch |
| `ZT_ROUTER_PROMPT` | (LLM prompt) | override the system prompt |

Host-only env: `ZT_HOST_REAL=1` runs the real audio pipeline (needs audio + ASR
deps); `ZT_HOST_MIC=1` captures the microphone instead of system loopback.

### Configuration via .env

Secrets and tunables are read from a project-root `.env` file (loaded by
`config.py` before any env read; a real environment variable still overrides it).
Copy the template and edit:

```bash
cp .env.example .env      # then set ZT_ROUTER_KEY etc.
```

`.env` is gitignored — the repo ships no credentials, and `config.ROUTER_API_KEY`
defaults to empty so a missing key fails loudly at the gateway rather than
shipping a token. `.env.example` documents every knob.

### Audio input enrichment (better ASR)

`src/audio_enrich.py` conditions each captured block before VAD/ASR to raise
recognition accuracy on real meeting audio. Dependency-free (numpy; uses scipy's
`lfilter` when present, else a fast pure-Python fallback):

- **High-pass filter** (`ZT_AUDIO_HIGHPASS_HZ`, default 80 Hz) removes DC offset
  and sub-speech rumble (fans, hum) that otherwise bias the VAD energy gate.
- **Soft AGC** (`ZT_AUDIO_TARGET_RMS`, default 0.05) gains quiet speakers up to a
  consistent loudness, capped (`ZT_AUDIO_MAX_GAIN`) and floored
  (`ZT_AUDIO_NOISE_FLOOR_RMS`) so silence/room tone is never amplified.
- **Anti-aliased resampling** (`resample_audio`): loopback audio is usually
  48 kHz and must be downsampled to 16 kHz for ASR. Plain linear interpolation
  aliases high frequencies into the speech band (smears consonants); we use
  `soxr` (VHQ) when installed, then `scipy.resample_poly`, then linear as a
  last resort. Install `soxr` for the accuracy win.
- **Per-utterance loudness normalization** (`normalize_utterance`,
  `ZT_AUDIO_UTTERANCE_NORM`): peak-normalizes a whole recognized segment toward
  `ZT_AUDIO_UTTERANCE_PEAK` (0.95) right before ASR — a consistent
  utterance-level loudness helps the recognizer without pumping gain mid-word.

ASR decode tuning (`src/asr.py`): `ZT_ASR_BLANK_PENALTY` (default 1.0) penalizes
the transducer blank symbol to reduce dropped sentence onsets / clipped leading
mora; applied only when the installed sherpa-onnx build accepts it. The VAD
pre-onset collar is 320 ms (`ZT_VAD_PREROLL_MS`) to keep soft onsets.

Toggle the per-block stage with `ZT_AUDIO_ENRICH` (default on). It runs in
`AudioCapture` between resample and enqueue, ~0.5 ms per 200 ms block.

### Conversational / meeting register

The router system prompt (`_ROUTER_DEFAULT_PROMPT` in config.py) is tuned for
spoken meeting dialogue, not written text: it preserves Japanese keigo politeness
with Vietnamese markers (ạ/dạ/vâng), maps 自分/私→tôi (never "bạn"), keeps
back-channels natural (はい→Vâng, うん→Ừ, never "Được"), lets fragments trail off
with "...", and pins emergency-dispatch + IT terminology. Crucially it forbids
commentary/questions so a strong model never slips into assistant mode.

---

## 4. Models on disk

`models/` ships these; the `-Router` path uses only the ASR model:

| Model | Role | Needed for `-Router`? |
|-------|------|-----------------------|
| `reazonspeech-k2-v2` | offline ASR (Japanese) | **yes** (to hear audio) |
| `streaming-zipformer-multi` | streaming ASR | optional (auto-detected) |
| `nllb-200-distilled-600M-ct2-int8` | NLLB translation | only as fallback |
| `qwen2.5-1.5b/3b-instruct` | local LLM translation | no (replaced by 9router) |

`_RealEngine` auto-detects the streaming model: present → streaming ASR, absent →
offline ReazonSpeech. So missing `streaming-zipformer-multi` is not an error.

---

## 5. Troubleshooting

- **`.ps1` parse error "string is missing the terminator"** — caused by non-ASCII
  characters (em-dash `—`, box-drawing `─`) in PowerShell *string literals*: PS 5.1
  reads a BOM-less file in the system ANSI codepage and byte `0x94` of those UTF-8
  sequences decodes to a `"`. Fix: keep `.ps1` strings ASCII-only and save with a
  UTF-8 BOM (both launchers already are).
- **`Streaming ASR model directory not found`** — only happens if something forces
  streaming without the model; `_RealEngine` now auto-falls back to offline ASR.
- **Console spammed with `[0] Speakers ...`** — the device list helper used to
  `print()` every device on each poll; `host_server.list_input_devices` now
  enumerates quietly.
- **Subtitles look like "Xin lỗi, câu nhập có lỗi ASR…"** — a Vietnamese term was
  substituted into a Japanese source, making a hybrid sentence. Only Latin
  replacements are injected now; Vietnamese terms go through the prompt glossary.
- **Subtitles contain commentary / questions / quoted text** (e.g. `"Chưa hiểu." -
  Cần gì? Chi tiết vấn đề?`) — the model is in *assistant mode*, translating then
  adding its own remarks. Two causes: (1) a **weak model** ignores the
  translate-only prompt — `gh/claude-haiku-4.5` returns `部長`→"Vâng, tôi đây ạ"
  and prepends "Vâng/ạ" to most lines even with `ZT_ROUTER_CONTEXT=0`; the
  default is `gh/claude-sonnet-4.6`, which obeys the prompt. (2) `ROUTER_SYSTEM_PROMPT`
  is the verbose local-Qwen prompt (it tells the model to "help" when input looks
  broken) — the default `_ROUTER_DEFAULT_PROMPT` is a terse translate-only prompt;
  override via `ZT_ROUTER_PROMPT` only with an equally strict prompt.
- **Translations slow / stalling** — lower `ZT_ROUTER_TIMEOUT`, set
  `ZT_ROUTER_CONTEXT=0`, or pick a faster model. Check the gateway is reachable:
  `GET http://127.0.0.1:20128/v1/models`.
