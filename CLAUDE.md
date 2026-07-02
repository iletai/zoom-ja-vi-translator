# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Real-time **Japanese → Vietnamese** translator for Zoom meetings. Captures system/meeting
audio, runs streaming speech recognition, translates, and renders Vietnamese subtitles live.

## Run

```bash
python3 main.py --list-devices      # enumerate audio input devices
python3 main.py                      # run the translator (from project root)
./scripts/run.sh                     # convenience launcher (macOS/Linux)
./scripts/run-host.ps1               # web UI launcher (Windows, most common)
```

Always run from the project root so relative imports under `src/` resolve.

All launcher scripts (`run*.ps1`, `run*.sh`, `START-*.bat`) live in `scripts/`.

## Architecture

Multi-threaded pipeline (`src/pipeline.py`) with three decoupled stages connected by
**bounded queues**, keeping the UI responsive and bounding memory:

```
audio_capture → VAD → asr → sentence_aggregator → translator → display
```

### Modules (`src/`)

- **audio_capture.py** — captures meeting audio from the selected input device.
- **asr.py / streaming_asr.py** — speech recognition; streaming ASR emits incremental hypotheses.
- **local_agreement.py** — LocalAgreement stabilization: commits ASR tokens only once
  consecutive hypotheses agree, reducing subtitle flicker.
- **sentence_aggregator.py** — assembles committed tokens into full sentences for translation.
- **cloud_translator.py** — JA→VI translation via a cloud translation API.
- **router_translator.py** — JA→VI via a local OpenAI-compatible gateway ("9router");
  drop-in backend, selected with `ZT_TRANSLATOR=router`. Sends sentences sequentially
  when `ZT_ROUTER_CONTEXT > 0` (each sentence enters history before the next request),
  parallel otherwise. Pure filler/back-channel utterances (はい, なるほど, etc.) are
  short-circuited via an in-process map without a network round-trip.
  See `docs/WEBUI_9ROUTER.md`.
- **display.py** — renders the live Vietnamese subtitles.
- **pipeline.py** — orchestrates the stages and the bounded queues between them.
- **main.py** — CLI entry point (`--list-devices`, device selection, startup).

## Web UI + 9router backend

A browser UI and a 9router translation backend were added on top of the CLI pipeline
(ASR stays local; only translation is offloaded). See `docs/WEBUI_9ROUTER.md` for the
full picture. Key entry points:

- **webui/host_server.py** — stdlib WebSocket host that serves `webui/rd_ui_v1.1.html`
  and bridges it to the pipeline. Run via `scripts/run-host.ps1` (Windows) / `scripts/run-host.sh`.
- **src/router_translator.py** — `RouterTranslator`, the `ZT_TRANSLATOR=router` backend.
- **Windows launcher**: `scripts/run.ps1 -Router` runs the CLI with local ASR + 9router.
- Translation backends: `nllb` (default), `llm` (local Qwen), `router` (9router).
- `.ps1` files must stay ASCII-only in string literals + keep a UTF-8 BOM (PowerShell 5.1
  mis-decodes BOM-less non-ASCII and breaks parsing).

## Conventions

- Pure Python project; keep new modules under `src/` and import them through the pipeline.
- Preserve the bounded-queue, decoupled-stage design — do not block the capture or display
  threads on slow ASR/translation work.
- The capture → ASR → translate → display flow is latency-sensitive; prefer streaming and
  incremental processing over batch.

## Notes

- `__pycache__/` is build output; do not edit.
