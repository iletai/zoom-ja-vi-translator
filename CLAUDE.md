# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Real-time **Japanese → Vietnamese** translator for Zoom meetings. Captures system/meeting
audio, runs streaming speech recognition, translates, and renders Vietnamese subtitles live.

## Run

```bash
python3 main.py --list-devices      # enumerate audio input devices
python3 main.py                      # run the translator (from project root)
./run.sh                             # convenience launcher
```

Always run from the project root so relative imports under `src/` resolve.

## Architecture

Multi-threaded pipeline (`src/pipeline.py`) with three decoupled stages connected by
**bounded queues**, keeping the UI responsive and bounding memory:

```
audio_capture → VAD → streaming_asr → sentence_aggregator → cloud_translator → display
```

### Modules (`src/`)

- **audio_capture.py** — captures meeting audio from the selected input device.
- **asr.py / streaming_asr.py** — speech recognition; streaming ASR emits incremental hypotheses.
- **local_agreement.py** — LocalAgreement stabilization: commits ASR tokens only once
  consecutive hypotheses agree, reducing subtitle flicker.
- **sentence_aggregator.py** — assembles committed tokens into full sentences for translation.
- **cloud_translator.py** — JA→VI translation via a cloud translation API.
- **display.py** — renders the live Vietnamese subtitles.
- **pipeline.py** — orchestrates the stages and the bounded queues between them.
- **main.py** — CLI entry point (`--list-devices`, device selection, startup).

## Conventions

- Pure Python project; keep new modules under `src/` and import them through the pipeline.
- Preserve the bounded-queue, decoupled-stage design — do not block the capture or display
  threads on slow ASR/translation work.
- The capture → ASR → translate → display flow is latency-sensitive; prefer streaming and
  incremental processing over batch.

## Notes

- `__pycache__/` is build output; do not edit.
