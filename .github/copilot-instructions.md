# Copilot Instructions

## Project

Real-time **Japanese → Vietnamese** translator for Zoom meetings. Captures system audio via loopback, runs streaming speech recognition (ReazonSpeech k2 via sherpa-onnx), translates (NLLB-200 via CTranslate2 or optional Azure cloud backend), and renders bilingual subtitles live. Runs 100% locally on CPU by default.

## Run

```bash
python main.py --list-devices       # enumerate audio input devices
python main.py --system-audio       # auto-select system loopback
python main.py --system-audio --streaming   # low-latency live captions
python main.py --system-audio --cloud azure # cloud backend (fastest)
```

Always run from the project root so `src/` relative imports resolve.

## Testing

Tests live in `tests/` and exercise real models against actual Japanese speech WAVs (no mocking). No test framework (raw scripts, not pytest):

```bash
# Run a single test
python tests/test_components.py
python tests/test_pipeline_stream.py --fast

# All tests print "RESULT: PASS" and exit 0 on success
```

Test WAVs must be downloaded first (see README "Testing" section). Tests use `sys.path.insert` to resolve the project root.

## Architecture

Multi-threaded pipeline (`src/pipeline.py`) with decoupled stages connected by **bounded queues**:

```
audio_capture → VAD → ASR → sentence_aggregator → translator → display
                              ↓ (streaming mode)
                         online ASR (live partial) + offline re-decode (accurate final)
```

Key design invariants:
- **Never block capture or display** on slow ASR/translation work — queues decouple stages.
- **Never drop recognized text** — the text queue uses backpressure (not drop-oldest) because dropping = permanent content loss.
- Audio queue uses drop-oldest (acceptable — raw audio can be regenerated from ongoing capture).
- Translation is **sentence-aware**: NLLB silently drops sentences when given multi-sentence input, so text is segmented into individual sentences before translation.

### Streaming mode (`--streaming`)

Uses a hybrid approach: online Zipformer drives the live partial display, but finalized segments are **re-decoded by the stronger offline model** before being translated (configurable via `ZT_NO_REDECODE=1`).

## Key Modules (`src/`)

| Module | Role |
|--------|------|
| `pipeline.py` | Orchestrates threads + queues |
| `audio_capture.py` | Cross-platform loopback (WASAPI / BlackHole / PulseAudio) |
| `vad.py` | WebRTC VAD with pre-roll collar and max-utterance cut |
| `asr.py` | Offline ReazonSpeech k2 recognition |
| `streaming_asr.py` | Online Zipformer with endpoint detection |
| `local_agreement.py` | Commits tokens only after N consecutive hypotheses agree (reduces flicker) |
| `sentence_aggregator.py` | Reassembles ASR fragments into grammatical sentences |
| `translator.py` | NLLB CTranslate2 with sentence splitting + batching |
| `llm_translator.py` | Optional Qwen2.5 LLM translation via llama.cpp |
| `cloud_translator.py` | Azure Speech Translation backend |
| `evidence_log.py` | Opt-in JSONL logging for debugging pipeline losses |
| `display.py` | Terminal subtitle rendering |

## Configuration

All tunable parameters live in `config.py` (not scattered across modules). Environment variable overrides use `ZT_` prefix. Key knobs:

- `VAD_SILENCE_MS` / `VAD_MAX_UTTERANCE_MS` — segmentation timing
- `STREAMING_RULE2_SILENCE` / `STREAMING_RULE3_UTTERANCE` — streaming endpoint detection
- `NLLB_BEAM_SIZE` — translation quality vs speed (4 default, or 2 with `ZT_FAST=1`)
- `ASR_NUM_THREADS` / `NLLB_INTRA_THREADS` — CPU thread budget (auto-scales to physical cores)

## Conventions

- Pure Python; new modules go under `src/` and integrate through the pipeline.
- Preserve the bounded-queue, decoupled-stage threading model.
- Prefer streaming/incremental processing over batch — the pipeline is latency-sensitive.
- `config.py` is the single source of truth for parameters; modules import from it.
- Tests are standalone scripts (not pytest), print `RESULT: PASS`/`FAIL`, and exit with appropriate codes.
- Python 3.9–3.12 (3.13+ lacks ML wheel support).

## Dependencies

```bash
pip install -r requirements.txt           # core (CPU-only)
pip install -r requirements-cloud.txt     # Azure cloud backend
pip install -r requirements-web.txt       # Streamlit live dashboard
```

Models (~1 GB) are downloaded separately:
```bash
python scripts/download_models.py             # offline ASR + NLLB
python scripts/download_models.py --streaming # streaming ASR model
```
