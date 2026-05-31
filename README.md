# Zoom JA→VI Translator

Real-time **Japanese → Vietnamese** translator for Zoom (and any meeting/video)
audio. Runs **100% locally and free** using open-source Hugging Face models — no
GPU required, no API keys, no per-minute cost.

- **ASR (Japanese)**: [ReazonSpeech k2-v2](https://github.com/reazon-research/ReazonSpeech)
  via `sherpa-onnx` (int8, CPU). Best-in-class Japanese accuracy (CER ~6.6%, beats Whisper-Large-v3) at only ~154 MB.
- **Translation (JA→VI)**: [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
  via [CTranslate2](https://github.com/OpenNMT/CTranslate2) (int8) — 4–8× faster than PyTorch on CPU.
- **Audio capture**: cross-platform loopback via [`soundcard`](https://github.com/bastibe/SoundCard)
  (Windows WASAPI native, macOS BlackHole, Linux PulseAudio monitor).
- **Display**: terminal subtitles (bilingual JP + VI).

> **License note:** NLLB-200 is **CC-BY-NC 4.0** (non-commercial / personal use only).
> For commercial use, switch the translation model to `Helsinki-NLP/opus-mt-ja-vi`
> (Apache 2.0). See *Performance Tuning* below.

---

## Architecture

```
Zoom audio (Japanese)
   │  (system loopback)
   ▼
[audio_capture] ──16kHz float32──▶ [vad] ──utterances──▶ [asr] ──JP text──▶ [translator] ──VI text──▶ [display]
   capture thread                  silence-based         ReazonSpeech k2     NLLB-600M (CT2)        terminal
                                    segmentation          (sherpa-onnx)
```

Three decoupled threads connected by bounded queues (drop-oldest) keep latency low.
VAD emits an utterance as soon as the speaker pauses, so translation starts
immediately instead of waiting for a fixed-size chunk.

**Expected latency (CPU-only):** ~3–5 s per spoken sentence (near-real-time).
This is the practical floor for local ASR+MT without a GPU.

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4-core (Intel i5 gen 8+, Apple M1, Ryzen 3) | 8-core i7 / M1 Pro+ |
| RAM | 4 GB free | 8 GB |
| Disk | ~1 GB (models + deps) | — |
| GPU | **Not required** | — |
| OS | macOS 12+, Windows 10+, Linux | — |
| Python | 3.9 – 3.12 | 3.11 |

---

## Installation

### Quick start (one command)

```bash
cd zoom-translator
./run.sh                 # macOS / Linux  — sets up venv, installs, downloads, runs
# Windows (PowerShell):
./run.ps1                # creates venv, installs, downloads models, runs
```

The launcher creates the virtualenv, installs CPU-only dependencies, downloads
the models on first run (~1 GB, cached afterwards), and starts capturing system
audio. Subsequent runs start instantly.

> **Apple Silicon (M1/M2/M3):** `run.sh` automatically uses a **native arm64**
> Python and a separate `.venv-arm64`. Running natively instead of under Rosetta 2
> makes local translation about **3× faster** (NLLB MT ~1 s vs ~3 s per sentence)
> at zero cost. If you set things up manually, create the venv with an arm64
> interpreter (e.g. `/usr/bin/python3`) — see below.

### Manual setup

```bash
cd zoom-translator

# 1. Create a virtual environment (Python 3.9-3.12; 3.13+ has no ML wheels yet)
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies (CPU-only)
pip install -r requirements.txt

# 3. Download the models (~1 GB, one-time)
python3 scripts/download_models.py
```

On **Apple Silicon**, build the venv with a native arm64 interpreter instead of
an Intel/Homebrew one (which would run under Rosetta 2 and be ~3× slower):

```bash
# /usr/bin/python3 is a universal binary that runs arm64 natively on M-series.
/usr/bin/python3 -m venv .venv-arm64
source .venv-arm64/bin/activate
python3 -c 'import platform; print(platform.machine())'   # must print: arm64
pip install -r requirements.txt
```

### Platform-specific audio setup

The tool captures the audio **you hear** from Zoom (system output), not your mic.

#### macOS — install BlackHole (required)

macOS has **no built-in system-audio loopback** (the `soundcard` library reports
*"macOS does not support loopback recording"*), so you must install the free
[BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual device. Run
the helper (it installs BlackHole and prints the routing steps):

```bash
./scripts/setup_macos_audio.sh       # asks for your admin password
```

Or do it manually:

```bash
brew install blackhole-2ch
```

Then route Zoom's audio through it so you still hear it **and** the app can capture it:

1. Open **Audio MIDI Setup.app**.
2. Click **＋ → Create Multi-Output Device**.
3. Check both **your speakers/headphones** and **BlackHole 2ch**.
4. In **System Settings → Sound → Output**, select the Multi-Output Device.
5. The app auto-detects BlackHole as the capture source (`--system-audio`).

#### Windows — nothing to install (native parity)

`soundcard` uses **WASAPI loopback** to capture the default speaker directly.
`--system-audio` automatically selects the **loopback of your current default
output device**, so you just make sure Zoom plays through that output. No
VB-Cable needed (though VB-Audio / Voicemeeter / "Stereo Mix" are auto-detected
if present).

#### Linux — PulseAudio / PipeWire

The app captures the PulseAudio **monitor** of your default sink automatically.

---

## Usage

```bash
# List available audio devices (find your loopback index)
python3 main.py --list-devices

# Run, auto-selecting the system loopback device
python3 main.py --system-audio

# Run with a specific device index
python3 main.py --device-index 3
```

Output looks like:

```
[11:32:05]
  🇯🇵 こんにちは、今日の会議を始めます。
  🇻🇳 Xin chào, chúng ta bắt đầu cuộc họp hôm nay.
```

Press **Ctrl+C** to stop.

---

## Low-latency streaming mode (recommended for live meetings)

The default (offline) mode waits for a full pause before transcribing a whole
utterance, so Japanese appears only after the speaker stops. For near-real-time
captions, use **streaming mode**: a streaming Zipformer model emits partial
Japanese text *while* the person is still talking, then finalizes and translates
each sentence at the natural pause.

```bash
# One-time: download the streaming ASR model (~320 MB, includes Japanese)
python3 scripts/download_models.py --streaming

# Run with live streaming captions
python3 main.py --system-audio --streaming
```

Trade-offs:

- **Pro**: Japanese shows up almost instantly (live partial captions), so you
  follow the conversation in real time instead of trailing 3–7 s behind.
- **Con**: Streaming ASR is slightly less accurate than the offline model, and
  the Vietnamese translation still goes through NLLB so it trails the Japanese
  by the machine's MT latency (a few seconds per sentence on CPU).

Test with **real Japanese speech**, not a song — singing has no sentence pauses,
so the endpoint detector never fires and the backlog grows.

Segment length is tuned for accuracy and low lag in `config.py`:
`STREAMING_RULE2_SILENCE` (finalize after a short pause) and
`STREAMING_RULE3_UTTERANCE` (hard cap on a run-on segment, so continuous speech
is never handed to the translator as one giant block — which drops words and
loses context). Lower them for snappier captions, raise them for longer,
more coherent segments.

---

## Cloud backend — lowest latency (optional)

Local CPU translation (NLLB) floors at several seconds per sentence. For
sub-second, near-real-time captions like a streaming subtitle service, use the
optional **Azure Speech Translation** backend: it does Japanese recognition AND
Japanese→Vietnamese translation in a single streaming cloud call (~0.5–1 s).

```bash
# 1. Install the cloud extra
pip install -r requirements-cloud.txt

# 2. Create a free Azure "Speech" resource (F0 tier = 5 audio hours/month free)
#    at https://portal.azure.com, then export its key + region:
export AZURE_SPEECH_KEY=your_key_here
export AZURE_SPEECH_REGION=southeastasia   # your resource's region

# 3. Run with the cloud backend
python3 main.py --system-audio --cloud azure
```

Trade-offs:

- **Pro**: Lowest latency by far (~0.5–1 s end-to-end), best JA→VI quality, no
  local model load, runs on minimal hardware.
- **Con**: Audio is sent to Microsoft Azure (not offline), needs internet, and
  costs ~$2.50/hour after the 5 free hours/month.

Notes:

- **DeepL is not an option** — it does not support Vietnamese.
- **Zoom's own translated captions** do not list Vietnamese as a target either,
  so a cloud ASR+MT backend is required for JA→VI.
- For fully offline / zero-cost use, omit `--cloud` and stay on the local
  backend (optionally with `--streaming`).

---

## Verify in a real Zoom meeting

1. **Start audio routing**
   - **macOS**: set output to the Multi-Output Device (see above) so the meeting
     plays to your speakers *and* BlackHole.
   - **Windows**: just ensure Zoom plays through your default speakers.
2. **Join a Japanese Zoom meeting** (or play any Japanese audio / YouTube to test).
3. **Run the translator** capturing system audio:

   ```bash
   ./run.sh                 # macOS/Linux
   ./run.ps1                # Windows
   # or: python3 main.py --system-audio
   ```

4. Confirm the right device was picked — the startup line prints e.g.
   `Using loopback device: BlackHole 2ch` (macOS) or
   `Using loopback device: Speakers (...)` (Windows).
5. When someone speaks Japanese, a `🇯🇵 … / 🇻🇳 …` subtitle pair appears within a
   few seconds.

**Smoke-test without a meeting** (validates the full pipeline on real Japanese
speech — no microphone needed):

```bash
python3 tests/test_pipeline_stream.py        # see the Testing section
```

---

## Performance Tuning

Edit `config.py`:

| Goal | Change |
|------|--------|
| **Apple Silicon: ~3× faster (free)** | Use a native arm64 venv (`run.sh` does this automatically; manual: build the venv with `/usr/bin/python3`) |
| **Lowest latency (~0.5–1s)** | Run with `--cloud azure` (cloud ASR+MT; see above) |
| **Near-real-time captions** | Run with `--streaming` (live partial Japanese; see above) |
| **Lower latency** | Reduce `VAD_SILENCE_MS` (e.g. 400); set `NLLB_BEAM_SIZE = 1`, or run with `ZT_FAST=1` (beam 2) |
| **Higher translation quality** | `NLLB_BEAM_SIZE = 4` (slower, the default) |
| **Less CPU / RAM** | Use fewer threads (`ASR_NUM_THREADS`, `NLLB_INTRA_THREADS`); defaults now auto-scale to `os.cpu_count()` to avoid oversubscription |
| **Faster but lower JA accuracy** | Swap ASR to faster-whisper `base` (alternative backend) |
| **Commercial license** | Replace NLLB with `Helsinki-NLP/opus-mt-ja-vi` (Apache 2.0, ~75 MB int8). Convert with `ct2-opus-mt-converter` and adjust `translator.py` (opus-mt is bilingual — no language prefix token) |

### Upgrading translation quality (NLLB-1.3B)

For an opt-in quality upgrade, convert Meta's larger NLLB distilled model to the
same CTranslate2 int8 format. It improves JA→VI translation quality by about
+4 chrF++, but uses about 2.0-2.5 GB RAM at runtime and roughly 300-700 ms per
sentence on a 4-core CPU.

```bash
./scripts/convert_nllb_1p3b.sh
```

The translator auto-detects `models/nllb-200-distilled-1.3B-ct2-int8/` on next
launch, so no code or tokenizer change is needed. To revert, remove that
directory or force the 600M model:

```bash
export NLLB_CT2_DIR=models/nllb-200-distilled-600M-ct2-int8
```

---

## Reliability: complete translation & evidence logging

Two safeguards keep a long meeting from silently losing content.

### Sentence-aware translation (no dropped sentences)

NLLB-200 is sentence-trained: handed a block of several Japanese sentences it
translates only the **first** and silently drops the rest (e.g.
`本日の会議を始めます。資料を確認してください。` → only "Tôi sẽ bắt đầu cuộc họp hôm nay.",
losing the second sentence). The translator now follows documented MT best
practice — it segments the source into whole sentences first, translates each in
a single batched CTranslate2 call, then rejoins them — so no sentence is dropped.
Cross-item batching also raises throughput, shrinking the backlog that used to
force the queue to shed recognized speech. Recognized text is now preserved via
backpressure instead of being dropped under load. Disable splitting for
benchmarking with `ZT_NO_SENTENCE_SPLIT=1`.

Because the Japanese streaming ASR emits **no punctuation** and ends segments on
acoustic silence (not grammar), it can glue several turns into one run-on. The
segmenter therefore also splits at polite terminals (です/ます/…) **and at
greeting set-phrases** (こんにちは / ようこそ / おはよう / …) that carry no
terminal, so a greeting run-on like
`皆さんこんにちは…ようこそ皆さんお元気ですか` is separated into three sentences
instead of reaching NLLB as one block (which dropped the trailing
`お元気ですか`). Over-splitting only affects wording, never loses content.

### Evidence logging (debug where data was lost)

For auditing a meeting where "a line went missing", enable a structured JSONL
log of every pipeline stage:

```bash
python3 main.py --system-audio --streaming --log
# or to a specific file / via env:
python3 main.py --system-audio --log run.jsonl
ZT_EVIDENCE_LOG=run.jsonl python3 main.py --system-audio
```

Each recognized utterance carries a monotonic `seq` id so its whole life is
traceable end to end:

```text
asr_final / aggregator_emit → enqueue → translate → display
                                    ↘ dedup_skip / queue drop (loss is logged)
```

Every record includes `seq`, a monotonic timestamp, the thread, and stage-
specific fields (`jp`, `vi`, `latency_ms`, `batch`, `queue_size`). Filtering the
log by a `seq` shows exactly where — if anywhere — a sentence was lost. Logging
is opt-in and a no-op when not configured.

### Saving the transcript

When a session that used `--log` ends (Ctrl+C or normal stop), a human-readable
bilingual transcript is written automatically next to the evidence log:
`run_<ts>.txt` (timestamped JP/VI pairs) and `run_<ts>.srt` (bilingual subtitles
with correct timing derived from each utterance's audio window). Generate any
format from an existing log at any time:

```bash
python3 -m src.transcript_export run.jsonl -f srt -o run.srt
python3 -m src.transcript_export run.jsonl -f md   --stats   # table + run summary
```

Formats: `txt`, `md`, `srt`, `json`.

### Live web dashboard (Streamlit)

For a visual, auto-refreshing view of a running meeting, a Streamlit dashboard
tails the evidence log and renders the live bilingual feed plus run metrics
(segment count, max-cut %, median latency, loss events) with one-click transcript
export. It is fully decoupled — a separate process that only reads the log — so it
adds no latency to and cannot drop data from the translator.

```bash
# 1. Install the web extra (one-time)
pip install -r requirements-web.txt

# 2. Start the translator with logging
python3 main.py --system-audio --log test_audio/evidence/live.jsonl

# 3. In another terminal, open the dashboard and pick that log in the sidebar
streamlit run webui/streamlit_app.py
```

---

```
zoom-translator/
├── requirements.txt
├── requirements-cloud.txt    # optional Azure cloud backend extras
├── requirements-web.txt      # optional Streamlit live dashboard extras
├── config.py                 # all tunable parameters
├── main.py                   # CLI entrypoint
├── run.sh / run.ps1          # one-command launchers (macOS·Linux / Windows)
├── webui/
│   └── streamlit_app.py      # live web dashboard (tails the evidence log)
├── scripts/
│   ├── download_models.py    # downloads ReazonSpeech + NLLB (CT2)
│   └── setup_macos_audio.sh  # installs BlackHole + prints routing steps
├── tests/
│   ├── test_components.py     # ASR/VAD/translation on real JA wavs
│   ├── test_pipeline_stream.py # end-to-end streaming (simulated Zoom)
│   └── test_streaming_pipeline.py # low-latency streaming ASR pipeline
└── src/
    ├── audio_capture.py      # cross-platform loopback capture
    ├── vad.py                # silence-based utterance segmentation
    ├── asr.py                # ReazonSpeech k2 Japanese ASR
    ├── streaming_asr.py      # streaming Zipformer JA ASR (--streaming)
    ├── cloud_translator.py   # Azure Speech Translation backend (--cloud azure)
    ├── translator.py         # NLLB-600M CTranslate2 JA→VI (sentence-aware, batched)
    ├── sentence_aggregator.py # JP sentence segmentation (shared by translator)
    ├── evidence_log.py       # opt-in JSONL per-stage logging (--log)
    ├── transcript_export.py  # build/save transcript (txt·md·srt·json) from a log
    ├── pipeline.py           # multi-threaded orchestration
    └── display.py            # terminal subtitle output
```

---

## Testing

Two scripts under `tests/` validate the real models against actual Japanese
speech (no microphone required):

```bash
# 1. Fetch the sherpa-onnx ReazonSpeech sample wavs (real Japanese utterances)
mkdir -p test_audio && cd test_audio
curl -L -o m.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
tar xjf m.tar.bz2 '*/test_wavs/*' && cd ..

# 2. Component test: ASR + VAD + NLLB translation on each wav
python3 tests/test_components.py

# 3. End-to-end streaming test: simulates a live Zoom meeting by feeding the
#    wavs into the pipeline in 0.2 s blocks (VAD → ASR → translate → display).
python3 tests/test_pipeline_stream.py          # real-time pacing
python3 tests/test_pipeline_stream.py --fast   # as fast as the CPU allows
```

Both tests print `RESULT: PASS` and exit `0` on success. They exercise the same
threading, queues and shutdown path used by `main.py`, so a green run means the
live capture path differs only in its audio source.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No audio captured on macOS | Confirm Multi-Output Device is the system output and BlackHole is checked; grant **Microphone** permission to your terminal |
| `FileNotFoundError` for models | Run `python3 scripts/download_models.py` |
| High latency | Lower `VAD_SILENCE_MS` and `NLLB_BEAM_SIZE`; close other CPU-heavy apps |
| Garbled / empty Japanese | Ensure the speaker audio is clear; ReazonSpeech expects 16 kHz mono (handled automatically) |
| Permission denied (screen/mic) | Grant Microphone (macOS) permission in System Settings → Privacy |

---

## Credits

Built on open-source work: ReazonSpeech (reazon-research), sherpa-onnx (k2-fsa),
NLLB-200 (Meta AI), CTranslate2 (OpenNMT), SoundCard (bastibe). Architecture
informed by `magicpro97/tui-translator`, `kizuna-ai-lab/sokuji`, and
`Vanyoo/realtime-subtitle`.
