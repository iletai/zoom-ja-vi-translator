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

```bash
cd zoom-translator

# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies (CPU-only)
pip install -r requirements.txt

# 3. Download the models (~1 GB, one-time)
python3 scripts/download_models.py
```

### Platform-specific audio setup

The tool captures the audio **you hear** from Zoom (system output), not your mic.

#### macOS — install BlackHole

macOS has no built-in system-audio loopback, so install the free
[BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual device:

```bash
brew install blackhole-2ch
```

Then route Zoom's audio through it so you still hear it **and** the app can capture it:

1. Open **Audio MIDI Setup.app**.
2. Click **＋ → Create Multi-Output Device**.
3. Check both **your speakers/headphones** and **BlackHole 2ch**.
4. In **System Settings → Sound → Output**, select the Multi-Output Device.
5. The app auto-detects BlackHole as the capture source.

#### Windows — nothing to install

`soundcard` uses **WASAPI loopback** to capture the default speaker directly.
Just make sure Zoom plays through your default output device.

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

## Performance Tuning

Edit `config.py`:

| Goal | Change |
|------|--------|
| **Lower latency** | Reduce `VAD_SILENCE_MS` (e.g. 400); set `NLLB_BEAM_SIZE = 1` |
| **Higher translation quality** | `NLLB_BEAM_SIZE = 4` (slower) |
| **Less CPU / RAM** | Use fewer threads (`ASR_NUM_THREADS`, `NLLB_INTRA_THREADS`) |
| **Faster but lower JA accuracy** | Swap ASR to faster-whisper `base` (alternative backend) |
| **Commercial license** | Replace NLLB with `Helsinki-NLP/opus-mt-ja-vi` (Apache 2.0, ~75 MB int8). Convert with `ct2-opus-mt-converter` and adjust `translator.py` (opus-mt is bilingual — no language prefix token) |

---

## Project Structure

```
zoom-translator/
├── requirements.txt
├── config.py                 # all tunable parameters
├── main.py                   # CLI entrypoint
├── scripts/
│   └── download_models.py    # downloads ReazonSpeech + NLLB (CT2)
├── tests/
│   ├── test_components.py     # ASR/VAD/translation on real JA wavs
│   └── test_pipeline_stream.py # end-to-end streaming (simulated Zoom)
└── src/
    ├── audio_capture.py      # cross-platform loopback capture
    ├── vad.py                # silence-based utterance segmentation
    ├── asr.py                # ReazonSpeech k2 Japanese ASR
    ├── translator.py         # NLLB-600M CTranslate2 JA→VI
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
