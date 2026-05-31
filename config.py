"""Central configuration for the Zoom Japanese->Vietnamese translator.

All tunable parameters live here so modules stay decoupled and the pipeline
can be adjusted without touching implementation code.
"""
from __future__ import annotations

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ASR_MODEL_DIR = MODELS_DIR / "reazonspeech-k2-v2"
# Translation model: prefer the higher-quality NLLB-200-distilled-1.3B int8 build
# when it is present (drop-in upgrade, ~+4 chrF++, same tokenizer/API). Falls back
# to the 600M build otherwise. Convert the 1.3B model with
# scripts/convert_nllb_1p3b.sh. Override either path via env if desired.
_NLLB_CT2_1P3B = MODELS_DIR / "nllb-200-distilled-1.3B-ct2-int8"
_NLLB_CT2_600M = MODELS_DIR / "nllb-200-distilled-600M-ct2-int8"
NLLB_CT2_DIR = Path(
    os.environ.get(
        "NLLB_CT2_DIR",
        str(_NLLB_CT2_1P3B if _NLLB_CT2_1P3B.exists() else _NLLB_CT2_600M),
    )
)

# ─── HuggingFace Hub offline mode ────────────────────────────────────────
# Loading the NLLB tokenizer via from_pretrained contacts the HF Hub to check
# for updated files on EVERY launch. Unauthenticated (no HF_TOKEN), those calls
# are rate-limited and can stall ~80s before falling back to the cache — the
# single biggest cause of slow startup. Once the models are downloaded we never
# need the network, so enable HF offline mode (cache loads in <1s). These env
# vars are read by huggingface_hub at import time, so they MUST be set here in
# config (imported before transformers) to take effect.
# Force online with ZT_HF_ONLINE=1 (e.g. the first-time tokenizer download).
HF_OFFLINE = (
    not os.environ.get("ZT_HF_ONLINE")
    and NLLB_CT2_DIR.exists()
)
if HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ─── Audio ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz — required by both ReazonSpeech and VAD
CHANNELS = 1                  # mono
CAPTURE_BLOCK_SECONDS = 0.2   # size of each captured block before queueing
CAPTURE_QUEUE_MAXSIZE = 64    # drop-oldest beyond this to bound latency

# ─── VAD (Voice Activity Detection) ──────────────────────────────────────
# webrtcvad aggressiveness: 0 (least) .. 3 (most aggressive at filtering non-speech)
VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 30             # webrtcvad supports 10 / 20 / 30 ms frames
# End an utterance after this much trailing silence (lower = lower latency).
VAD_SILENCE_MS = 450
# Ignore utterances shorter / longer than these bounds. The upper bound also
# force-flushes long continuous speech so it is transcribed and translated in
# bounded chunks instead of waiting for a pause — this keeps latency predictable
# (translation cost grows with input length) during a fast-talking meeting.
VAD_MIN_UTTERANCE_MS = 300
VAD_MAX_UTTERANCE_MS = 7_000

# ─── ASR (ReazonSpeech k2 via sherpa-onnx) ───────────────────────────────
ASR_NUM_THREADS = int(os.environ.get("ASR_NUM_THREADS", "4"))
ASR_PROVIDER = "cpu"

# ─── Streaming ASR (online zipformer, opt-in via --streaming) ─────────────
# Multilingual streaming zipformer (incl. Japanese). Emits partial hypotheses
# as audio arrives, so the recognized text appears almost immediately instead of
# waiting for an end-of-utterance pause. Trades a little accuracy for latency.
STREAMING_ASR_MODEL_DIR = MODELS_DIR / "streaming-zipformer-multi"
STREAMING_ASR_NUM_THREADS = int(os.environ.get("STREAMING_ASR_NUM_THREADS", "4"))
# Endpoint detection (seconds). Lower rule2 = the recognizer finalizes a segment
# sooner after a brief pause, so translation starts earlier. rule3 caps a run-on:
# without natural pauses (fast/continuous speech) it forces a boundary so the
# translator never receives a giant multi-sentence block (which drops words and
# loses context). Keep it short enough for accurate, low-lag translation.
STREAMING_RULE1_SILENCE = 2.4    # finalize after this silence even with no decode
STREAMING_RULE2_SILENCE = 0.85   # finalize after this silence once words decoded
STREAMING_RULE3_UTTERANCE = 7.0  # force a segment boundary after this length

# Audio-overlap window (seconds). Streaming endpoint resets start the next segment
# from acoustic silence, so the recognizer loses context and drops sentence heads
# (e.g. 「部長三日の…」 becomes 「がどうしたんですか」). On reset we re-prime the new
# stream with this much trailing audio from the previous segment so the acoustic
# model keeps context across the boundary. 0 disables the overlap. ~0.6s is the
# sweet spot: enough to recover the head without re-decoding a whole prior clause.
STREAMING_AUDIO_OVERLAP_SEC = 0.6

# LocalAgreement-2 commit policy for the live partial. A character/word is only
# "committed" (shown solid, never rewritten) once it has appeared unchanged in
# this many consecutive partial hypotheses. The still-volatile tail is shown dim.
# This removes the flicker/stale-fragment garbage of re-rendering the full
# hypothesis every chunk. 1 disables it (commit everything immediately).
STREAMING_LOCAL_AGREEMENT_N = 2

# ─── Sentence aggregation (streaming finals -> well-formed sentences) ─────
# The online recognizer's endpoints fall on acoustic pauses, not grammatical
# boundaries: it cuts mid-word (変|更), drops sentence heads, and merges several
# speaker turns into one run-on. Feeding those fragments straight to NLLB (which
# is sentence-trained) produces dropped words and mistranslations. A small
# aggregation layer re-joins consecutive fragments and re-splits them at Japanese
# sentence-final boundaries so the translator receives whole sentences.
# Force-flush guards keep latency bounded when no clean boundary ever appears.
STREAM_SENTENCE_MAX_CHARS = 60       # flush a pending buffer once it grows past this
STREAM_SENTENCE_MAX_WAIT_SEC = 1.5   # flush a pending buffer after this idle time

# Consecutive-duplicate suppression. Streaming endpoints frequently re-emit the
# exact same finalized sentence (the recognizer replays a buffered utterance after
# a reset), producing duplicated JP/VI lines in the output. Skip translating &
# displaying a sentence that is identical to the immediately-previous one when it
# is at least this many characters long and arrives within this time window. Short
# back-channel words (はい/ええ) are exempt so legitimate repeats still show.
STREAM_DEDUP_MIN_CHARS = 6
STREAM_DEDUP_WINDOW_SEC = 30.0

# ─── Translation (NLLB-600M via CTranslate2) ─────────────────────────────
NLLB_HF_MODEL = "facebook/nllb-200-distilled-600M"   # for tokenizer
NLLB_SOURCE_LANG = "jpn_Jpan"   # Japanese (Kanji + Kana)
NLLB_TARGET_LANG = "vie_Latn"   # Vietnamese (Latin script)
NLLB_BEAM_SIZE = int(os.environ.get("NLLB_BEAM_SIZE", "4"))  # beam search: +1-3 BLEU vs greedy
NLLB_INTER_THREADS = 1
NLLB_INTRA_THREADS = int(os.environ.get("NLLB_INTRA_THREADS", "4"))
NLLB_COMPUTE_TYPE = "int8"
NLLB_MAX_INPUT_LENGTH = 512
NLLB_MAX_DECODING_LENGTH = 256
# Anti-repetition / quality knobs applied to CTranslate2 decoding. These fix the
# observed "Tôi xin xin" 2-gram loops, dropped trailing words, and empty outputs:
#   - no_repeat_ngram_size: forbid repeating any n-gram of this size (kills loops)
#   - repetition_penalty: >1.0 discourages re-emitting recent tokens
#   - min_decoding_length: force at least this many target tokens (avoids "")
NLLB_NO_REPEAT_NGRAM_SIZE = 2
NLLB_REPETITION_PENALTY = 1.1
NLLB_MIN_DECODING_LENGTH = 2

# Domain glossary: NLLB-600M renders some proper nouns / loanwords badly
# (新幹線 -> "đường cao tốc", 箱根 -> "đáy hộp", 北海道 -> "Bắc Hải"). We replace
# the Japanese term in the SOURCE text with a Latin rendering, which NLLB copies
# through reliably. Verified per-entry against the real model; only add an entry
# after confirming it improves output (see test_audio/evidence/).
NLLB_GLOSSARY = {
    "新幹線": "tàu Shinkansen",
    "北海道": "tỉnh Hokkaido",
    "箱根": "Hakone",
}

# Pre-converted CTranslate2 NLLB model to download if local convert is skipped.
NLLB_CT2_HF_REPO = "entai2965/nllb-200-distilled-600M-ctranslate2"

# ─── Cloud backend (optional, --cloud) ──────────────────────────────────
# Optional low-latency backend that streams audio to Azure Speech Translation
# (single JA->VI streaming call, ~0.5-1s latency, 5 audio hours/month free).
# Requires credentials via environment variables and the cloud extras
# (pip install -r requirements-cloud.txt). Audio is sent to Microsoft Azure;
# use the local backend for fully offline operation.
CLOUD_PROVIDER = os.environ.get("CLOUD_PROVIDER", "azure")
CLOUD_SOURCE_LANG = os.environ.get("CLOUD_SOURCE_LANG", "ja-JP")   # Azure BCP-47
CLOUD_TARGET_LANG = os.environ.get("CLOUD_TARGET_LANG", "vi")      # Azure target code
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")

# ─── Display ─────────────────────────────────────────────────────────────
USE_COLOR = True

# ─── Shutdown ────────────────────────────────────────────────────────────
# Generous join timeout so an in-flight native ASR/translation call can finish
# before the process exits (a native call interrupted by interpreter teardown
# can segfault). The worker still returns as soon as the current item is done.
WORKER_SHUTDOWN_TIMEOUT = 30.0
