"""Central configuration for the Zoom Japanese->Vietnamese translator.

All tunable parameters live here so modules stay decoupled and the pipeline
can be adjusted without touching implementation code.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    """True when env var ``name`` is set to an affirmative value.

    Using ``not os.environ.get(name)`` is wrong: it treats "0"/"false" as set
    (non-empty string is truthy), so ``ZT_NO_SENTENCE_SPLIT=0`` would disable
    splitting and ``ZT_HF_ONLINE=0`` would force online mode — the opposite of
    the user's intent. Accept only the usual affirmative spellings.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

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
    not _env_flag("ZT_HF_ONLINE")
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
# Recognized-text queue between ASR and translation. Unlike raw audio (which
# cannot block capture indefinitely), text here is already-recognized speech —
# dropping it permanently loses meeting content, the user's exact complaint. So
# it is generously sized and the producer applies backpressure instead of
# dropping (see pipeline). Kept large so backpressure is a rare last resort.
TEXT_QUEUE_MAXSIZE = int(os.environ.get("ZT_TEXT_QUEUE_MAXSIZE", "256"))

# ─── VAD (Voice Activity Detection) ──────────────────────────────────────
# webrtcvad aggressiveness: 0 (least) .. 3 (most aggressive at filtering non-speech).
# Default 3 so the silence endpoint fires on live system/mic capture of continuous
# podcasts/meetings — at 2 a constant background keeps every frame "voiced" and each
# utterance is force-cut at the 7s max (evidence run_20260531_144156: 19/23 segments
# = 7.02s), cramming several turns into one chunk that the offline ASR then partially
# drops. The leading-mora clipping that 3 alone caused (気象庁 -> 町長) is repaired by
# the pre-roll collar below. The segment_finalized evidence event logs reason=
# "max_utterance" vs "silence" so the segmentation regime is visible per run.
VAD_AGGRESSIVENESS = int(os.environ.get("ZT_VAD_AGGRESSIVENESS", "3"))
# Pre-onset collar: prepend this much audio before each detected speech onset so
# aggressiveness=3 (which fires natural boundaries in continuous/background audio)
# does not clip the quiet leading mora of the first word (気象庁 -> 町長 without it).
VAD_PREROLL_MS = int(os.environ.get("ZT_VAD_PREROLL_MS", "240"))
VAD_FRAME_MS = 30             # webrtcvad supports 10 / 20 / 30 ms frames
# End an utterance after this much trailing silence (lower = lower latency).
# 300ms separates fast back-and-forth dialog turns so two speakers are not merged
# into one segment (which the offline model would then transcribe as run-on
# speech); env-tunable for quieter single-speaker setups.
VAD_SILENCE_MS = int(os.environ.get("ZT_VAD_SILENCE_MS", "300"))
# Ignore utterances shorter / longer than these bounds. The upper bound also
# force-flushes long continuous speech so it is transcribed and translated in
# bounded chunks instead of waiting for a pause — this keeps latency predictable
# (translation cost grows with input length) during a fast-talking meeting.
# 120ms lower bound keeps short Japanese back-channels (はい / うん / ああ) instead
# of dropping them before they reach the offline model.
VAD_MIN_UTTERANCE_MS = int(os.environ.get("ZT_VAD_MIN_MS", "120"))
VAD_MAX_UTTERANCE_MS = int(os.environ.get("ZT_VAD_MAX_MS", "7000"))
# Optional RMS gate: OFF by default. RMS is compared in float32 scale (0..1),
# so the static PCM-16 margin is converted in VadSegmenter before use.
VAD_ENERGY_GATE = _env_flag("ZT_VAD_ENERGY_GATE", False)
VAD_ENERGY_NOISE_ALPHA = 0.10
VAD_ENERGY_MULTIPLIER = 1.8
VAD_ENERGY_MARGIN_RMS = 120.0

# ─── CPU thread budget (latency) ─────────────────────────────────────────
# The offline re-decode ASR pool and the NLLB translator pool each spin up their
# own CTranslate2/sherpa-onnx threads and are both active on the hot path; with
# --streaming a third (online ASR) pool joins them. Hardcoding 4 threads per pool
# oversubscribes anything smaller than an 8-core machine, and context-switch
# thrash *raises* tail latency.
#
# Evidence-based allocation (CTranslate2 perf docs + sherpa-onnx ReazonSpeech
# example, which uses num_threads=2): the int8 Zipformer encoder is memory-
# bandwidth bound, so ASR gains plateau past ~2 threads. Start ASR at 2 threads,
# then scale modestly on larger CPUs as physical_cores//4, capped at 4. NLLB gets
# the remaining budget after reserving for offline re-decode ASR plus streaming
# ASR (when --streaming is active); in non-streaming mode this is conservative but
# avoids oversubscribing the three-pool streaming hot path. The NLLB floor remains
# 2, so very small CPUs may still exceed the budget slightly. Apple Silicon has no
# SMT so os.cpu_count() == physical; on x86 with hyperthreading, physical ≈
# count//2. Explicit env vars still win for hand-tuning.
import platform as _platform

_LOGICAL_CORES = os.cpu_count() or 4
_IS_X86 = _platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")
_PHYSICAL_CORES = max(2, _LOGICAL_CORES // 2) if _IS_X86 else _LOGICAL_CORES
# ASR: 2 on typical 8-core machines; scales to 3-4 only on bigger CPUs.
_ASR_THREADS_DEFAULT = max(2, min(4, _PHYSICAL_CORES // 4))
# NLLB: remaining physical cores after offline + streaming ASR reservations.
_NLLB_THREADS_DEFAULT = max(2, _PHYSICAL_CORES - 2 * _ASR_THREADS_DEFAULT)

# Opt-in low-latency profile (ZT_FAST=1): trades a sliver of MT quality for speed
# by using beam_size 2 instead of 4 (~1.6-1.9x faster NLLB decode, negligible
# chrF++ change on the short sentences this pipeline feeds it). The default keeps
# beam 4 because accuracy is the stated priority; flip ZT_FAST=1 on a slow CPU.
FAST_PROFILE = _env_flag("ZT_FAST", False)

# ─── ASR (ReazonSpeech k2 via sherpa-onnx) ───────────────────────────────
ASR_NUM_THREADS = int(os.environ.get("ASR_NUM_THREADS", str(_ASR_THREADS_DEFAULT)))
ASR_PROVIDER = "cpu"

# ─── ASR Hotwords (boost IT/domain vocabulary) ────────────────────────────
# Path to hotwords file (one term per line, optional :score suffix).
# Enables modified_beam_search decoding which biases the transducer toward
# known domain terms. Set to "" or non-existent path to disable.
ASR_HOTWORDS_FILE = Path(
    os.environ.get("ZT_HOTWORDS_FILE", str(PROJECT_ROOT / "hotwords_it.txt"))
)
ASR_HOTWORDS_SCORE = float(os.environ.get("ZT_HOTWORDS_SCORE", "1.5"))
ASR_DECODING_METHOD = os.environ.get(
    "ZT_ASR_DECODING",
    "modified_beam_search" if ASR_HOTWORDS_FILE.is_file() else "greedy_search"
)

# ─── Streaming ASR (online zipformer, opt-in via --streaming) ─────────────
# Multilingual streaming zipformer (incl. Japanese). Emits partial hypotheses
# as audio arrives, so the recognized text appears almost immediately instead of
# waiting for an end-of-utterance pause. Trades a little accuracy for latency.
STREAMING_ASR_MODEL_DIR = MODELS_DIR / "streaming-zipformer-multi"
STREAMING_ASR_NUM_THREADS = int(
    os.environ.get("STREAMING_ASR_NUM_THREADS", str(_ASR_THREADS_DEFAULT))
)
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

# ─── Hybrid streaming: online partials + offline re-decode (accuracy fix) ──
# The online streaming zipformer is great for low-latency live captions but is
# acoustically weaker than the offline ReazonSpeech model and loses content at
# segment boundaries: it drops sentence onsets after an endpoint reset
# (「気象庁は…」 -> 「Jは…」, 「すぐに」 -> 「ぐに」) and re-decodes its overlap window
# into duplicates (「皆さん皆さん」, 「ともとともに」). Empirically the offline model
# transcribes the same audio with zero loss. Best practice for accurate streaming
# ASR (whisper_streaming / sherpa VAD examples) is therefore: drive the live
# partial display from the online model, but produce the text that is actually
# TRANSLATED by re-decoding each completed utterance's raw audio with the strong
# offline model. The online model never feeds the translator in this mode.
#   ZT_NO_REDECODE=1 falls back to the old online-only streaming behaviour.
STREAMING_REDECODE_OFFLINE = not _env_flag("ZT_NO_REDECODE")

# Bound the finalized-utterance hand-off queue (online worker -> offline
# re-decode worker). Utterances arrive at roughly speech-turn rate (a few per
# 10 s), so this is generous headroom; if offline decode ever falls behind under
# CPU overload the online worker blocks here (logged as segment_queue_backpressure)
# rather than dropping already-captured audio.
SEGMENT_QUEUE_MAX = 64

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
STREAM_DEDUP_ENABLED = not _env_flag("ZT_NO_STREAM_DEDUP")
STREAM_DEDUP_MIN_CHARS = 6
STREAM_DEDUP_WINDOW_SEC = 30.0

# ─── Offline ASR sentence aggregation (re-decode + non-streaming paths) ───
# The offline VAD endpoints fall on acoustic pauses, not grammatical sentence
# boundaries (evidence run_20260531_215444: 141/143 segments cut on silence,
# median 7 chars, 26% <=3 chars). Feeding those sub-sentence fragments straight to
# NLLB — which is sentence-trained — produces hallucinated/wrong Vietnamese (主に
# "mainly" -> "Thậm chí"; one spoken sentence split across 3 VAD segments and
# translated as 3 disconnected fragments). The SAME SentenceAggregator the
# streaming path uses is therefore applied to the offline ASR output too:
# consecutive fragments are re-joined and re-split at Japanese sentence-final
# boundaries so the translator receives whole sentences with context. Set
# ZT_NO_OFFLINE_AGGREGATE=1 to fall back to per-segment translation for
# debugging/benchmarking.
OFFLINE_AGGREGATE_SENTENCES = not _env_flag("ZT_NO_OFFLINE_AGGREGATE")
# Idle time after the last recognized fragment before a still-incomplete buffer is
# force-flushed, so a sentence that never reaches a clean terminal (or a final
# back-channel) is still translated. This is deliberately generous: evidence
# run_20260531_215444 shows consecutive fragments of ONE spoken sentence arrive
# ~2s apart (median inter-fragment gap 2.57s) because each fragment takes that long
# to speak — a short window would flush every fragment on its own and defeat the
# aggregation. Completed sentences are emitted immediately on their grammatical
# terminal (です/ます/。), so this idle cap only adds latency to a genuinely
# trailing/incomplete tail. Lower it for fast single-speaker back-and-forth where
# turn-merging is a concern.
OFFLINE_SENTENCE_MAX_WAIT_SEC = float(
    os.environ.get("ZT_OFFLINE_SENTENCE_MAX_WAIT_SEC", "2.5")
)

# ─── Translator backend selection ─────────────────────────────────────────
# "nllb" (default): NLLB-600M/1.3B via CTranslate2 — fast, no context
# "llm": Qwen2.5-3B via llama-cpp-python — context-aware, better IT quality
TRANSLATOR_BACKEND = os.environ.get("ZT_TRANSLATOR", "nllb").lower()

# ─── Translation (NLLB-600M via CTranslate2) ─────────────────────────────
NLLB_HF_MODEL = "facebook/nllb-200-distilled-600M"   # for tokenizer
# Hugging Face Hub revision used when fetching the tokenizer / model. Pinning a
# revision makes downloads reproducible and avoids silently picking up upstream
# changes. "main" tracks the latest commit; pin a full commit SHA for a fully
# reproducible, tamper-evident download.
NLLB_HF_REVISION = os.environ.get("NLLB_HF_REVISION", "main")
NLLB_SOURCE_LANG = "jpn_Jpan"   # Japanese (Kanji + Kana)
NLLB_TARGET_LANG = "vie_Latn"   # Vietnamese (Latin script)
NLLB_BEAM_SIZE = int(
    os.environ.get("NLLB_BEAM_SIZE", "2" if FAST_PROFILE else "4")
)  # beam search: +1-3 BLEU vs greedy; ZT_FAST lowers to 2 for ~1.7x faster decode
NLLB_INTER_THREADS = 1
NLLB_INTRA_THREADS = int(
    os.environ.get("NLLB_INTRA_THREADS", str(_NLLB_THREADS_DEFAULT))
)
NLLB_COMPUTE_TYPE = "int8"
NLLB_MAX_INPUT_LENGTH = 512
# 350 (target tokens) is a safety cap, not a target: Vietnamese is more verbose
# than Japanese, so long compound sentences could clip at the old 256.
NLLB_MAX_DECODING_LENGTH = 350
# Anti-repetition / quality knobs applied to CTranslate2 decoding. These fix the
# observed "Tôi xin xin" short loops, dropped trailing words, and empty outputs:
#   - no_repeat_ngram_size: forbid repeating any n-gram of this size without
#     blocking legitimate repeated bigrams such as Vietnamese reduplication
#   - repetition_penalty: >1.0 discourages re-emitting recent tokens
#   - min_decoding_length: force at least this many target tokens (avoids "")
NLLB_NO_REPEAT_NGRAM_SIZE = 3
NLLB_REPETITION_PENALTY = 1.1
# 1: with accurate offline ASR, short back-channels (はい→"Vâng") no longer need
# padding to a 2-token minimum (which previously forced filler like "Vâng vâng").
NLLB_MIN_DECODING_LENGTH = 1

# Segment a multi-sentence Japanese block into single sentences before
# translating. NLLB silently drops trailing sentences when given more than one
# at once (verified), so this is on by default; set ZT_NO_SENTENCE_SPLIT=1 to
# fall back to the legacy single-sequence behaviour for debugging/benchmarking.
TRANSLATE_SPLIT_SENTENCES = not _env_flag("ZT_NO_SENTENCE_SPLIT")
# Max recognized items the translate worker drains and translates in one batch.
# Cross-item batching raises throughput so a fast meeting does not build the
# backlog that forces the text queue to shed data.
TRANSLATE_MAX_BATCH = int(os.environ.get("TRANSLATE_MAX_BATCH", "8"))

# Domain glossary: NLLB-600M renders some proper nouns / loanwords badly
# (新幹線 -> "đường cao tốc", 箱根 -> "đáy hộp", 北海道 -> "Bắc Hải"). We replace
# the Japanese term in the SOURCE text with a Latin rendering, which NLLB copies
# through reliably. Verified per-entry against the real model; only add an entry
# after confirming it improves output (see test_audio/evidence/).
NLLB_GLOSSARY = {
    "新幹線": "tàu Shinkansen",
    "北海道": "tỉnh Hokkaido",
    "箱根": "Hakone",
    "ヤンバルクイナ": "chim Yanbaru kuina",
    "ポッドキャスト": "podcast",
    "テーマ": "chủ đề",
}

# Pre-converted CTranslate2 NLLB model to download if local convert is skipped.
NLLB_CT2_HF_REPO = "entai2965/nllb-200-distilled-600M-ctranslate2"
NLLB_CT2_HF_REVISION = os.environ.get("NLLB_CT2_HF_REVISION", "main")

# ─── LLM Translation (Qwen2.5-3B via llama-cpp-python, opt-in) ───────────
# Alternative to NLLB for higher-quality context-aware translation.
# Activate with ZT_TRANSLATOR=llm or --llm flag.
# Auto-detect model: prefer 1.5B (lighter, faster on ≤16GB RAM systems) if present,
# otherwise fall back to 3B. Override with ZT_LLM_MODEL env var.
_LLM_1P5B_DIR = MODELS_DIR / "qwen2.5-1.5b-instruct"
_LLM_3B_DIR = MODELS_DIR / "qwen2.5-3b-instruct"
_LLM_1P5B_FILE = _LLM_1P5B_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
_LLM_3B_FILE = _LLM_3B_DIR / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
# Pick whichever model exists; prefer 1.5B for RAM-constrained machines.
if _LLM_1P5B_FILE.is_file():
    _LLM_DEFAULT_MODEL = _LLM_1P5B_FILE
    LLM_MODEL_DIR = _LLM_1P5B_DIR
else:
    _LLM_DEFAULT_MODEL = _LLM_3B_FILE
    LLM_MODEL_DIR = _LLM_3B_DIR
LLM_MODEL_PATH = Path(os.environ.get("ZT_LLM_MODEL", str(_LLM_DEFAULT_MODEL)))
LLM_N_CTX = int(os.environ.get("ZT_LLM_CTX", "512"))
# Reserve 2 cores for audio capture + ASR; give the rest to LLM.
LLM_N_THREADS = int(os.environ.get("ZT_LLM_THREADS", str(max(2, _PHYSICAL_CORES - 2))))
LLM_N_BATCH = int(os.environ.get("ZT_LLM_BATCH", "512"))
LLM_TEMPERATURE = float(os.environ.get("ZT_LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.environ.get("ZT_LLM_TOP_P", "0.3"))
LLM_FREQUENCY_PENALTY = float(os.environ.get("ZT_LLM_FREQ_PENALTY", "0.1"))
LLM_MAX_TOKENS = int(os.environ.get("ZT_LLM_MAX_TOKENS", "150"))
LLM_CONTEXT_SENTENCES = int(os.environ.get("ZT_LLM_CONTEXT", "3"))
LLM_USE_MLOCK = _env_flag("ZT_LLM_MLOCK", False)
LLM_SYSTEM_PROMPT = os.environ.get(
    "ZT_LLM_PROMPT",
    "Bạn là chuyên gia dịch thuật trong lĩnh vực CNTT và kinh doanh. "
    "Dịch chính xác từ tiếng Nhật sang tiếng Việt. "
    "Giữ nguyên thuật ngữ kỹ thuật (deploy, sprint, backlog...) khi phù hợp. "
    "Chỉ trả lời bản dịch, không giải thích thêm.",
)

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

# ─── Evidence logging (opt-in, for debugging dropped data) ───────────────
# When set (env ZT_EVIDENCE_LOG=<path> or --log <path>), every pipeline stage
# writes a structured JSONL event so a long meeting can be audited for exactly
# where a sentence was lost (queue_drop / dedup_skip / translate / display).
EVIDENCE_LOG_PATH = os.environ.get("ZT_EVIDENCE_LOG", "")

# ─── Shutdown ────────────────────────────────────────────────────────────
# Generous join timeout so an in-flight native ASR/translation call can finish
# before the process exits (a native call interrupted by interpreter teardown
# can segfault). The worker still returns as soon as the current item is done.
WORKER_SHUTDOWN_TIMEOUT = 30.0
