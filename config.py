"""Central configuration for the Zoom Japanese->Vietnamese translator.

All tunable parameters live here so modules stay decoupled and the pipeline
can be adjusted without touching implementation code.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from a .env file into ``os.environ``.

    Zero-dependency (no python-dotenv). Real environment variables always win —
    a value already set in the environment is NOT overwritten, so the .env file
    is a default, overridable per-run by ``ZT_FOO=... python main.py``. Supports
    ``#`` comments, blank lines, ``export KEY=val``, and quoted values; malformed
    lines are skipped silently so a typo never crashes startup.

    Secrets (the 9router URL/key/model) live here instead of hardcoded in this
    file, so the repo ships no credentials and each machine configures its own.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if (len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env from the project root BEFORE any os.environ.get() below, so file
# values are visible to every config read. A real env var still overrides it.
_load_dotenv(Path(__file__).resolve().parent / ".env")


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

# ─── Audio input enrichment (pre-ASR conditioning) ───────────────────────
# Light, dependency-free DSP applied to each captured block before VAD/ASR to
# raise recognition accuracy on real meeting audio (quiet speakers, HVAC rumble,
# inconsistent levels from different participants). All knobs are env-overridable
# and the whole stage is a no-op when ZT_AUDIO_ENRICH is off.
AUDIO_ENRICH = _env_flag("ZT_AUDIO_ENRICH", True)
# High-pass cutoff (Hz): removes DC offset + low-frequency rumble (fans, mic
# bumps, room hum) below the speech band. 0 disables. ~80 Hz is safe for speech.
AUDIO_HIGHPASS_HZ = float(os.environ.get("ZT_AUDIO_HIGHPASS_HZ", "80"))
# Target RMS for soft AGC: quiet blocks are gained up toward this level so the
# ASR sees a consistent loudness. 0 disables gain. ~0.05 is a comfortable
# speech level for float32 audio in [-1, 1].
AUDIO_TARGET_RMS = float(os.environ.get("ZT_AUDIO_TARGET_RMS", "0.05"))
# Cap the AGC gain so near-silent background blocks are not amplified into noise.
AUDIO_MAX_GAIN = float(os.environ.get("ZT_AUDIO_MAX_GAIN", "8.0"))
# Below this RMS a block is treated as silence and left unGained (avoids pumping
# noise up during pauses).
AUDIO_NOISE_FLOOR_RMS = float(os.environ.get("ZT_AUDIO_NOISE_FLOOR_RMS", "0.005"))

# Per-utterance loudness normalization, applied to a WHOLE recognized utterance
# right before ASR (not per-200ms block). A consistent utterance-level loudness
# helps the recognizer's log-mel features; doing it on the aggregated segment
# (rather than per block) avoids pumping gain mid-word. Peak-normalizes toward
# this target with a gain cap; silence (below the noise floor) is left alone.
AUDIO_UTTERANCE_NORM = _env_flag("ZT_AUDIO_UTTERANCE_NORM", True)
AUDIO_UTTERANCE_PEAK = float(os.environ.get("ZT_AUDIO_UTTERANCE_PEAK", "0.95"))
AUDIO_UTTERANCE_MAX_GAIN = float(os.environ.get("ZT_AUDIO_UTTERANCE_MAX_GAIN", "10.0"))

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
# 320ms (up from 240) gives aggressiveness=3 more margin to keep soft onsets that
# were still being clipped in real meeting logs.
VAD_PREROLL_MS = int(os.environ.get("ZT_VAD_PREROLL_MS", "320"))
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
# Penalize the transducer's blank symbol during decoding. >0 makes the model
# emit fewer blanks, which reduces dropped sentence onsets / clipped leading
# mora (a real failure in meeting audio). 0 = off (model default). Only applied
# when the installed sherpa-onnx build accepts the parameter (checked at init).
# Keep modest: too high inserts spurious tokens. ~1.0 is a safe start.
ASR_BLANK_PENALTY = float(os.environ.get("ZT_ASR_BLANK_PENALTY", "1.0"))
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
# 128 target tokens safely covers JP→VI (ratio ~1:1.2). The old value of 350
# caused worst-case latency spikes from repetition loops; now capped tighter.
NLLB_MAX_DECODING_LENGTH = int(os.environ.get("ZT_NLLB_MAX_DECODE", "128"))
# Anti-repetition / quality knobs applied to CTranslate2 decoding. These fix the
# observed "Tôi xin xin" short loops, dropped trailing words, and empty outputs:
#   - no_repeat_ngram_size: forbid repeating any n-gram of this size without
#     blocking legitimate repeated bigrams such as Vietnamese reduplication
#   - repetition_penalty: >1.0 discourages re-emitting recent tokens
#   - min_decoding_length: force at least this many target tokens (avoids "")
NLLB_NO_REPEAT_NGRAM_SIZE = 4
NLLB_REPETITION_PENALTY = 1.2
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
    # Only entries that NLLB copies through reliably (proper nouns / short loanwords).
    # Full Vietnamese phrases in source CONFUSE NLLB — use post-translation fix instead.
    "新幹線": "tàu Shinkansen",
    "北海道": "tỉnh Hokkaido",
    "箱根": "Hakone",
    "ヤンバルクイナ": "chim Yanbaru kuina",
    "ポッドキャスト": "podcast",
    "テーマ": "chủ đề",
    # IT terms — short English that NLLB tends to copy through
    "クロステナント": "Cross-Tenant",
    "マルチテナント": "Multi-Tenant",
    "テナント": "Tenant",
    "ユースケース": "Use-Case",
    "デプロイ": "Deploy",
    "マイクロサービス": "Microservice",
    "ステージング": "Staging",
    # Emergency medical / rescue domain terms (NLLB reads kanji literally → hallucinates)
    # Evidence: 引き継ぎ → "thai nhi" (fetus), 病院連携 → "mùi miệng" (bad breath)
    "引き継ぎ": "handover",
    "病院連携": "hospital coordination",
    "傷病者": "injured person",
    "搬送": "transport",
    "搬送者": "transporter",
    "搬送元": "transport origin",
    "搬送決定": "transport decision",
    "出動": "dispatch",
    "受入": "acceptance",
    "救急搬送": "emergency transport",
    "救急隊": "EMS squad",
    "救急車": "ambulance",
    "消防": "fire department",
    "消防署": "fire station",
    "広域地図": "wide-area map",
    "多数傷病者": "mass casualties",
    "消火": "firefighting",
    "消火活動": "firefighting operation",
    "救助": "rescue",
    "救助活動": "rescue operation",
    "通信指令": "communication dispatch",
    "通信指令台": "dispatch console",
    "現場": "scene",
    "現場到着": "scene arrival",
    "患者": "patient",
    "バイタル": "vital signs",
    "トリアージ": "triage",
    "医療機関": "medical facility",
    "災害": "disaster",
    "災害時": "during disaster",
}

# Post-translation corrections: fix known BAD Vietnamese outputs from NLLB.
# Applied AFTER translation to fix domain terms NLLB mistranslates.
# Format: wrong_vietnamese → correct_vietnamese
NLLB_POST_TRANSLATION = {
    # IT terms NLLB mistranslates
    "người thuê nhà": "tenant",
    "người thuê qua": "cross-tenant",
    "căn hộ": "tenant",
    "nhà khoa học": "cross-tenant",
    # Rescue domain
    "thừa kế thai nhi": "bàn giao",
    "nhiều thai nhi": "bàn giao nhiều lần",
    "thai nhi": "bàn giao",
    "mùi miệng": "liên hệ bệnh viện",
    "người chuyển": "bệnh nhân vận chuyển",
    "cảnh sát cứu hỏa": "trạm cứu hỏa",
    # Name hallucination fixes (NLLB reads kanji names literally)
    "nghệ thuật của rừng": "phía Omori",
    "rừng nghệ thuật": "Omori",
    "cảnh sát cứu hỏa láng giềng": "trạm cứu hỏa lân cận",
    # Bad SVO / grammar fixes
    "và tôi thấy nó rất phấn khích": "tôi rất ngạc nhiên",
}

# Pre-converted CTranslate2 NLLB model to download if local convert is skipped.
NLLB_CT2_HF_REPO = "entai2965/nllb-200-distilled-600M-ctranslate2"
NLLB_CT2_HF_REVISION = os.environ.get("NLLB_CT2_HF_REVISION", "main")

# ─── LLM Translation (Qwen2.5-3B via llama-cpp-python, opt-in) ───────────
# Alternative to NLLB for higher-quality context-aware translation.
# Activate with ZT_TRANSLATOR=llm or --llm flag.
# Auto-detect model: prefer 3B (better translation quality ~30%) if present,
# otherwise fall back to 1.5B. Override with ZT_LLM_MODEL env var.
_LLM_1P5B_DIR = MODELS_DIR / "qwen2.5-1.5b-instruct"
_LLM_3B_DIR = MODELS_DIR / "qwen2.5-3b-instruct"
_LLM_1P5B_FILE = _LLM_1P5B_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
_LLM_3B_FILE = _LLM_3B_DIR / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
# Pick whichever model exists; prefer 3B for better accuracy.
if _LLM_3B_FILE.is_file():
    _LLM_DEFAULT_MODEL = _LLM_3B_FILE
    LLM_MODEL_DIR = _LLM_3B_DIR
else:
    _LLM_DEFAULT_MODEL = _LLM_1P5B_FILE
    LLM_MODEL_DIR = _LLM_1P5B_DIR
LLM_MODEL_PATH = Path(os.environ.get("ZT_LLM_MODEL", str(_LLM_DEFAULT_MODEL)))
LLM_N_CTX = int(os.environ.get("ZT_LLM_CTX", "1024"))
# LLM RAM cache capacity in MB. The LlamaRAMCache stores KV state for prompt
# prefix reuse. Lower values reduce peak RSS on memory-constrained systems.
LLM_RAM_CACHE_MB = int(os.environ.get("ZT_LLM_CACHE_MB", "256"))
# Reserve 2 cores for audio capture + ASR; give the rest to LLM.
LLM_N_THREADS = int(os.environ.get("ZT_LLM_THREADS", str(max(2, _PHYSICAL_CORES - 2))))
LLM_N_BATCH = int(os.environ.get("ZT_LLM_BATCH", "512"))
LLM_N_GPU_LAYERS = int(os.environ.get("ZT_LLM_GPU_LAYERS", "-1"))  # -1 = all layers to GPU if available
LLM_TEMPERATURE = float(os.environ.get("ZT_LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.environ.get("ZT_LLM_TOP_P", "0.3"))
LLM_FREQUENCY_PENALTY = float(os.environ.get("ZT_LLM_FREQ_PENALTY", "0.1"))
LLM_MAX_TOKENS = int(os.environ.get("ZT_LLM_MAX_TOKENS", "150"))
LLM_CONTEXT_SENTENCES = int(os.environ.get("ZT_LLM_CONTEXT", "3"))
LLM_USE_MLOCK = _env_flag("ZT_LLM_MLOCK", False)
# Enable GBNF grammar to hard-constrain output to Latin/Vietnamese characters only.
# This provides a 100% guarantee against CJK output but adds ~10-30ms latency per token.
# The logit_bias approach (always active) is usually sufficient; enable this as extra safety.
LLM_USE_GRAMMAR = _env_flag("ZT_LLM_GRAMMAR", False)
LLM_SYSTEM_PROMPT = os.environ.get(
    "ZT_LLM_PROMPT",
    # Bilingual prompt: English first (strongest instruction pathway for Qwen),
    # then Vietnamese reinforcement.
    "You are a Japanese-to-Vietnamese translator for IT meetings "
    "about emergency medical dispatch systems (救急搬送システム) for Japan's fire/EMS service. "
    "CRITICAL: Output ONLY Vietnamese using Latin script. "
    "NEVER use Chinese characters. NEVER use Japanese kana. "
    "Output exactly ONE line of Vietnamese translation.\n"
    "Bạn là máy dịch Nhật→Việt cho cuộc họp CNTT về hệ thống điều phối cứu hộ Nhật Bản. "
    "CHỈ xuất tiếng Việt (chữ Latin). "
    "KHÔNG ĐƯỢC dùng chữ Hán/tiếng Trung. "
    "Giữ nguyên thuật ngữ IT: Cloud, AWS, API, deploy, sprint, Lambda, EC2, S3. "
    "Dịch thuật ngữ cứu hộ: 消防＝cứu hỏa, 救急＝cấp cứu, 搬送＝vận chuyển, "
    "傷病者＝nạn nhân, 引き継ぎ＝bàn giao, 出動＝điều động, "
    "割り込み＝task gián đoạn, 親タスク＝parent task, 子タスク＝child task, "
    "受入＝tiếp nhận. "
    "Dịch ngắn gọn, tự nhiên.\n"
    "QUAN TRỌNG - Giữ tên người Nhật (romaji):\n"
    "  - '中野さん' → 'Anh Nakano' / 'Nakano-san'\n"
    "  - '川村さんのほう' → 'phía Kawamura'\n"
    "  - '羽根さん' → 'anh Hane'\n"
    "  - 'カリスさん' → 'anh Caris'\n"
    "  - 'ハレ井さん' → 'anh Harei'\n"
    "  - 'ジャンさん' → 'anh Jan'\n"
    "  - Tên + さん/様 KHÔNG ĐƯỢC bỏ hoặc đổi thành 'bạn'/'ông'/'cô'\n"
    "  - Tên Nhật (Kanji như 深瀬, 大森, 河合) giữ romaji: Fukase, Omori, Kawai\n"
    "CẢNH BÁO - Không tự ý tách kanji:\n"
    "  - 関係 = quan hệ (KHÔNG tách thành 関 Seki + 係 phòng ban)\n"
    "  - 関数 = hàm số (KHÔNG tách thành 関 Seki + 数 số)\n"
    "  - 関連 = liên quan (KHÔNG tách thành 関 Seki + 連)\n"
    "  - Đây là từ ghép tiếng Nhật thông thường, KHÔNG phải tên riêng\n"
    "CẢNH BÁO - Không bịa nghĩa từ ASR lỗi:\n"
    "  - Nếu câu nhập vô nghĩa (ASR lỗi), dịch sát nghĩa đen\n"
    "  - KHÔNG ĐƯỢC bịa thêm ngữ cảnh không có trong câu gốc\n"
    "  - Ưu tiên dịch sát > dịch hay",
)

# ─── Router backend (9router / OpenAI-compatible gateway, opt-in) ─────────
# Route translation through a local OpenAI-compatible gateway (9router) that
# fronts hosted models (Claude / GPT / DeepSeek / …). Activate with
# ZT_TRANSLATOR=router. Network calls leave the machine via the gateway, so this
# is NOT an offline backend — use nllb/llm for fully local operation.
#
# The gateway speaks POST {ROUTER_BASE_URL}/chat/completions with the standard
# OpenAI body; the Vietnamese text is read from choices[0].message.content.
#
# Credentials come from the environment / .env (see .env.example) — never
# hardcode a key here. ROUTER_API_KEY defaults to empty so a missing .env fails
# loudly at the gateway rather than shipping a real token in the repo.
ROUTER_BASE_URL = os.environ.get("ZT_ROUTER_BASE_URL", "http://127.0.0.1:20128/v1")
ROUTER_API_KEY = os.environ.get("ZT_ROUTER_KEY", "")
# Default sonnet, not haiku: haiku-4.5 ignores the translate-only prompt and
# slips into assistant mode on spoken fragments — 部長 ("manager", calling the
# boss) came back as "Vâng, tôi đây ạ", and most lines got a hallucinated
# "Vâng/ạ" prefix. sonnet-4.6 obeys the prompt (部長→"Trưởng phòng") at ~+0.6s/seg.
ROUTER_MODEL = os.environ.get("ZT_ROUTER_MODEL", "gh/claude-sonnet-4.6")
# 0.0 (greedy): MT evidence (Peng et al. EMNLP 2023, arXiv:2303.13780) shows
# translation quality degrades monotonically as temperature rises — translation
# is analytical, not creative. Effect is larger for distant/non-English-centric
# pairs (so more so for JA→VI). Gateway accepts 0.0 fine (verified).
ROUTER_TEMPERATURE = float(os.environ.get("ZT_ROUTER_TEMPERATURE", "0.0"))
# 180 target tokens comfortably covers a JA→VI sentence (ratio ~1:1.2) without
# the latency of the old 256 ceiling. Raise only if long sentences get clipped.
ROUTER_MAX_TOKENS = int(os.environ.get("ZT_ROUTER_MAX_TOKENS", "180"))
# Hard per-request deadline. Live captions must fail fast: a stalled segment is
# better dropped (and the next one shown) than blocking the meeting for 20s.
ROUTER_TIMEOUT_S = float(os.environ.get("ZT_ROUTER_TIMEOUT", "12"))
# Sentences of prior context fed to the model for coherent terminology across a
# meeting (0 disables). Reuses the LLM backend's JA→VI system prompt.
ROUTER_CONTEXT_SENTENCES = int(os.environ.get("ZT_ROUTER_CONTEXT", "3"))
# A dedicated, terse prompt for STRONG hosted models (Claude/GPT). The local
# Qwen prompt (LLM_SYSTEM_PROMPT) must NOT be reused here: its instruction to
# "translate literally if the input is nonsense / ASR-broken" makes a capable
# model switch into assistant mode — it wraps the translation in quotes, adds
# commentary, and asks the speaker clarifying questions instead of translating
# (observed in real sessions: ~40% of segments). This prompt forbids exactly
# that: translate only, output nothing else, never converse.
_ROUTER_DEFAULT_PROMPT = (
    "You are a machine translation engine, Japanese to Vietnamese, for a live "
    "IT meeting about a Japanese emergency-medical-dispatch system.\n"
    # XML section tags: Claude 4.x parses <rules>/<examples> as distinct blocks
    # more reliably than prose labels (Anthropic prompt-eng docs). The user input
    # is wrapped in <source_ja> by _build_messages, so these tag names never clash.
    "<rules>\n"
    "- Output ONLY the Vietnamese translation. No quotes, no notes, no labels, "
    "no explanation, no questions back to the speaker.\n"
    "- Translate every input as-is, even if it is a fragment, filler, or looks "
    "garbled. Never comment that the input is unclear or an ASR error — just "
    "translate the words literally. If the input trails off, let the Vietnamese "
    "trail off too (end with '...'); do not invent an ending. "
    "If the entire input is a single functional word with no content "
    "(ちょっと, ね, よ, か, が, は, を, に, も) output '...'\n"
    "- This is spoken meeting dialogue, not written text. Use natural spoken "
    "Vietnamese, keep the speaker's politeness: render Japanese keigo (です/ます, "
    "いただく, お願いします, いたします) with Vietnamese politeness markers (ạ, dạ, "
    "vâng, xin, được không ạ) — never reduce a polite request to a bare command.\n"
    "- This is turn-based dialogue. When a sentence begins with a question marker "
    "(か、かしら、ですか) render it as a question in Vietnamese (…không?). "
    "ませんか is an invitation/proposal — render with …nhé or …được không, not a blunt negative question. "
    "When a sentence is a response to a question, preserve any implicit agreement or disagreement tone.\n"
    "- First person 自分/私 = 'tôi' (I), never 'bạn' (you).\n"
    "- Short back-channels (はい, うん, ええ, なるほど, えっと) are handled upstream; "
    "if one slips through, render it as a brief acknowledgement.\n"
    "- Keep Japanese personal names in romaji with their honorific "
    "(中野さん → Nakano-san).\n"
    "- Render Japanese numbers and counters as Arabic numerals in Vietnamese "
    "(3時間→3 giờ, 第2フェーズ→Giai đoạn 2, 2割→20%, 1時間半→1 giờ rưỡi).\n"
    "- Output exactly one line. Be concise and natural; never add information "
    "that is not in the source.\n"
    "- Never use Chinese characters or Japanese kana in the output.\n"
    "</rules>\n"
    # Few-shot exemplars anchor the input→output pattern that weak models drop
    # (research: Peng et al. + OpenAI/Anthropic prompt-eng docs). They include
    # the exact trap cases: 部長 is the TITLE "Trưởng phòng" (addressing the
    # boss), NOT a greeting to answer; a bare keigo fragment must keep its
    # politeness WITHOUT gaining an invented "Vâng"; a trailing-off fragment
    # stays unfinished. These are the failures observed live with weak models.
    "<examples>\n"
    "部長 → Trưởng phòng\n"
    "時間早まったの → Thời gian bị đẩy sớm lên à\n"
    "ですから新幹線の時間も1時間早めた方がよろしいかと → "
    "Vậy nên tàu Shinkansen cũng dời sớm 1 tiếng thì tốt hơn ạ\n"
    "はい → Vâng\n"
    "この素材は吸水性に大変優れておりまして → "
    "Vật liệu này có khả năng hút ẩm rất tốt...\n"
    "対応に3時間かかりました → Mất 3 giờ để xử lý\n"
    "一緒に確認しませんか → Cùng kiểm tra nhé\n"
    "中野さんが説明します → Nakano-san sẽ giải thích\n"
    "デプロイが完了しました → deploy đã hoàn tất\n"
    "</examples>\n"
    # Sandwich defense: restate the task AFTER the examples so it is the most
    # recent instruction before the model sees the user input.
    "Remember: output Vietnamese only — one line, no kana or kanji in output, "
    "never reply or add anything. Translate the Japanese in the next message."
)
ROUTER_SYSTEM_PROMPT = os.environ.get("ZT_ROUTER_PROMPT", _ROUTER_DEFAULT_PROMPT)
# Max concurrent HTTP requests when translating a drained batch. The translate
# worker hands us up to TRANSLATE_MAX_BATCH sentences at once; fanning them out
# keeps a batch ~1 round-trip instead of N. Keep modest to avoid hammering the
# gateway / hitting its rate limits.
ROUTER_MAX_PARALLEL = int(os.environ.get("ZT_ROUTER_MAX_PARALLEL", "4"))

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
# Automatically scroll the terminal viewport to the latest output whenever new
# subtitle text is printed. Useful for real-time monitoring — prevents the view
# from "sticking" when you accidentally scroll up during a meeting.
AUTO_SCROLL = _env_flag("ZT_AUTO_SCROLL", default=True)

# ─── Evidence logging (opt-in, for debugging dropped data) ───────────────
# When set (env ZT_EVIDENCE_LOG=<path> or --log <path>), every pipeline stage
# writes a structured JSONL event so a long meeting can be audited for exactly
# where a sentence was lost (queue_drop / dedup_skip / translate / display).
EVIDENCE_LOG_PATH = os.environ.get("ZT_EVIDENCE_LOG", "")

# ─── Debug log ────────────────────────────────────────────────────────────
# Full debug log capturing ALL events: inputs, outputs, errors, warnings.
# Always enabled; written to LOG_DIR with timestamped filename. Use for
# post-session analytics and troubleshooting.
LOG_DIR = Path(os.environ.get("ZT_LOG_DIR", str(PROJECT_ROOT / "logs")))
# Log level for the file handler. DEBUG captures everything; INFO is less noisy.
LOG_LEVEL = os.environ.get("ZT_LOG_LEVEL", "DEBUG").upper()

# ─── Shutdown ────────────────────────────────────────────────────────────
# Generous join timeout so an in-flight native ASR/translation call can finish
# before the process exits (a native call interrupted by interpreter teardown
# can segfault). The worker still returns as soon as the current item is done.
WORKER_SHUTDOWN_TIMEOUT = 30.0
