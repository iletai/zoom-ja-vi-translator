#!/usr/bin/env bash
set -euo pipefail

# Convert facebook/nllb-200-distilled-1.3B to CTranslate2 int8 for an opt-in
# translation quality upgrade (~+4 chrF++). Expect about 2.0-2.5GB RAM at
# runtime and roughly 300-700ms per sentence on a 4-core CPU.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/models/nllb-200-distilled-1.3B-ct2-int8"
OUTPUT_DIR_DISPLAY="models/nllb-200-distilled-1.3B-ct2-int8"
MODEL_ID="facebook/nllb-200-distilled-1.3B"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[NLLB-1.3B] ${OUTPUT_DIR_DISPLAY} already exists and is non-empty; skipping."
  echo "[NLLB-1.3B] The translator will auto-use this model on next launch."
  exit 0
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/ct2-transformers-converter" ]]; then
  CONVERTER="${PROJECT_ROOT}/.venv/bin/ct2-transformers-converter"
elif [[ -x "${PROJECT_ROOT}/.venv-arm64/bin/ct2-transformers-converter" ]]; then
  CONVERTER="${PROJECT_ROOT}/.venv-arm64/bin/ct2-transformers-converter"
elif command -v ct2-transformers-converter >/dev/null 2>&1; then
  CONVERTER="$(command -v ct2-transformers-converter)"
else
  cat >&2 <<'EOF'
[NLLB-1.3B] Missing 'ct2-transformers-converter'.
Install CTranslate2 in the project venv, then retry:
  python -m pip install ctranslate2
EOF
  exit 1
fi

cd "${PROJECT_ROOT}"

COMMAND=(
  "${CONVERTER}"
  --model "${MODEL_ID}"
  --output_dir "${OUTPUT_DIR_DISPLAY}"
  --quantization int8
  --copy_files tokenizer.json tokenizer_config.json sentencepiece.bpe.model
)

echo "[NLLB-1.3B] Converting ${MODEL_ID} to ${OUTPUT_DIR_DISPLAY}"
echo "[NLLB-1.3B] Running:"
printf '  %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"

echo "[NLLB-1.3B] Done: ${OUTPUT_DIR_DISPLAY}"
echo "[NLLB-1.3B] The translator will auto-use the 1.3B model on next launch (no code change needed)."
echo "[NLLB-1.3B] To disable it, remove ${OUTPUT_DIR_DISPLAY} or set NLLB_CT2_DIR=models/nllb-200-distilled-600M-ct2-int8."
