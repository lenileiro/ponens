#!/usr/bin/env bash
# Idempotent environment setup for a RunPod GPU pod (CUDA).
# Assumes the experiment dir (code + data/) has been synced to $WORKDIR.
# Safe to re-run. Provisions a CUDA PyTorch venv with the experiment deps.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/fer_relational}"
CUDA_WHL="${CUDA_WHL:-cu124}"   # match the pod's CUDA (cu121/cu124/cu126)
# venv goes on the pod's LOCAL disk (not the /workspace network volume, which flakes under
# uv's fast writes -> "Stale file handle"). copy link-mode avoids hardlink/clone FS issues.
VENV="${VENV:-/root/fer-venv}"
export UV_LINK_MODE=copy

cd "$WORKDIR"
echo "== setup in $WORKDIR (CUDA wheels: $CUDA_WHL) =="

# uv if missing
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# venv (Python 3.12 for torch wheel coverage) — on local disk (see VENV above)
uv venv --python 3.12 "$VENV"

# Torch: if a CUDA torch is already present in a runpod/pytorch base image we could reuse it,
# but we install into our own venv for reproducibility.
VIRTUAL_ENV=$VENV uv pip install --quiet \
  "torch" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
BASE_DEPS=(numpy pandas scikit-learn tokenizers nltk pillow)
if [ "${INSTALL_IMAGE_EMBED_DEPS:-0}" = "1" ]; then
  BASE_DEPS+=(transformers accelerate protobuf)
fi
if [ "${INSTALL_IMAGE_SCORE_DEPS:-0}" = "1" ]; then
  BASE_DEPS+=(transformers accelerate protobuf)
fi
if [ "${INSTALL_IMAGE_TEXT_SEQUENCE_DEPS:-0}" = "1" ]; then
  BASE_DEPS+=(transformers accelerate sentencepiece protobuf)
fi
if [ "${INSTALL_IMAGE_CAPTION_DEPS:-0}" = "1" ]; then
  BASE_DEPS+=(transformers accelerate sentencepiece protobuf)
fi
if [ "${INSTALL_IMAGE_HF_AE_DEPS:-0}" = "1" ]; then
  BASE_DEPS+=(diffusers transformers accelerate safetensors protobuf)
fi
VIRTUAL_ENV=$VENV uv pip install --quiet "${BASE_DEPS[@]}"
# WordNet for dictionary.py (genus-projected A1 dictionary)
$VENV/bin/python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# verify CUDA visible
$VENV/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu"))
PY

echo "== setup done =="
