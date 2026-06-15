#!/bin/bash
# Scaled coding-agent run: ~300M causal LM, code-heavy+tool manifest, all fixes, no reasoning stack.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/Users/leiro/workspace/llm/.venv/bin/python
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}" "$PY" runpod/launch_thinking.py \
  --gpu "NVIDIA H100 80GB HBM3" --name coding-agent \
  --train-steps 14000 \
  --multimodal --multimodal-manifest data/manifest.jsonl \
  --multimodal-dim 1024 --multimodal-layers 16 --multimodal-heads 16 \
  --multimodal-batch 48 --multimodal-lr 3e-4 \
  --multimodal-max-vocab 32000 --multimodal-max-len 256 \
  --multimodal-decode-objective causal \
  --multimodal-source-balance-w 0.2 \
  --multimodal-repetition-unlikelihood-w 0.5 \
  "$@"
