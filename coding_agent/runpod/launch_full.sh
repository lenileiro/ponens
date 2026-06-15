#!/bin/bash
# FULL data-scaled coding-agent run: proven dim768 causal LM on the 4.63GB manifest via robust scp.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/Users/leiro/workspace/llm/.venv/bin/python
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}" "$PY" runpod/launch_thinking.py \
  --gpu "NVIDIA H100 80GB HBM3" --disk 100 --name coding-agent \
  --train-steps 18000 \
  --multimodal --multimodal-manifest /tmp/bigcorpus/manifest.jsonl --multimodal-upload-manifest \
  --multimodal-dim 768 --multimodal-layers 12 --multimodal-heads 12 \
  --multimodal-batch 160 --multimodal-lr 3e-4 \
  --multimodal-max-vocab 32000 --multimodal-max-len 256 \
  --multimodal-decode-objective causal \
  --multimodal-source-balance-w 0.2 \
  --multimodal-repetition-unlikelihood-w 0.5 \
  "$@"
