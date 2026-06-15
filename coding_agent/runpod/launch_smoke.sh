#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/Users/leiro/workspace/llm/.venv/bin/python
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}" "$PY" runpod/launch_thinking.py \
  --gpu "NVIDIA H100 80GB HBM3" --disk 80 --name coding-agent \
  --train-steps 300 \
  --multimodal --multimodal-manifest /tmp/bigcorpus/manifest.jsonl --multimodal-upload-manifest \
  --multimodal-dim 256 --multimodal-layers 4 --multimodal-heads 4 \
  --multimodal-batch 32 --multimodal-max-vocab 32000 --multimodal-max-len 256 \
  --multimodal-decode-objective causal \
  "$@"
