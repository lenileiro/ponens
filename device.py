"""Device selection: CUDA (RunPod GPU) > MPS (local Mac) > CPU.

Set FER_DEVICE to force (e.g. FER_DEVICE=cuda or cpu). Otherwise auto-detect.
"""
import os
import torch


def get_device():
    forced = os.environ.get("FER_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info():
    d = get_device()
    if d == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} dev)"
    return d
