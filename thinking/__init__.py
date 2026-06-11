"""Datalog thinking flow — production package.

The model THINKS in typed inference steps; the Datalog engine CHECKS each emitted line at decode
time (a proof checker in the loop, never a prover). See thinking/cli.py for entry points:

  python -m thinking.cli selftest                      # correctness core (no GPU)
  python -m thinking.cli train --sup steps --out runs/x
  python -m thinking.cli eval runs/x --mode verified
  python -m thinking.cli demo runs/x --k 6
  python -m thinking.cli sweep --out runs/grid         # full comparison grid
"""
from .config import Config
from .world import ChainWorld, RULES, entity_pools
from .trace import Vocab, render_example, parse_line, build_vocab, pack_batch
from .verify import StepChecker
from .flow import FlowRuntime
from .train import Trainer, load_run
from .evaluate import evaluate
from .deep_eval import deep_eval
from .probes import probe_report

__all__ = ["Config", "ChainWorld", "RULES", "entity_pools", "Vocab", "render_example",
           "parse_line", "build_vocab", "pack_batch", "StepChecker", "FlowRuntime",
           "Trainer", "load_run", "evaluate", "deep_eval", "probe_report"]
