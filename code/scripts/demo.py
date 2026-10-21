"""Run the policy on a generated instance; no external data are required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "husarnet"))

import torch

from baseline_suite import run_baseline
from environment import HUSAREnv
from inference import decode
from model import HUSARNet, HUSARNetConfig
from objective import constraint_violations, husar_objective
from synthetic import make_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--family", choices=["A", "B", "C"], default="B")
    parser.add_argument("--agg", choices=["mean", "min", "cvar"], default="min")
    parser.add_argument("--mode", choices=["greedy", "sample", "beam"], default="greedy")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--beam", type=int, default=4)
    args = parser.parse_args()
    if args.n < 20:
        parser.error("--n must be at least 20 for the default synthetic hierarchy")
    if not 1 <= args.budget <= args.n:
        parser.error("require 1 <= --budget <= --n")
    if min(args.scenarios, args.n_samples, args.beam) < 1:
        parser.error("--scenarios, --n-samples and --beam must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checkpoint = None
    config = HUSARNetConfig()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if "net_cfg" not in checkpoint:
            parser.error("checkpoint lacks net_cfg; use a checkpoint from this release")
        config = HUSARNetConfig(**checkpoint["net_cfg"])
    model = HUSARNet(config).to(args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    instance = make_instance(family=args.family, n=args.n, n_scen=args.scenarios,
                             seed=args.seed)
    env = HUSAREnv(instance, args.budget, agg=args.agg, device=args.device)
    result = decode(model, env, mode=args.mode, seed=args.seed,
                    n_samples=args.n_samples, beam=args.beam)
    selection = result["selection"]
    violations = constraint_violations(instance, selection, args.budget)
    if violations:
        raise RuntimeError(f"infeasible selection: {violations} violations")
    reference = run_baseline("celf_lcb", instance, args.budget, agg=args.agg)
    print(json.dumps({
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "untrained": checkpoint is None,
        "network": config.tag(),
        "family": args.family,
        "agg": args.agg,
        "mode": args.mode,
        "selection": selection,
        "objective": husar_objective(instance, selection, args.agg),
        "lcb_reference_objective": reference["J_objective"],
        "constraint_violations": violations,
    }, indent=2))


if __name__ == "__main__":
    main()
