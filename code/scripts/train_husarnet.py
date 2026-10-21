# -*- coding: utf-8 -*-
"""
训练入口：HUSAR-Net（含网络消融配置）。

用法
----
    python code/scripts/train_husarnet.py --tag main_s0 --seed 0 --iters 400
    python code/scripts/train_husarnet.py --tag N5_noscen --seed 0 --iters 400 \
        --no-scen-encoder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PROJ = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "husarnet"))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

from model import HUSARNet, HUSARNetConfig      # noqa: E402
from train import TrainConfig, train            # noqa: E402


def build_cfg(args) -> HUSARNetConfig:
    """消融配置（N0-N6）。"""
    kw = {}
    if args.tag.startswith("N0"):
        kw = dict(use_scenario_encoder=False, use_scenario_projection=False,
                  use_hierarchy=False)
    elif args.tag.startswith("N1"):
        kw = dict(use_prov=False)
    elif args.tag.startswith("N2"):
        kw = dict(use_city=False)
    elif args.tag.startswith("N3"):
        kw = dict(use_scenario_encoder=False, use_scenario_projection=False)
    elif args.tag.startswith("N5"):
        kw = dict(use_scenario_encoder=False)
    elif args.tag.startswith("N6"):
        kw = dict(use_hierarchy=False)
    if args.no_scen_encoder:
        kw["use_scenario_encoder"] = False
    if args.no_scen_projection:
        kw["use_scenario_projection"] = False
    if args.no_hierarchy:
        kw["use_hierarchy"] = False
    if args.no_prov:
        kw["use_prov"] = False
    if args.no_city:
        kw["use_city"] = False
    return HUSARNetConfig(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--instances-per-update", type=int, default=6)
    ap.add_argument("--baseline", default="rollout", choices=["rollout", "value", "mean", "none"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-min", type=int, default=120)
    ap.add_argument("--n-max", type=int, default=400)
    ap.add_argument("--out-root", default=os.path.join(PROJ, "outputs"))
    ap.add_argument("--no-scen-encoder", action="store_true")
    ap.add_argument("--no-scen-projection", action="store_true")
    ap.add_argument("--no-hierarchy", action="store_true")
    ap.add_argument("--no-prov", action="store_true")
    ap.add_argument("--no-city", action="store_true")
    args = ap.parse_args()
    if args.iters < 1 or args.instances_per_update < 1:
        ap.error("--iters and --instances-per-update must be positive")
    if args.n_min < 20 or args.n_max < args.n_min:
        ap.error("require 20 <= --n-min <= --n-max")
    torch.set_num_threads(1)

    out_dir = os.path.join(args.out_root, f"husarnet_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    cfg = TrainConfig(iters=args.iters, instances_per_update=args.instances_per_update,
                      baseline=args.baseline, lr=args.lr, entropy_coef=args.entropy_coef,
                      seed=args.seed, device=args.device, out_dir=out_dir,
                      n_min=args.n_min, n_max=args.n_max, tag=args.tag)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mcfg = build_cfg(args)
    model = HUSARNet(mcfg)
    log(f"=== train {args.tag} seed={args.seed} iters={args.iters} "
        f"net={mcfg.tag()} baseline={args.baseline} ===")
    t0 = time.time()
    model, summary = train(cfg, model=model, log_fn=log)
    log(f"=== done {args.tag} elapsed={time.time() - t0:.0f}s best={summary['best']} "
        f"params={summary['n_param']} peakVRAM={summary['peak_vram_MB']}MB ===")
    # 保存最终权重与配置
    torch.save({"model": model.state_dict(), "net_cfg": vars(mcfg),
                "train_cfg": json.loads(cfg.to_json()), "iter": args.iters},
               os.path.join(out_dir, "final.pt"))
    logf.close()


if __name__ == "__main__":
    main()
