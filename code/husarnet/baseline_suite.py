# -*- coding: utf-8 -*-
"""
HUSAR-Net 对照基线套件（baseline_suite；全部在 objective.py 的统一评价口径下比较）。

注意：本文件刻意不命名为 `baselines.py`，以免与既有实现 `usma_heuristics/baselines.py`
（作为对照算法被导入）同名冲突。

基线实现**复用**本项目 v0.2/v0.3 的既有实现（同为本文作者代码，已由 38 项单测与真实
实例结果验证），通过 `to_legacy()` 把 HUSAR 实例（区域权重 a_g、层级权重、λ）映射为
旧实现接受的实例（v_g = a_g），从而保证"同一实例、同一预算、同一评价目标"的公平对照。

方法键名
--------
quota            按辖区在册企业数比例配额 + 辖区内随机（B1，非学习下限）
rank_rule        历史违规计数规则排序（B2；分数由 dataio 提供）
rank_mean/rank_lcb   纯风险排序（B3；无空间项）
celf_base        基础代理目标（情景均值 + 加权饱和覆盖）的 CELF 惰性贪心（B4/B5/B6 系列）
celf_lcb         LCB 代理目标（逐企业 10% 分位 + 加权覆盖）的 CELF 惰性贪心（B6）
direct_maxmin    正确实现的直接 max-min 贪心（B7；每轮全量重算精确边际）
lcb_1swap        LCB 解 + 1-Swap 局部搜索（B8）
swg_exact        情景加权贪心（B8'；对照）
lcb_alns         LCB-ALNS 自适应大邻域搜索（B9；同墙钟预算）
milp_maxmin / milp_cvar / milp_base   MILP 参考（B10；仅报告 proven_optimal）
husarnet_greedy / husarnet_sample / husarnet_beam   HUSAR-Net（在 inference.py，由脚本调用）
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from objective import HUSARInstance, all_metrics

_LEGACY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legacy"))
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)


def to_legacy(inst: HUSARInstance):
    """HUSAR 实例 -> 旧实现实例（v_g = a_g）。要求旧的 objectives.py 可导入。"""
    import objectives as legacy
    return legacy.USMAInstance(
        Q=inst.Q, city_idx=inst.city_idx, prov_idx=inst.prov_idx,
        n_firm_city=inst.n_firm_city, n_firm_prov=inst.n_firm_prov,
        lam=inst.lam, alpha=inst.alpha, b_cap=inst.b_cap,
        omega_city=inst.omega_city, omega_prov=inst.omega_prov,
        v_city=inst.a_city, v_prov=inst.a_prov,
        q_mean=inst.q_mean, q_lcb=inst.q_lcb,
        city_scale=inst.omega_city > 0, prov_scale=inst.omega_prov > 0,
        name=inst.name, meta=dict(inst.meta))


# =====================================================================
def _pack(inst, method, S, t0, agg, tau, labels=None, K=None, **extra) -> dict:
    met = all_metrics(inst, S, agg=agg, tau=tau, labels=labels, K=K)
    out = {"method": method, "selection": [int(i) for i in S],
           "time": time.time() - t0, "agg_eval": agg}
    out.update(met)
    out.update(extra)
    return out


def run_baseline(name: str, inst: HUSARInstance, K: int, agg: str = "min",
                 tau: float = 0.25, labels=None, scores: np.ndarray | None = None,
                 time_budget: float = 60.0, seed: int = 0, milp_gap: float = 1e-6,
                 milp_time: float = 600.0, verify: bool = False) -> dict:
    """统一入口：运行一个基线并返回统一评价指标。"""
    t0 = time.time()
    legacy = to_legacy(inst)
    import baselines as B
    import heapq  # noqa: F401

    if name == "quota":
        r = B.proportional_quota(legacy, K, seed=seed)
        S = r["selection"]
        return _pack(inst, name, S, t0, agg, tau, labels, K, evals=r.get("evals"))

    if name == "rank_rule":
        if scores is None:
            raise ValueError("rank_rule 需要 scores（历史违规计数规则分）")
        S = np.argsort(-np.asarray(scores, float))[:K]
        return _pack(inst, name, S, t0, agg, tau, labels, K)

    if name in ("rank_mean", "rank_lcb", "rank_worst"):
        if name == "rank_worst":
            r = B.rank_worst_scen(legacy, K, seed=seed)
        else:
            r = B.rank_select(legacy, K, score=name.split("_")[1], seed=seed)
        return _pack(inst, name, r["selection"], t0, agg, tau, labels, K,
                     evals=r.get("evals"))

    if name in ("celf_base", "celf_lcb"):
        r = B.celf_greedy(legacy, K, proxy="base" if name == "celf_base" else "lcb")
        return _pack(inst, name, r["selection"], t0, agg, tau, labels, K,
                     evals=r.get("evals"), time_internal=r["time"])

    if name == "direct_maxmin":
        r = B.direct_maxmin_true(legacy, K)
        return _pack(inst, name, r["selection"], t0, agg, tau, labels, K,
                     evals=r.get("evals"), time_internal=r["time"])

    if name == "lcb_1swap":
        init = B.celf_greedy(legacy, K, proxy="lcb")["selection"]
        r = B.local_search_1swap(legacy, init)
        return _pack(inst, name, r["selection"], t0, agg, tau, labels, K,
                     evals=r.get("evals"), time_internal=r["time"])

    if name == "swg_exact":
        import scenario_weighted_greedy as SWG
        r = SWG.swg(legacy, K, eta=SWG.eta_from_kappa(
            SWG.outcome_scale(legacy, B.celf_greedy(legacy, K, proxy="lcb")["selection"]), 1.0),
            variant="exact")
        return _pack(inst, name, r["selection"], t0, agg, tau, labels, K,
                     time_internal=r["time"])

    if name == "lcb_alns":
        import alns as ALNSM
        cfg = ALNSM.ALNSConfig(objective="maxmin" if agg == "min" else "cvar",
                               tau=tau, seed=20261016 + seed)
        r = ALNSM.run_alns(legacy, K, cfg=cfg, time_budget=time_budget)
        return _pack(inst, name, r["selection"], t0, agg, tau, labels, K,
                     iters=r.get("iters"), time_internal=r["time"])

    if name in ("milp_maxmin", "milp_cvar", "milp_base"):
        import milp_reference as M
        if name == "milp_maxmin":
            r = M.milp_maxmin(legacy, K, time_limit=milp_time, mip_gap=milp_gap)
        elif name == "milp_cvar":
            r = M.milp_cvar(legacy, K, tau=tau, time_limit=milp_time, mip_gap=milp_gap)
        else:
            r = M.milp_base(legacy, K, time_limit=milp_time, mip_gap=milp_gap)
        S = r["selection"]
        return _pack(inst, name, S, t0, agg, tau, labels, K,
                     proven_optimal=r.get("proven_optimal"),
                     dual_bound=r.get("dual_bound"), incumbent=r.get("incumbent"),
                     mip_gap=r.get("mip_gap"), time_internal=r.get("runtime_s"))

    raise ValueError(f"unknown baseline: {name}")


# =====================================================================
BASELINE_GROUPS = {
    "B1": ["quota"],
    "B2": ["rank_rule"],
    "B3": ["rank_mean", "rank_lcb"],
    "B4": ["celf_base"],
    "B5": ["celf_base"],           # 单尺度：通过实例 omega_prov=0 实现
    "B6": ["celf_lcb"],
    "B7": ["direct_maxmin"],
    "B8": ["lcb_1swap"],
    "B9": ["lcb_alns"],
    "B10": ["milp_maxmin", "milp_cvar", "milp_base"],
}
