# -*- coding: utf-8 -*-
"""
HUSAR-Net 推理协议。

三种解码
--------
1. deterministic（greedy）：每步取合法动作中策略概率最大者；
2. stochastic sampling ：按策略分布采样 k 条合法序列，用共享评价器选最优解；
3. beam search（可选）：宽度 w 的束搜索，末端用共享评价器在同一批候选中选最优。

所有协议统一报告：网络前向次数、样本条数、墙钟时间、目标值（由 objective.py 计算）。
公平性：与经典算法对比时必须同时报告解码开销（网络调用次数与墙钟），不得隐藏。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from objective import husar_objective


# =====================================================================
@torch.no_grad()
def decode(model, env, mode: str = "greedy", n_samples: int = 1, temperature: float = 1.0,
           seed: int | None = None, beam: int = 4, timed: bool = True) -> dict:
    """在单个实例上解码。env 提供 K/agg/τ；调用后 env 处于终止状态。"""
    t0 = time.time()
    gen = torch.Generator(device=env.device)
    if seed is not None:
        gen.manual_seed(int(seed))
    model.eval()
    T = env.T
    cache = model.precompute(T)
    n_calls = 0

    if mode == "greedy":
        env.reset()
        while not env.done:
            logits, _, _ = model.logits(cache, T, env)
            n_calls += 1
            mask = env.action_mask()
            logits = logits.masked_fill(~mask, float("-inf"))
            i = int(torch.argmax(logits).item())
            env.step(i)
        sel = env.selection()
        val = env.current_objective()
        return {"selection": sel, "objective": float(val), "mode": mode,
                "n_network_calls": n_calls, "n_samples": 1,
                "time": time.time() - t0 if timed else None, "env": env}

    if mode == "sample":
        best, best_val, best_env = None, -np.inf, None
        for s in range(n_samples):
            env.reset()
            while not env.done:
                logits, _, _ = model.logits(cache, T, env)
                n_calls += 1
                mask = env.action_mask()
                logits = logits.masked_fill(~mask, float("-inf")) / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=0)
                i = int(torch.multinomial(probs, 1, generator=gen).item())
                env.step(i)
            v = env.current_objective()
            if v > best_val:
                best_val, best, best_env = v, list(env.selection()), env
        return {"selection": best, "objective": float(best_val), "mode": mode,
                "n_network_calls": n_calls, "n_samples": n_samples,
                "time": time.time() - t0 if timed else None, "env": best_env}

    if mode == "beam":
        return _beam_decode(model, env, cache, T, width=beam, t0=t0, timed=timed)

    raise ValueError(f"unknown decode mode: {mode}")


# =====================================================================
@torch.no_grad()
def _beam_decode(model, env, cache, T, width: int = 4, t0: float | None = None,
                 timed: bool = True) -> dict:
    """束搜索：维护 width 条部分序列（按累计对数概率），末端在候选中按目标值选优。

    说明：束搜索不改变可行性约束（每步仍只扩展合法动作）；其目标是最大化策略似然，
    最终解由**共享评价器**在束末端候选中选取，因此与论文的目标口径一致。
    """
    K = env.K
    beams: list[tuple[float, list[int]]] = [(0.0, [])]
    n_calls = 0
    for _ in range(K):
        cands: list[tuple[float, list[int]]] = []
        for lp, seq in beams:
            env.reset()
            for i in seq:
                env.step(i)
            logits, _, _ = model.logits(cache, T, env)
            n_calls += 1
            mask = env.action_mask()
            logits = logits.masked_fill(~mask, float("-inf"))
            logp = torch.log_softmax(logits, dim=0)
            topk = torch.topk(logp, k=min(width, int(mask.sum().item())))
            for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                cands.append((lp + v, seq + [int(idx)]))
        cands.sort(key=lambda x: -x[0])
        beams = cands[:width]
    best, best_val = None, -np.inf
    for _, seq in beams:
        v = husar_objective(env.inst, seq, env.agg, env.tau)
        if v > best_val:
            best_val, best = v, seq
    return {"selection": list(best), "objective": float(best_val), "mode": "beam",
            "n_network_calls": n_calls, "n_samples": len(beams),
            "time": time.time() - t0 if timed else None, "env": None}


# 评价出口见 evaluation.py（本模块刻意不接触任何标签，以维持静态泄漏护栏）
