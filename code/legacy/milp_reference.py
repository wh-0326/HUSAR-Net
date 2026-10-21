# -*- coding: utf-8 -*-
"""
MILP 参考求解器（新增实验的"精确/界"基准）。

包含三个模型：
  1. milp_base    : 基础目标（情景均值风险 + 多尺度饱和覆盖），线性化 y_g <= n_g/b_g, y_g <= 1
  2. milp_maxmin  : max-min 鲁棒目标  max t  s.t.  t <= lam*C(S) + sum_i q_i^(b) x_i  for all b
  3. milp_cvar    : 收益下尾 CVaR（Rockafellar–Uryasev 精确线性化，**支持 tau*B 非整数**）
                    max  c - (1/(tau*B)) * sum_b u_b
                    s.t. u_b >= c - F_b(S),  u_b >= 0
                    该式对任意 tau ∈ (0,1] 与离散情景分布给出精确下尾 CVaR（含分位点部分概率质量）。

报告纪律：
  * 同时返回 incumbent 值与**对偶界**（ObjBound）。只有当求解器以 OPTIMAL 状态终止
    且 MIPGap <= 容差时，才把 incumbent 称为"已证明最优值"（proven_optimal=True）；
    否则只报告"最好整数可行解 + 有效上界"。
"""
from __future__ import annotations

import time

import numpy as np

import gurobipy as gp
from gurobipy import GRB

from objectives import USMAInstance, cvar_lower_tail


def _add_coverage(m, inst: USMAInstance, x, name="yc"):
    """加入城市/省级饱和覆盖的线性化辅助变量，返回覆盖项表达式。"""
    cov = 0.0
    if inst.omega_city > 0:
        y_c = m.addVars(inst.n_city, lb=0.0, ub=1.0, name=name)
        for g in range(inst.n_city):
            members = np.where(inst.city_idx == g)[0]
            if len(members) == 0:
                continue
            m.addConstr(y_c[g] <= gp.quicksum(x[int(i)] for i in members) / float(inst.b_city[g]))
            m.addConstr(y_c[g] <= 1.0)
        cov = cov + inst.lam * inst.omega_city * gp.quicksum(
            inst.v_city[g] * y_c[g] for g in range(inst.n_city))
    if inst.omega_prov > 0:
        y_p = m.addVars(inst.n_prov, lb=0.0, ub=1.0, name=name + "p")
        for g in range(inst.n_prov):
            members = np.where(inst.prov_idx == g)[0]
            if len(members) == 0:
                continue
            m.addConstr(y_p[g] <= gp.quicksum(x[int(i)] for i in members) / float(inst.b_prov[g]))
            m.addConstr(y_p[g] <= 1.0)
        cov = cov + inst.lam * inst.omega_prov * gp.quicksum(
            inst.v_prov[g] * y_p[g] for g in range(inst.n_prov))
    return cov


def _new_model(name, inst, K, time_limit, mip_gap, verbose):
    m = gp.Model(name)
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)
    m.Params.Threads = 0
    x = m.addVars(inst.n, vtype=GRB.BINARY, name="x")
    m.addConstr(gp.quicksum(x[i] for i in range(inst.n)) <= K)
    return m, x


def _finish(m, inst, x, t0, model_tag, extra=None):
    status = int(m.Status)
    has_sol = m.SolCount > 0
    S = [i for i in range(inst.n) if has_sol and x[i].X > 0.5]
    gap = float(m.MIPGap) if has_sol else None
    bound = float(m.ObjBound) if m.ObjBound is not None and np.isfinite(m.ObjBound) else None
    proven = bool(has_sol and status == GRB.OPTIMAL and (gap is not None and gap <= 1e-9 + 0))
    out = {"model": model_tag, "status": status,
           "status_name": {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
                           GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                           GRB.INTERRUPTED: "INTERRUPTED"}.get(status, str(status)),
           "selection": S,
           "incumbent": float(m.ObjVal) if has_sol else None,
           "dual_bound": bound,
           "mip_gap": gap,
           "proven_optimal": proven,
           "runtime_s": round(time.time() - t0, 3)}
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------------ 1) 基础目标
def milp_base(inst: USMAInstance, K: int, time_limit=600.0, mip_gap=1e-6, verbose=False):
    """基础目标精确解：max Σ q_mean_i x_i + λ·C(S)。"""
    t0 = time.time()
    m, x = _new_model("usma_base", inst, K, time_limit, mip_gap, verbose)
    cov = _add_coverage(m, inst, x)
    m.setObjective(gp.quicksum(inst.q_mean[i] * x[i] for i in range(inst.n)) + cov, GRB.MAXIMIZE)
    m.optimize()
    return _finish(m, inst, x, t0, "milp_base")


# ------------------------------------------------------------------ 2) max-min
def milp_maxmin(inst: USMAInstance, K: int, time_limit=600.0, mip_gap=1e-6, verbose=False):
    """max-min 鲁棒目标：max t s.t. t <= λ·C(S) + Σ_i q_i^(b) x_i, ∀b。"""
    t0 = time.time()
    m, x = _new_model("usma_maxmin", inst, K, time_limit, mip_gap, verbose)
    cov = _add_coverage(m, inst, x)
    t = m.addVar(lb=-GRB.INFINITY, name="t")
    Q = inst.Q
    for b in range(inst.B):
        row = Q[b]
        nz = np.where(np.abs(row) > 0)[0]
        m.addConstr(t <= cov + gp.quicksum(float(row[int(i)]) * x[int(i)] for i in nz))
    m.setObjective(t, GRB.MAXIMIZE)
    m.optimize()
    return _finish(m, inst, x, t0, "milp_maxmin")


# ------------------------------------------------------------------ 3) CVaR（精确尾部）
def milp_cvar(inst: USMAInstance, K: int, tau: float = 0.25, time_limit=600.0,
              mip_gap=1e-6, verbose=False):
    """收益下尾 CVaR 的精确 MILP（RU 线性化，支持 tau*B 非整数）。

    max  c - (1/(τB)) Σ_b u_b ;  u_b ≥ c - F_b(S) ;  u_b ≥ 0 ;  c 自由
    """
    t0 = time.time()
    m, x = _new_model("usma_cvar", inst, K, time_limit, mip_gap, verbose)
    cov = _add_coverage(m, inst, x)
    c = m.addVar(lb=-GRB.INFINITY, name="c")
    u = m.addVars(inst.B, lb=0.0, name="u")
    Q = inst.Q
    for b in range(inst.B):
        row = Q[b]
        nz = np.where(np.abs(row) > 0)[0]
        Fb = cov + gp.quicksum(float(row[int(i)]) * x[int(i)] for i in nz)
        m.addConstr(u[b] >= c - Fb)
    m_den = float(tau) * inst.B
    m.setObjective(c - (1.0 / m_den) * gp.quicksum(u[b] for b in range(inst.B)), GRB.MAXIMIZE)
    m.optimize()
    return _finish(m, inst, x, t0, "milp_cvar", extra={"tau": tau, "tauB": m_den})


# ------------------------------------------------------------------ 工具
def enumerated_optimum(inst: USMAInstance, K: int, pool: int, mode: str = "maxmin",
                       tau: float = 0.25):
    """小规模穷举最优（仅用于单元测试）：从 pool 个候选中枚举所有 C(pool,K)。"""
    from itertools import combinations
    from objectives import g_cvar as _g_cvar
    best, best_val, cnt = None, -np.inf, 0
    for S in combinations(range(pool), K):
        v = float(inst.Q[:, S].sum(axis=1).min()) if mode == "maxmin" else _g_cvar(inst, S, tau)
        cnt += 1
        if v > best_val:
            best_val, best = v, S
    return list(best), float(best_val), cnt


def objective_gap(inst: USMAInstance, S, ref_value: float, mode: str = "maxmin",
                  tau: float = 0.25) -> dict:
    """相对参考值（证明最优或有效上界）的质量差距。"""
    from objectives import g_min as _g_min, g_cvar as _g_cvar
    v = _g_min(inst, S) if mode == "maxmin" else _g_cvar(inst, S, tau)
    return {"value": float(v), "reference": float(ref_value),
            "abs_gap": float(ref_value - v),
            "rel_gap_pct": float(100.0 * (ref_value - v) / abs(ref_value)) if ref_value else None}
