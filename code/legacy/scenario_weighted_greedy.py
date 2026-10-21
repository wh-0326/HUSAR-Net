# -*- coding: utf-8 -*-
"""
SWG —— Scenario-Weighted Greedy（情景加权贪心）。

动机
----
直接对 G_min(S) = min_b F_b(S) 做贪心属于"当前绑定情景"驱动：只要最差情景不变，
所有对其它情景有改善的候选都被视为零增益，容易在并列/近并列候选间停滞。
SWG 用**平滑最坏情景收益**替代硬 min：

        R_eta(S) = -eta * log[ (1/B) * sum_b exp( -F_b(S) / eta ) ],     eta > 0

eta -> 0+ 时 R_eta -> min_b F_b；eta -> inf 时 R_eta -> mean_b F_b。
由 softmax 得情景自适应权重：

        w_b(S) = exp(-F_b(S)/eta) / sum_j exp(-F_j(S)/eta)

收益越低的情景权重越大。SWG 的构造过程使用 R_eta（或其一阶线性化）**仅用于产生候选解**；
最终解一律用统一目标 G_min / G_CVaR 评价（objectives.py）。

精确边际（本模块 variant='exact'）
---------------------------------
        Delta R_eta(i | S) = R_eta(S∪{i}) - R_eta(S)
                           = lam*DeltaC(i) - eta * log( sum_b w_b exp(-q_i^(b)/eta) )

推导：F_b(S∪{i}) = F_b(S) + q_i^(b) + lam*DeltaC(i)，代入定义并约去公共因子即得。
该式对每个候选只需一次 log-sum-exp，且**数值稳定**实现（见 _logsumexp_weighted）。

线性化评分（本模块 variant='weighted'）
-------------------------------------
        Delta_weighted(i | S) = sum_b w_b(S) * [ F_b(S∪{i}) - F_b(S) ]
                              = sum_b w_b(S) * q_i^(b) + lam*DeltaC(i)

即"把当前情景权重冻结后，对 F 取加权平均边际"。它与 Delta R_eta 的关系必须准确表述：

  (a) **Jensen 不等式给出元素级序关系**（恒成立，已由单元测试验证）
          Delta R_eta(i|S) = -eta*log( sum_b w_b exp(-q_i^(b)/eta) )
                           <= sum_b w_b q_i^(b) + lam*DeltaC(i) = Delta_weighted(i|S)
      因为 exp 为凸函数。二者之差是情景间风险在该权重下的离散度（方差型修正项）。

  (b) Delta_weighted 是 Delta R_eta 的**高温（eta 相对风险量级很大）一阶近似**：
          sum_b w_b exp(-q/eta) = 1 - (1/eta) sum_b w_b q + O(eta^-2)
      故 -eta*log(.) -> sum_b w_b q  当 eta -> inf。低温下该近似不成立。

  (c) 极限行为（关键区别，已由单元测试逐项验证）
        eta -> 0+（利用 logw_b = (F_min - F_b)/eta + O(eta)）：
          Delta R_eta  -> min_b [ F_b(S) + q_i^(b) ] - min_b F_b(S)
                        = Delta G_min(i|S)
            即**平滑目标的精确边际在低温下退化为真实的 max-min 边际**。
            这给出 SWG 的一个有意义解释：以 eta 为插值参数，SWG-exact 在
            eta -> 0 时是"直接 max-min 贪心"，在 eta -> inf 时是"情景均值贪心"。
          Delta_weighted -> q_i^(b*) + lam*DeltaC(i)
                          （b* = 当前情景收益最小者，即"绑定情景"上的取值）
        两者只在逐企业最差情景恰好等于绑定情景时才重合。
"""
from __future__ import annotations

import time

import numpy as np

from objectives import (USMAInstance, SolutionState, all_metrics, cov_gain_vec, cvar_lower_tail)

_CHUNK_BUDGET = 3_000_000     # 单个临时数组元素上限，用于大规模实例分块


def _logsumexp(a, axis=0):
    """数值稳定的 logsumexp（沿 axis）。"""
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    return (amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))).squeeze(axis)


# =====================================================================
# 温度标定
# =====================================================================
def outcome_scale(inst: USMAInstance, S_ref, kind: str = "std") -> float:
    """收益量纲标定：以参考解处情景收益 {F_b(S_ref)} 的离散程度作为 eta 的尺度。

    kind='std' -> 标准差；'iqr' -> 四分位距；'range' -> 极差。
    """
    S = list(S_ref)
    if len(S) == 0:
        return 1.0
    f = inst.Q[:, np.asarray(S, dtype=np.int64)].sum(axis=1)
    if kind == "iqr":
        return float(np.subtract(*np.percentile(f, [75, 25])))
    if kind == "range":
        return float(f.max() - f.min())
    return float(f.std(ddof=0))


def eta_from_kappa(scale: float, kappa: float) -> float:
    return float(max(scale, 1e-12) * kappa)


# =====================================================================
# 情景权重
# =====================================================================
def _lse1(z: np.ndarray) -> float:
    """一维稳定 logsumexp。"""
    m = float(z.max())
    return m + float(np.log(np.sum(np.exp(z - m))))


def log_scenario_weights(st: SolutionState, eta: float) -> np.ndarray:
    """log w_b(S) = -F_b/eta - logsumexp(-F/eta)。

    以 log 空间直接返回，避免"先归一化再取对数"在极小温度下产生 log(0) = -inf。
    """
    z = -st.F() / eta
    return z - _lse1(z)


def scenario_weights(st: SolutionState, eta: float) -> np.ndarray:
    """w_b(S) = softmax(-F_b(S)/eta)，数值稳定实现（极小温度下可能下溢为 0，属正常）。"""
    return np.exp(log_scenario_weights(st, eta))


def r_eta(st: SolutionState, eta: float) -> float:
    """R_eta(S) = -eta * log( mean_b exp(-F_b/eta) )（稳定实现）。"""
    z = -st.F() / eta
    return float(-eta * (_lse1(z) - np.log(z.size)))


# =====================================================================
# 边际评分（向量化 + 分块）
# =====================================================================
def delta_exact(inst: USMAInstance, st: SolutionState, cand, eta: float) -> np.ndarray:
    """精确 Delta R_eta(i|S)（含覆盖项），稳定 log-sum-exp。"""
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    logw = log_scenario_weights(st, eta)                     # log w_b（log 空间，稳定）
    cov = inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)
    out = np.empty(len(cand), dtype=float)
    chunk = max(1, _CHUNK_BUDGET // max(inst.B, 1))
    for s in range(0, len(cand), chunk):
        cc = cand[s:s + chunk]
        a = logw[:, None] - inst.Q[:, cc] / eta              # (B, m)
        out[s:s + chunk] = -eta * _logsumexp(a, axis=0)
    return out + cov


def delta_weighted(inst: USMAInstance, st: SolutionState, cand, eta: float) -> np.ndarray:
    """线性化评分 Delta_weighted(i|S) = sum_b w_b q_i^(b) + lam*DeltaC(i)（近似，非精确）。"""
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    w = scenario_weights(st, eta)
    cov = inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)
    out = np.empty(len(cand), dtype=float)
    chunk = max(1, _CHUNK_BUDGET // max(inst.B, 1))
    for s in range(0, len(cand), chunk):
        cc = cand[s:s + chunk]
        out[s:s + chunk] = w @ inst.Q[:, cc]
    return out + cov


# =====================================================================
# 主算法
# =====================================================================
def swg(inst: USMAInstance, K: int, eta: float, variant: str = "exact",
        time_budget: float | None = None, track_anytime: bool = False,
        fix_weights_at: int | None = None, pool=None) -> dict:
    """情景加权贪心构造。

    Parameters
    ----------
    eta : 平滑温度（>0）。小 -> 接近 min，大 -> 接近 mean。
    variant : 'exact'（精确 Delta R_eta）或 'weighted'（加权线性化评分）。
    fix_weights_at : 若给定 t，则前 t 步用 exact 权重，之后冻结第 t 步的情景权重
                     （用于检验"动态更新权重"这一组件是否必要）。
    pool : 可选候选池（企业在 n 中的下标数组）。在大规模实例上把每轮的候选集限制为
           候选池，使每轮代价由 O(B·n) 降为 O(B·|pool|)。候选池由统一规则生成
           （见 build_candidate_pool），所有受影响的构造型算法使用**同一**候选池，
           并在结果中记录 pool_size，以保证方法间可比。
    """
    if variant not in ("exact", "weighted"):
        raise ValueError(variant)
    t0 = time.time()
    st = SolutionState(inst, [])
    S = []
    evals = 0
    anytime = []
    frozen_w = None
    while len(S) < K:
        if pool is None:
            cand = np.where(~st.inS)[0]
        else:
            cand = pool[~st.inS[pool]]
        if len(cand) == 0:
            break
        if fix_weights_at is not None and len(S) >= fix_weights_at and frozen_w is not None:
            cov = inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)
            if variant == "weighted":
                m = ((np.exp(frozen_w) @ inst.Q[:, cand])) + cov
            else:
                a = frozen_w[:, None] - inst.Q[:, cand] / eta
                m = -eta * _logsumexp(a, axis=0) + cov
        else:
            m = delta_exact(inst, st, cand, eta) if variant == "exact" \
                else delta_weighted(inst, st, cand, eta)
            if fix_weights_at is not None and len(S) == fix_weights_at - 1:
                frozen_w = log_scenario_weights(st, eta).copy()
        evals += len(cand)
        i = int(cand[int(np.argmax(m))])
        st.add(i)
        S.append(i)
        if track_anytime:
            anytime.append((time.time() - t0, st.f_min()))
        if time_budget is not None and time.time() - t0 > time_budget:
            break
    st.validate()
    met = all_metrics(inst, S)
    out = {"method": f"swg_{variant}", "selection": S, "n": len(S),
           "g_min": met["g_min"], "g_cvar": met["g_cvar_tau"], "g_mean": met["g_mean"],
           "cities_covered": met["cities_covered"], "provs_covered": met["provs_covered"],
           "time": time.time() - t0, "evals": evals, "eta": float(eta),
           "variant": variant, "fix_weights_at": fix_weights_at,
           "pool_size": None if pool is None else int(len(pool)),
           "r_eta_final": r_eta(st, eta)}
    if track_anytime:
        out["anytime"] = anytime
    return out


def swg_variants(inst: USMAInstance, K: int, kappa: float = 1.0, scale_kind: str = "std",
                 S_ref=None) -> list[dict]:
    """在同一标定下运行 exact / weighted 两个变体（用于对照与敏感性）。"""
    if S_ref is None:
        from baselines import celf_greedy
        S_ref = celf_greedy(inst, K, proxy="lcb")["selection"]
    scale = outcome_scale(inst, S_ref, kind=scale_kind)
    eta = eta_from_kappa(scale, kappa)
    out = []
    for v in ("exact", "weighted"):
        r = swg(inst, K, eta=eta, variant=v)
        r["kappa"] = kappa
        r["scale"] = scale
        r["scale_kind"] = scale_kind
        out.append(r)
    return out
