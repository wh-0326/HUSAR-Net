# -*- coding: utf-8 -*-
"""
基线方法（新增实验对照，全部在统一目标 objectives.py 下评价）。

方法清单
--------
celf_base          : 基础目标（情景均值风险 q_mean + 多尺度饱和覆盖）的 CELF 惰性贪心
                     —— 复刻原文 greedy_solve(lazy=True)，单调次模，有 (1-1/e) 保证
greedy_lcb         : LCB 代理目标（逐企业 10% 分位数 q_lcb + 覆盖）的 CELF 惰性贪心
                     —— 原文 B7a / USMA-C，单调次模
direct_maxmin_true : **正确实现**的 max-min 直接贪心（每轮全量重算 G_min 的精确边际）
legacy_maxmin      : 复刻原文 ScenarioMinProblem.marginal 的贪心（审计用；该边际实现
                     恒等于 lam·ΔC，故实质是"仅覆盖"贪心，见 experiment_audit.md §3）
local_search_1swap : 复刻原文 local_search（候选裁剪 80 在选 / 150 未选，首改进接受，≤60 轮）
rank_lcb / rank_mean : 纯风险排序（无空间项）
proportional_quota : 按城市在册企业数比例配额 + 城市内随机（B1 基线，仅真实数据）

统一返回 dict：{"method", "selection", "g_min", "g_cvar", "time", "evals", "extra"}
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from objectives import (USMAInstance, SolutionState, g_min, g_cvar, all_metrics,
                        marginal_gmin, marginal_lcb_score, marginal_mean_score)

# 原文局部搜索的候选裁剪与轮数（保持不变以便公平对照）
LS_N_REMOVAL = 80
LS_N_ADDITION = 150
LS_MAX_ROUNDS = 60
LS_TOL = 1e-10


def _pack(inst, S, t0, method, evals=None, **extra):
    S = [int(i) for i in S]
    met = all_metrics(inst, S)
    out = {"method": method, "selection": S, "n": len(S),
           "g_min": met["g_min"], "g_cvar": met["g_cvar_tau"], "g_mean": met["g_mean"],
           "cities_covered": met["cities_covered"], "provs_covered": met["provs_covered"],
           "time": time.time() - t0, "evals": evals}
    out.update(extra)
    return out


def check_selection(inst: USMAInstance, S, K: int, require_full: bool = True) -> None:
    """约束校验：无重复、数量不超过 K（默认要求恰好 K）。"""
    S = [int(i) for i in S]
    assert len(S) == len(set(S)), "selection contains duplicates"
    assert all(0 <= i < inst.n for i in S), "index out of range"
    if require_full:
        assert len(S) == K, f"|S|={len(S)} != K={K}"
    else:
        assert len(S) <= K


# =====================================================================
# 1. CELF 惰性贪心（单调次模代理目标）
# =====================================================================
def celf_greedy(inst: USMAInstance, K: int, proxy: str = "base", lazy: bool = True) -> dict:
    """proxy ∈ {'base','lcb'}：分别对应 q_mean（基础目标）与 q_lcb（LCB 代理）。

    采用 Minoux/CELF 思想的大顶堆惰性求值。代理目标单调次模，故惰性剪枝有效，
    结果与朴素全扫描贪心一致（由 tests 验证）。
    """
    t0 = time.time()
    score = inst.q_mean if proxy == "base" else inst.q_lcb
    st = SolutionState(inst, [])
    n = inst.n
    evals = 0
    if lazy:
        heap = []
        for i in range(n):
            m = float(score[i]) + inst.lam * float(
                _cov_inc(inst, st, i))
            heap.append((-m, i, 0))
            evals += 1
        heapq.heapify(heap)
        S = []
        while len(S) < K:
            neg, i, stamp = heapq.heappop(heap)
            if stamp == len(S):
                S.append(i)
                st.add(i)
            else:
                m = float(score[i]) + inst.lam * float(_cov_inc(inst, st, i))
                evals += 1
                heapq.heappush(heap, (-m, i, len(S)))
    else:
        S = []
        for _ in range(K):
            gains = marginal_mean_score(inst, st, np.arange(n)) if proxy == "base" \
                else marginal_lcb_score(inst, st, np.arange(n))
            gains[st.inS] = -np.inf
            evals += n
            i = int(np.argmax(gains))
            S.append(i)
            st.add(i)
    st.validate()
    return _pack(inst, S, t0, f"celf_{proxy}", evals=evals, proxy=proxy)


def _cov_inc(inst: USMAInstance, st: SolutionState, i: int) -> float:
    g = inst.city_idx[i]
    gp = inst.prov_idx[i]
    inc = 0.0
    if inst.omega_city > 0 and st.n_sel_city[g] < inst.b_city[g]:
        inc += inst.omega_city * inst.v_city[g] / inst.b_city[g]
    if inst.omega_prov > 0 and st.n_sel_prov[gp] < inst.b_prov[gp]:
        inc += inst.omega_prov * inst.v_prov[gp] / inst.b_prov[gp]
    return inc


# =====================================================================
# 2. 直接 max-min 贪心（正确实现）
# =====================================================================
def build_candidate_pool(inst: USMAInstance, K: int, factor: int = 4, floor: int = 2000):
    """统一候选池：按 LCB 代理分降序取前 max(factor*K, floor) 个企业。

    大规模实例（n 很大）上，全量重扫型构造算法每轮代价为 O(B·n)。为保证实验可运行，
    对 n 超过阈值的实例，所有"全量重扫型"构造算法（direct_maxmin_true 与 swg）统一使用
    本函数生成的候选池，并在结果中记录 pool_size。排名型与 CELF 惰性贪心不依赖该池
    （其复杂度本就与 n 的关系很弱），因此不受影响。
    """
    M = min(inst.n, max(factor * int(K), int(floor)))
    return np.argsort(-inst.q_lcb)[:M].astype(np.int64)


def direct_maxmin_true(inst: USMAInstance, K: int, time_budget: float | None = None,
                       track_anytime: bool = False, pool=None) -> dict:
    """每轮在**当前状态**下对所有候选精确重算 G_min 的边际增益，取最大者加入。

    pool 给定时，候选集限制为候选池中尚未被选择的元素（与 swg 使用同一候选池）。
    """
    t0 = time.time()
    st = SolutionState(inst, [])
    S = []
    evals = 0
    anytime = []
    while len(S) < K:
        if pool is None:
            cand = np.where(~st.inS)[0]
        else:
            cand = pool[~st.inS[pool]]
        if len(cand) == 0:
            break
        m = marginal_gmin(inst, st, cand)
        evals += len(cand)
        i = int(cand[int(np.argmax(m))])
        st.add(i)
        S.append(i)
        if track_anytime:
            anytime.append((time.time() - t0, st.f_min()))
        if time_budget is not None and time.time() - t0 > time_budget:
            break
    st.validate()
    out = _pack(inst, S, t0, "direct_maxmin_true", evals=evals)
    out["pool_size"] = None if pool is None else int(len(pool))
    if track_anytime:
        out["anytime"] = anytime
    return out


# =====================================================================
# 3. 原文（有缺陷的）max-min 贪心实现——审计与复现用
# =====================================================================
def legacy_maxmin_greedy(inst: USMAInstance, K: int, lazy: bool = True) -> dict:
    """复刻 research_upgrade/src/usma_solver.py 的 `ScenarioMinProblem.marginal`：

        new_min = min( cur_min, min_b( sums_b + Q[b,i] ) )
        marginal = new_min - cur_min + lam*ΔC

    因为 Q >= 0 使 min_b(sums_b + Q[b,i]) >= min_b(sums_b) = cur_min，
    该式恒等于 lam*ΔC(i)，与风险项无关。此函数仅用于复现论文报告值（45.10 / 78.60）
    并量化该实现缺陷的影响，**不作为"直接 max-min 贪心"的科学基准**。
    """
    t0 = time.time()
    n = inst.n
    n_c = np.zeros(inst.n_city)
    n_p = np.zeros(inst.n_prov)
    cur_sums = np.zeros(inst.B)
    cur_min = 0.0
    S = []
    evals = 0

    def marginal(i):
        new_min = float(np.minimum(cur_min, cur_sums + inst.Q[:, i]).min())
        g = inst.city_idx[i]
        gp = inst.prov_idx[i]
        inc = 0.0
        if inst.omega_city > 0 and n_c[g] < inst.b_city[g]:
            inc += inst.lam * inst.omega_city * inst.v_city[g] / inst.b_city[g]
        if inst.omega_prov > 0 and n_p[gp] < inst.b_prov[gp]:
            inc += inst.lam * inst.omega_prov * inst.v_prov[gp] / inst.b_prov[gp]
        return float(new_min - cur_min + inc)

    if lazy:
        heap = [(-marginal(i), i, 0) for i in range(n)]
        evals += n
        heapq.heapify(heap)
        while len(S) < K:
            neg, i, stamp = heapq.heappop(heap)
            if stamp == len(S):
                S.append(i)
                cur_sums = cur_sums + inst.Q[:, i]
                n_c[inst.city_idx[i]] += 1
                n_p[inst.prov_idx[i]] += 1
            else:
                m = marginal(i)
                evals += 1
                heapq.heappush(heap, (-m, i, len(S)))
    else:
        avail = np.ones(n, bool)
        for _ in range(K):
            best_i, best_m = -1, -np.inf
            for i in range(n):
                if avail[i]:
                    m = marginal(i)
                    evals += 1
                    if m > best_m:
                        best_m, best_i = m, i
            S.append(best_i)
            avail[best_i] = False
            cur_sums = cur_sums + inst.Q[:, best_i]
            n_c[inst.city_idx[best_i]] += 1
            n_p[inst.prov_idx[best_i]] += 1
    return _pack(inst, S, t0, "legacy_maxmin_greedy", evals=evals, lazy=lazy)


# =====================================================================
# 4. 1-Swap 局部搜索（复刻原文候选裁剪与接受规则）
# =====================================================================
def local_search_1swap(inst: USMAInstance, S_init, time_budget: float | None = None,
                       max_rounds: int = LS_MAX_ROUNDS, n_removal: int = LS_N_REMOVAL,
                       n_addition: int = LS_N_ADDITION, track_anytime: bool = False,
                       verify: bool = False) -> dict:
    """原文 local_search 的等价增量实现（首改进、单交换、候选裁剪）。

    等价性由 tests/test_baselines.py::test_local_search_equivalence 与原始实现逐步对拍。
    """
    t0 = time.time()
    st = SolutionState(inst, [int(i) for i in S_init])
    cur = st.f_min()
    swaps = 0
    evals = 0
    anytime = []
    for _ in range(max_rounds):
        inS = st.inS
        sel = st.members_arr()
        w_in = inst.Q[:, sel].min(axis=0)
        order_in = sel[np.argsort(w_in)[: min(len(sel), n_removal)]]
        outside = np.where(~inS)[0]
        if len(outside) == 0:
            break
        w_out = inst.Q[:, outside].min(axis=0)
        order_out = outside[np.argsort(w_out)[::-1][: min(len(outside), n_addition)]]
        improved = False
        for a in order_in:
            a = int(a)
            if not st.inS[a]:
                continue
            for b in order_out:
                b = int(b)
                if st.inS[b]:
                    continue
                st.remove(a)
                st.add(b)
                evals += 1
                v2 = st.f_min()
                if v2 > cur + LS_TOL:
                    cur = v2
                    swaps += 1
                    improved = True
                    break
                st.remove(b)
                st.add(a)
            if improved:
                break
        if verify:
            st.validate()
        if track_anytime:
            anytime.append((time.time() - t0, cur))
        if not improved:
            break
        if time_budget is not None and time.time() - t0 > time_budget:
            break
    st.validate()
    out = _pack(inst, st.members, t0, "local_search_1swap", evals=evals, swaps=swaps)
    if track_anytime:
        out["anytime"] = anytime
    return out


# =====================================================================
# 5. 纯风险排序
# =====================================================================
def rank_select(inst: USMAInstance, K: int, score: str = "lcb", seed: int = 7) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    v = inst.q_lcb if score == "lcb" else (inst.q_mean if score == "mean" else
                                           inst.Q.min(axis=0))
    tie = rng.random(inst.n)
    order = np.lexsort((-tie, -v))
    S = order[:K].tolist()
    return _pack(inst, S, t0, f"rank_{score}", evals=0)


def rank_worst_scen(inst: USMAInstance, K: int, seed: int = 7) -> dict:
    """按逐企业最差情景贡献 w_i = min_b Q[b,i] 排序（不含覆盖项）。"""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    v = inst.Q.min(axis=0)
    tie = rng.random(inst.n)
    order = np.lexsort((-tie, -v))
    return _pack(inst, order[:K].tolist(), t0, "rank_worst_scen", evals=0)


# =====================================================================
# 6. 比例配额（B1 基线；仅用于真实数据）
# =====================================================================
def proportional_quota(inst: USMAInstance, K: int, reps: int = 500, seed: int = 7) -> dict:
    """按城市在册企业数比例（最大余数法）配额、城市内随机抽取；返回期望命中频率更适用于召回评价，
    此处返回一次确定性抽样（seed 固定）用于鲁棒目标评价。"""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n_city = inst.n_firm_city
    quota_f = K * (n_city / n_city.sum())
    quota = np.floor(quota_f).astype(int)
    rem = int(K - quota.sum())
    if rem > 0:
        order = np.argsort(-(quota_f - quota))
        quota[order[:rem]] += 1
    members = {g: np.where(inst.city_idx == g)[0] for g in range(inst.n_city)}
    sel = []
    for g, q in enumerate(quota):
        if q <= 0 or len(members[g]) == 0:
            continue
        sel.extend(rng.choice(members[g], size=min(int(q), len(members[g])), replace=False).tolist())
    if len(sel) < K:
        pool = np.setdiff1d(np.arange(inst.n), np.asarray(sel, dtype=int))
        sel.extend(rng.choice(pool, size=K - len(sel), replace=False).tolist())
    return _pack(inst, sel[:K], t0, "proportional_quota", evals=0)
