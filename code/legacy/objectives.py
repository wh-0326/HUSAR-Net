# -*- coding: utf-8 -*-
"""
USMA 统一目标函数（新增启发式实验专用，全实验唯一口径）。

与原文一致的目标定义（论文第 2.2 节 / 式 (1)-(3)）：

    F_b(S) = sum_{i in S} q_i^(b) + lam * C(S),        b = 1..B
    C(S)   = sum_l omega_l * sum_{g in G_l} v_g * min{1, |S ∩ V_g| / b_g}

主（第一优先级）目标 —— max-min 鲁棒收益：

    G_min(S) = min_b F_b(S)

次目标 —— 收益下尾 CVaR（离散精确尾部定义，见 cvar_lower_tail）：

    G_CVaR(S) = CVaR_{tau}( {F_b(S)}_{b=1..B} )

设计约定（保证"所有算法在同一原始评价目标下比较"）：
  * 所有启发式（SWG / ALNS / 基线）内部用于**构造候选解**的代理目标
    （LCB 分位数、情景加权平滑目标等）都在本模块之外实现；
  * 但**最终求解质量**一律由本模块的 g_min / g_cvar 评价（single source of truth）。

本模块只依赖 numpy。正确性由 tests/test_objectives.py 对拍验证：
  - 与 research_upgrade/src/spatial_objectives.py + src/usma_solver.py 的原始实现逐步一致。
"""
from __future__ import annotations

import numpy as np

# 覆盖项的结构常数（与原文第 4.3 节一致）
ALPHA = 0.02      # 饱和比例
B_CAP = 10        # 饱和阈值上界
B_FLOOR = 1       # 饱和阈值下界
OMEGA_CITY = 0.5
OMEGA_PROV = 0.5
LAM = 0.10
TAU = 0.25


# =====================================================================
# 实例容器
# =====================================================================
class USMAInstance:
    """一个 USMA 配置实例的全部静态数据。

    Attributes
    ----------
    n : 候选企业数
    B : 情景数
    Q : (B, n) 情景风险分 q_i^(b)（float64）
    q_mean, q_lcb : (n,) 逐企业情景均值 / 低分位数（仅用于构造代理目标）
    city_idx, prov_idx : (n,) 地级单元 / 省级单元序号
    n_firm_city, n_firm_prov : 各单元在册企业数
    b_city, b_prov : 饱和阈值 b_g = clip(round(alpha * n_g), 1, 10)
    v_city, v_prov : 区域覆盖权重（默认 1）
    lam, omega_city, omega_prov : 权衡与层级权重
    """

    __slots__ = ("n", "B", "Q", "q_mean", "q_lcb", "city_idx", "prov_idx",
                 "n_firm_city", "n_firm_prov", "b_city", "b_prov",
                 "v_city", "v_prov", "lam", "omega_city", "omega_prov",
                 "name", "meta")

    def __init__(self, Q, city_idx, prov_idx, n_firm_city, n_firm_prov,
                 lam=LAM, alpha=ALPHA, b_cap=B_CAP, omega_city=OMEGA_CITY,
                 omega_prov=OMEGA_PROV, v_city=None, v_prov=None,
                 q_mean=None, q_lcb=None, city_scale=True, prov_scale=True,
                 name="instance", meta=None):
        self.Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float64))
        self.B, self.n = self.Q.shape
        self.city_idx = np.asarray(city_idx, dtype=np.int64)
        self.prov_idx = np.asarray(prov_idx, dtype=np.int64)
        self.n_firm_city = np.asarray(n_firm_city, dtype=float)
        self.n_firm_prov = np.asarray(n_firm_prov, dtype=float)
        self.lam = float(lam)
        self.b_city = np.clip(np.round(alpha * self.n_firm_city), B_FLOOR, b_cap)
        self.b_prov = np.clip(np.round(alpha * self.n_firm_prov), B_FLOOR, b_cap)
        self.v_city = np.ones(len(self.n_firm_city)) if v_city is None else np.asarray(v_city, float)
        self.v_prov = np.ones(len(self.n_firm_prov)) if v_prov is None else np.asarray(v_prov, float)
        self.omega_city = float(omega_city) if city_scale else 0.0
        self.omega_prov = float(omega_prov) if prov_scale else 0.0
        # 代理分（仅用于构造候选解；不参与最终评价）
        self.q_mean = self.Q.mean(axis=0) if q_mean is None else np.asarray(q_mean, float)
        self.q_lcb = self.q_mean if q_lcb is None else np.asarray(q_lcb, float)
        self.name = name
        self.meta = dict(meta or {})

    # -------------------------------------------------- 便捷属性
    @property
    def n_city(self):
        return len(self.n_firm_city)

    @property
    def n_prov(self):
        return len(self.n_firm_prov)

    def copy_with(self, name=None, **kw):
        """浅拷贝并替换若干字段（用于比较不同 lam/权重下的同一实例）。"""
        d = dict(Q=self.Q, city_idx=self.city_idx, prov_idx=self.prov_idx,
                 n_firm_city=self.n_firm_city, n_firm_prov=self.n_firm_prov,
                 lam=self.lam, omega_city=self.omega_city or OMEGA_CITY,
                 omega_prov=self.omega_prov or OMEGA_PROV,
                 v_city=self.v_city, v_prov=self.v_prov,
                 q_mean=self.q_mean, q_lcb=self.q_lcb,
                 city_scale=self.omega_city > 0, prov_scale=self.omega_prov > 0,
                 name=self.name if name is None else name, meta=dict(self.meta))
        d.update(kw)
        return USMAInstance(**d)


# =====================================================================
# 覆盖项 C(S) 及其边际（O(1) 增量）
# =====================================================================
def coverage_value(inst: USMAInstance, n_sel_city, n_sel_prov) -> float:
    """C(S) = sum_l omega_l * sum_g v_g * min(1, n_g/b_g)。"""
    cov = 0.0
    if inst.omega_city > 0:
        cov += inst.omega_city * float((inst.v_city * np.minimum(1.0, n_sel_city / inst.b_city)).sum())
    if inst.omega_prov > 0:
        cov += inst.omega_prov * float((inst.v_prov * np.minimum(1.0, n_sel_prov / inst.b_prov)).sum())
    return cov


def _cov_inc_city(inst, n_sel_city, g):
    if inst.omega_city <= 0:
        return 0.0
    return inst.omega_city * inst.v_city[g] / inst.b_city[g] if n_sel_city[g] < inst.b_city[g] else 0.0


def _cov_inc_prov(inst, n_sel_prov, gp):
    if inst.omega_prov <= 0:
        return 0.0
    return inst.omega_prov * inst.v_prov[gp] / inst.b_prov[gp] if n_sel_prov[gp] < inst.b_prov[gp] else 0.0


def cov_gain_vec(inst: USMAInstance, n_sel_city, n_sel_prov, cand) -> np.ndarray:
    """向量化：对候选集合 cand 中每个 i，返回 C(S∪{i}) - C(S)（未乘 lam）。"""
    cand = np.asarray(cand, dtype=np.int64)
    out = np.zeros(len(cand), dtype=float)
    if len(cand) == 0:
        return out
    if inst.omega_city > 0:
        g = inst.city_idx[cand]
        act = n_sel_city[g] < inst.b_city[g]
        out += np.where(act, inst.omega_city * inst.v_city[g] / inst.b_city[g], 0.0)
    if inst.omega_prov > 0:
        gp = inst.prov_idx[cand]
        act = n_sel_prov[gp] < inst.b_prov[gp]
        out += np.where(act, inst.omega_prov * inst.v_prov[gp] / inst.b_prov[gp], 0.0)
    return out


def coverage_components(inst: USMAInstance, members) -> dict:
    members = np.asarray(list(members), dtype=np.int64)
    n_city = len(inst.n_firm_city)
    n_prov = len(inst.n_firm_prov)
    if len(members):
        nsel_c = np.bincount(inst.city_idx[members], minlength=n_city).astype(float)
        nsel_p = np.bincount(inst.prov_idx[members], minlength=n_prov).astype(float)
    else:
        nsel_c = np.zeros(n_city)
        nsel_p = np.zeros(n_prov)
    return {"n_sel_city": nsel_c, "n_sel_prov": nsel_p,
            "cov_city": float((inst.v_city * np.minimum(1.0, nsel_c / inst.b_city)).sum()),
            "cov_prov": float((inst.v_prov * np.minimum(1.0, nsel_p / inst.b_prov)).sum()),
            "cities_covered": int((nsel_c > 0).sum()), "provs_covered": int((nsel_p > 0).sum())}


# =====================================================================
# 全量（非增量）目标评价
# =====================================================================
def scen_values(inst: USMAInstance, S) -> np.ndarray:
    """返回 (B,) 向量 F_b(S) = lam*C(S) + sum_{i in S} Q[b,i]。"""
    S = np.asarray(list(S), dtype=np.int64)
    cov = coverage_components(inst, S)
    c = inst.lam * (inst.omega_city * cov["cov_city"] + inst.omega_prov * cov["cov_prov"])
    if len(S) == 0:
        return np.full(inst.B, c)
    return inst.Q[:, S].sum(axis=1) + c


def g_min(inst: USMAInstance, S) -> float:
    """G_min(S) = min_b F_b(S)。"""
    v = scen_values(inst, S)
    return float(v.min())


def g_mean(inst: USMAInstance, S) -> float:
    v = scen_values(inst, S)
    return float(v.mean())


def cvar_lower_tail(values, tau: float) -> float:
    """收益下尾 CVaR 的**离散精确**定义（正确处理分位点处的部分概率质量）。

    令 m = tau*B, j = floor(m), frac = m - j，values 升序为 v_(1) <= ... <= v_(B)，则

        CVaR_tau = (1/m) * [ sum_{r=1..j} v_(r) + frac * v_(j+1) ]

    即"最差 tau 比例概率质量上收益的均值"。当 m 为整数时退化为最差 m 个情景的算术平均。
    当 m 非整数时**不**使用 ceil/round 近似（例如 B=150, tau=0.25 => m=37.5：
    取最差 37 个情景之和再加最差第 38 个情景的 0.5 倍，最后除以 37.5）。
    并列值（ties）不改变结果，因为定义基于排序后的取值而非情景标签。
    """
    v = np.asarray(values, dtype=float).ravel()
    n = v.size
    m = float(tau) * n
    if not (0.0 < float(tau) <= 1.0):
        raise ValueError("tau must be in (0, 1]")
    if m >= n - 1e-12:                     # tau == 1：退化为全体均值
        return float(v.mean())
    j = int(np.floor(m + 1e-12))
    frac = m - j
    sv = np.sort(v)
    if frac <= 1e-12:
        return float(sv[:j].mean())
    return float((sv[:j].sum() + frac * sv[j]) / m)


def g_cvar(inst: USMAInstance, S, tau: float = TAU) -> float:
    """G_CVaR(S)：收益下尾 CVaR（离散精确尾部定义）。"""
    return cvar_lower_tail(scen_values(inst, S), tau)


def all_metrics(inst: USMAInstance, S, tau: float = TAU) -> dict:
    """统一评价：鲁棒主/次目标 + 拆解项 + 结构指标。"""
    S = np.asarray(list(S), dtype=np.int64)
    v = scen_values(inst, S)
    cov = coverage_components(inst, S)
    risk = inst.Q[:, S].sum(axis=1) if len(S) else np.zeros(inst.B)
    cov_part = inst.lam * (inst.omega_city * cov["cov_city"] + inst.omega_prov * cov["cov_prov"])
    dn = int(v.argmin())
    return {
        "n": int(len(S)),
        "g_min": float(v.min()),
        "g_cvar_tau": cvar_lower_tail(v, tau),
        "g_mean": float(v.mean()),
        "g_max": float(v.max()),
        "g_std": float(v.std(ddof=0)),
        "worst_scenario": dn,
        "cov_term": float(cov_part),
        "cov_city": cov["cov_city"], "cov_prov": cov["cov_prov"],
        "cities_covered": cov["cities_covered"], "provs_covered": cov["provs_covered"],
        "risk_sum_worst": float(risk[dn]),
        "risk_sum_mean": float(risk.mean()),
    }


# =====================================================================
# 增量状态：删除/插入后 O(B) 维护，避免每次全量重算
# =====================================================================
class SolutionState:
    """可增删的候选解状态，维护：
        n_sel_city / n_sel_prov（辖区计数）
        scen_sums  (B,)   —— 已选企业的情景风险分之和
        cov_city / cov_prov —— 覆盖项累计值
    add/remove 的更新与全量重算在 tests 中逐操作对拍。
    """

    __slots__ = ("inst", "inS", "n_sel_city", "n_sel_prov", "scen_sums",
                 "cov_city", "cov_prov", "members")

    def __init__(self, inst: USMAInstance, members=()):
        self.inst = inst
        self.inS = np.zeros(inst.n, dtype=bool)
        self.n_sel_city = np.zeros(len(inst.n_firm_city), dtype=float)
        self.n_sel_prov = np.zeros(len(inst.n_firm_prov), dtype=float)
        self.scen_sums = np.zeros(inst.B, dtype=float)
        self.cov_city = 0.0
        self.cov_prov = 0.0
        self.members = []
        for i in members:
            self.add(int(i))

    # -------------------------------------------------- 基础操作
    def add(self, i: int) -> None:
        if self.inS[i]:
            raise ValueError(f"firm {i} already selected")
        inst = self.inst
        g = inst.city_idx[i]
        # cov_city / cov_prov 始终维护**未乘 omega** 的原始覆盖和；omega 在 cov_term 中施加
        if self.n_sel_city[g] < inst.b_city[g]:
            self.cov_city += inst.v_city[g] / inst.b_city[g]
        self.n_sel_city[g] += 1.0
        gp = inst.prov_idx[i]
        if self.n_sel_prov[gp] < inst.b_prov[gp]:
            self.cov_prov += inst.v_prov[gp] / inst.b_prov[gp]
        self.n_sel_prov[gp] += 1.0
        self.scen_sums += inst.Q[:, i]
        self.inS[i] = True
        self.members.append(i)

    def remove(self, i: int) -> None:
        if not self.inS[i]:
            raise ValueError(f"firm {i} not selected")
        inst = self.inst
        g = inst.city_idx[i]
        self.n_sel_city[g] -= 1.0
        if self.n_sel_city[g] < inst.b_city[g]:
            self.cov_city -= inst.v_city[g] / inst.b_city[g]
        gp = inst.prov_idx[i]
        self.n_sel_prov[gp] -= 1.0
        if self.n_sel_prov[gp] < inst.b_prov[gp]:
            self.cov_prov -= inst.v_prov[gp] / inst.b_prov[gp]
        self.scen_sums -= inst.Q[:, i]
        self.inS[i] = False
        self.members.remove(i)

    # -------------------------------------------------- 目标查询
    @property
    def cov_term(self) -> float:
        inst = self.inst
        return inst.lam * (inst.omega_city * self.cov_city + inst.omega_prov * self.cov_prov)

    def F(self) -> np.ndarray:
        return self.scen_sums + self.cov_term

    def f_min(self) -> float:
        return float(self.scen_sums.min()) + self.cov_term

    def f_cvar(self, tau: float = TAU) -> float:
        return cvar_lower_tail(self.F(), tau)

    def members_arr(self) -> np.ndarray:
        return np.asarray(self.members, dtype=np.int64)

    # -------------------------------------------------- 校验
    def validate(self, tol: float = 1e-9) -> None:
        """与全量重算对拍，任何不一致直接抛错（实验中每轮调用）。"""
        inst = self.inst
        ref = coverage_components(inst, self.members)
        assert abs(ref["cov_city"] - self.cov_city) < 1e-9 * max(1.0, abs(ref["cov_city"]))
        assert abs(ref["cov_prov"] - self.cov_prov) < 1e-9 * max(1.0, abs(ref["cov_prov"]))
        assert np.array_equal(ref["n_sel_city"], self.n_sel_city)
        assert np.array_equal(ref["n_sel_prov"], self.n_sel_prov)
        ref_sums = inst.Q[:, self.members_arr()].sum(axis=1) if self.members else np.zeros(inst.B)
        assert np.max(np.abs(ref_sums - self.scen_sums)) < 1e-8
        assert self.inS.sum() == len(self.members)

    def copy(self) -> "SolutionState":
        s = SolutionState.__new__(SolutionState)
        s.inst = self.inst
        s.inS = self.inS.copy()
        s.n_sel_city = self.n_sel_city.copy()
        s.n_sel_prov = self.n_sel_prov.copy()
        s.scen_sums = self.scen_sums.copy()
        s.cov_city = self.cov_city
        s.cov_prov = self.cov_prov
        s.members = list(self.members)
        return s


# =====================================================================
# 向量化边际增益（供各启发式共用）
# =====================================================================
def marginal_gmin(inst: USMAInstance, st: SolutionState, cand) -> np.ndarray:
    """G_min(S∪{i}) - G_min(S) 的精确值（i 遍历 cand）。

    = min_b (scen_sums_b + Q[b,i]) - min_b scen_sums_b + lam * ΔC(i)
    """
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    cur_min = float(st.scen_sums.min())
    new_min = (st.scen_sums[:, None] + inst.Q[:, cand]).min(axis=0)
    return (new_min - cur_min) + inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)


def marginal_gcvar(inst: USMAInstance, st: SolutionState, cand, tau: float = TAU) -> np.ndarray:
    """G_CVaR(S∪{i}) - G_CVaR(S) 的精确值（离散精确尾部，np.partition 实现）。

    注意：覆盖项 lam*C 对全部情景是同一常数，因此
        G_CVaR(S) = CVaR_tau(sums) + lam*C(S)
        Δ = [CVaR_tau(sums + q_i) - CVaR_tau(sums)] + lam*ΔC(i)
    这里必须减去**仅含风险项**的当前 CVaR，而不是含覆盖项的整体值。
    """
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    B = inst.B
    m = tau * B
    cur_risk = cvar_lower_tail(st.scen_sums, tau)
    cov_inc = inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)
    if m >= B - 1e-12:
        new = (st.scen_sums[:, None] + inst.Q[:, cand]).mean(axis=0)
    else:
        j = int(np.floor(m + 1e-12))
        frac = m - j
        sums2 = st.scen_sums[:, None] + inst.Q[:, cand]
        part = np.partition(sums2, j, axis=0)
        if frac <= 1e-12:
            new = part[:j].mean(axis=0)
        else:
            new = (part[:j].sum(axis=0) + frac * part[j]) / m
    return new + cov_inc - cur_risk


def marginal_lcb_score(inst: USMAInstance, st: SolutionState, cand) -> np.ndarray:
    """LCB 代理目标的精确边际：q_i^LCB + lam*ΔC(i)（单调次模，用于构造候选解）。"""
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    return inst.q_lcb[cand] + inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)


def marginal_mean_score(inst: USMAInstance, st: SolutionState, cand) -> np.ndarray:
    """情景均值风险 + lam*ΔC(i)（单调次模基础目标）。"""
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    return inst.q_mean[cand] + inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)


def marginal_worst_contrib(inst: USMAInstance, st: SolutionState, cand) -> np.ndarray:
    """逐企业最差情景贡献 w_i = min_b Q[b,i]（去掉 Q 行依赖的向量化形式）。"""
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    return inst.Q[:, cand].min(axis=0)
