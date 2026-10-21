# -*- coding: utf-8 -*-
"""
HUSAR 目标评价器（Hierarchical Uncertainty-Aware Spatial Allocation objective）。

本模块是全项目**唯一**的目标评价口径：HUSAR-Net 训练环境、全部基线、全部消融实验
与最终论文数字都通过本模块计算；禁止在别处重新实现目标函数。

目标定义
--------
对任意合法集合 S（|S| = K）与情景 b = 1..B：

    F_b(S) = sum_{i in S} q_i^(b) + lam * C_HU(S)

    C_HU(S) = sum_{l in {city,prov}} omega_l * sum_{g in G_l} a_g * phi_g(n_g(S))
    phi_g(n) = min{1, n / b_g},   n_g(S) = |S ∩ V_g|

    a_g >= 0 为**层级不确定性感知区域权重**（由训练期经验贝叶斯区域风险与区域
    Bootstrap 不确定性事前确定，见 region_weights.py；a_g ≡ 1 时退化为等权多尺度
    饱和覆盖目标）

风险聚合 R(S) 支持三种（论文 O 系列消融）：

    R_mean(S) = (1/B) sum_b F_b(S)
    R_min(S)  = min_b F_b(S)
    R_CVaR,tau(S) = 收益下尾 CVaR（离散精确尾部，见 cvar_lower_tail）

主目标（论文第一优先级）：  J_HUSAR(S) = R_min(S)（max-min 鲁棒口径）
次目标：                    J_HUSAR(S) = R_CVaR,tau(S)

覆盖项 lam*C_HU 与情景无关（对任意 b 取同一常数），因此对任意**平移等变性**风险泛函
（均值、min、CVaR 均满足）有

    R( {F_b} ) = R( {sum_{i in S} q_i^(b)} ) + lam * C_HU(S)

本模块据此把覆盖项与风险项分开计算（与增量缓存实现一致，并与全量重算逐步对拍）。

与既有实现的溯源
----------------
目标结构沿用本项目 v0.2/v0.3 的 `usma_heuristics/objectives.py`（同为本文作者实现），
其中 `v_g` 即本模块的 `a_g`（a_g ≡ 1 时两实现**逐位相同**，由
`tests/test_objective_crosscheck.py` 对拍验证）。本模块的差异与新增：
  1. 权重语义与构造（a_g 由区域风险 × 不确定性生成，见 region_weights.py）；
  2. 显式的 `R(S)` 聚合类型接口与 `recall_at_k` / `constraint_violations` 等评价字段；
  3. 面向 DRL 的向量化边际与批次接口（`marginal_objective` 等）。

数值约定
--------
- 所有集合 S 以 int64 索引数组表示；|S| 必须恰好等于预算 K（`constraint_violations` 检查）。
- 增量状态与全量重算的容差：1e-9（相对）。

只依赖 numpy。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------- 结构常数
ALPHA = 0.02      # 饱和比例：b_g = clip(round(alpha * n_g), 1, b_cap)
B_CAP = 10        # 饱和阈值上界
B_FLOOR = 1
OMEGA_CITY = 0.5
OMEGA_PROV = 0.5
LAM = 0.10
TAU = 0.25        # 收益下尾 CVaR 尾部水平

AGG_KINDS = ("mean", "min", "cvar")


# =====================================================================
# 实例容器
# =====================================================================
@dataclass
class HUSARInstance:
    """一个 HUSAR 配置实例的全部静态数据。

    Attributes
    ----------
    Q : (B, n) 情景风险分 q_i^(b)（float64）
    city_idx, prov_idx : (n,) 企业所属地级/省级单元序号（严格嵌套：每个城市只属一个省）
    n_firm_city, n_firm_prov : (n_city,), (n_prov,) 各单元在册企业数
    a_city, a_prov : (n_city,), (n_prov,) 区域权重（默认 1）
    lam, omega_city, omega_prov : 权衡与层级权重
    q_mean, q_lcb : (n,) 代理分（仅用于构造候选解，不参与最终评价）
    """

    Q: np.ndarray
    city_idx: np.ndarray
    prov_idx: np.ndarray
    n_firm_city: np.ndarray
    n_firm_prov: np.ndarray
    a_city: np.ndarray | None = None
    a_prov: np.ndarray | None = None
    lam: float = LAM
    omega_city: float = OMEGA_CITY
    omega_prov: float = OMEGA_PROV
    alpha: float = ALPHA
    b_cap: int = B_CAP
    q_mean: np.ndarray | None = None
    q_lcb: np.ndarray | None = None
    name: str = "instance"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.Q = np.ascontiguousarray(np.asarray(self.Q, dtype=np.float64))
        self.B, self.n = self.Q.shape
        self.city_idx = np.asarray(self.city_idx, dtype=np.int64)
        self.prov_idx = np.asarray(self.prov_idx, dtype=np.int64)
        self.n_firm_city = np.asarray(self.n_firm_city, dtype=float)
        self.n_firm_prov = np.asarray(self.n_firm_prov, dtype=float)
        self.b_city = np.clip(np.round(self.alpha * self.n_firm_city), B_FLOOR, self.b_cap)
        self.b_prov = np.clip(np.round(self.alpha * self.n_firm_prov), B_FLOOR, self.b_cap)
        self.a_city = (np.ones(len(self.n_firm_city)) if self.a_city is None
                       else np.asarray(self.a_city, dtype=float))
        self.a_prov = (np.ones(len(self.n_firm_prov)) if self.a_prov is None
                       else np.asarray(self.a_prov, dtype=float))
        if len(self.a_city) != len(self.n_firm_city) or len(self.a_prov) != len(self.n_firm_prov):
            raise ValueError("a_city / a_prov 长度必须与辖区数一致")
        if np.any(self.a_city < 0) or np.any(self.a_prov < 0):
            raise ValueError("区域权重 a_g 必须非负（保证覆盖项单调性）")
        if self.Q.shape[1] != len(self.city_idx) or len(self.prov_idx) != self.n:
            raise ValueError("索引长度与 Q 的列数不一致")
        self.q_mean = self.Q.mean(axis=0) if self.q_mean is None else np.asarray(self.q_mean, float)
        self.q_lcb = self.q_mean if self.q_lcb is None else np.asarray(self.q_lcb, float)

    # ------------------------------------------------------------ 便捷属性
    @property
    def n_city(self) -> int:
        return len(self.n_firm_city)

    @property
    def n_prov(self) -> int:
        return len(self.n_firm_prov)

    def copy_with(self, name: str | None = None, **kw) -> "HUSARInstance":
        """浅拷贝并替换字段（用于同一实例上比较不同 lam / a_g / 层级权重）。"""
        d = dict(Q=self.Q, city_idx=self.city_idx, prov_idx=self.prov_idx,
                 n_firm_city=self.n_firm_city, n_firm_prov=self.n_firm_prov,
                 a_city=self.a_city, a_prov=self.a_prov, lam=self.lam,
                 omega_city=self.omega_city, omega_prov=self.omega_prov,
                 alpha=self.alpha, b_cap=self.b_cap, q_mean=self.q_mean,
                 q_lcb=self.q_lcb, name=self.name if name is None else name,
                 meta=dict(self.meta))
        d.update(kw)
        return HUSARInstance(**d)


# =====================================================================
# 覆盖项 C_HU(S) 与边际
# =====================================================================
def coverage_components(inst: HUSARInstance, members: Sequence[int]) -> dict:
    """返回城市/省份选中计数、加权覆盖值、到达单元数。"""
    members = np.asarray(list(members), dtype=np.int64) if len(members) else np.zeros(0, np.int64)
    if members.size:
        nsel_c = np.bincount(inst.city_idx[members], minlength=inst.n_city).astype(float)
        nsel_p = np.bincount(inst.prov_idx[members], minlength=inst.n_prov).astype(float)
    else:
        nsel_c = np.zeros(inst.n_city)
        nsel_p = np.zeros(inst.n_prov)
    cov_city = float((inst.a_city * np.minimum(1.0, nsel_c / inst.b_city)).sum())
    cov_prov = float((inst.a_prov * np.minimum(1.0, nsel_p / inst.b_prov)).sum())
    return {"n_sel_city": nsel_c, "n_sel_prov": nsel_p,
            "cov_city": cov_city, "cov_prov": cov_prov,
            "cities_covered": int((nsel_c > 0).sum()),
            "provs_covered": int((nsel_p > 0).sum())}


def coverage_value(inst: HUSARInstance, n_sel_city, n_sel_prov) -> float:
    """C_HU(S)（未乘 lam）。"""
    cov = 0.0
    if inst.omega_city > 0:
        cov += inst.omega_city * float((inst.a_city * np.minimum(1.0, n_sel_city / inst.b_city)).sum())
    if inst.omega_prov > 0:
        cov += inst.omega_prov * float((inst.a_prov * np.minimum(1.0, n_sel_prov / inst.b_prov)).sum())
    return cov


def cov_gain_vec(inst: HUSARInstance, n_sel_city, n_sel_prov, cand) -> np.ndarray:
    """向量化：对候选 cand 中每个 i 返回 C_HU(S∪{i}) − C_HU(S)（未乘 lam）。"""
    cand = np.asarray(cand, dtype=np.int64)
    out = np.zeros(len(cand), dtype=float)
    if len(cand) == 0:
        return out
    if inst.omega_city > 0:
        g = inst.city_idx[cand]
        act = n_sel_city[g] < inst.b_city[g]
        out += np.where(act, inst.omega_city * inst.a_city[g] / inst.b_city[g], 0.0)
    if inst.omega_prov > 0:
        gp = inst.prov_idx[cand]
        act = n_sel_prov[gp] < inst.b_prov[gp]
        out += np.where(act, inst.omega_prov * inst.a_prov[gp] / inst.b_prov[gp], 0.0)
    return out


# =====================================================================
# 风险聚合泛函
# =====================================================================
def cvar_lower_tail(values, tau: float = TAU) -> float:
    """收益下尾 CVaR 的离散精确定义（正确处理分位点处的部分概率质量）。

    令 m = tau*B, j = floor(m), frac = m − j，values 升序为 v_(1) ≤ ... ≤ v_(B)：

        CVaR_tau = (1/m) * [ sum_{r=1..j} v_(r) + frac * v_(j+1) ]

    即"最差 tau 比例概率质量上收益的均值"。m 为整数时退化为最差 m 个情景的算术平均；
    B=150, tau=0.25 => m=37.5，取最差 37 个情景之和 + 第 38 个的 0.5 倍，除以 37.5。
    并列值不改变结果（定义基于排序后的取值）。
    """
    v = np.asarray(values, dtype=float).ravel()
    n = v.size
    if not (0.0 < float(tau) <= 1.0):
        raise ValueError("tau must be in (0, 1]")
    m = float(tau) * n
    if m >= n - 1e-12:
        return float(v.mean())
    j = int(np.floor(m + 1e-12))
    frac = m - j
    sv = np.sort(v)
    if frac <= 1e-12:
        return float(sv[:j].mean())
    return float((sv[:j].sum() + frac * sv[j]) / m)


def risk_sum_by_scenario(inst: HUSARInstance, S) -> np.ndarray:
    """（B,）向量 R_b(S) = sum_{i in S} q_i^(b)（不含覆盖项）。"""
    S = np.asarray(list(S), dtype=np.int64) if len(S) else np.zeros(0, np.int64)
    if S.size == 0:
        return np.zeros(inst.B)
    return inst.Q[:, S].sum(axis=1)


def scen_values(inst: HUSARInstance, S) -> np.ndarray:
    """（B,）向量 F_b(S) = lam * C_HU(S) + R_b(S)。"""
    cov = coverage_value_of(inst, S)
    c = inst.lam * cov
    return risk_sum_by_scenario(inst, S) + c


def coverage_value_of(inst: HUSARInstance, S) -> float:
    comp = coverage_components(inst, S)
    return inst.omega_city * comp["cov_city"] + inst.omega_prov * comp["cov_prov"]


def risk_aggregate(values: np.ndarray, agg: str = "min", tau: float = TAU) -> float:
    """对（B,）情景收益向量施加聚合泛函。"""
    v = np.asarray(values, dtype=float)
    if agg == "mean":
        return float(v.mean())
    if agg == "min":
        return float(v.min())
    if agg == "cvar":
        return cvar_lower_tail(v, tau)
    raise ValueError(f"unknown aggregation: {agg}")


def risk_mean(inst: HUSARInstance, S) -> float:
    return float(risk_sum_by_scenario(inst, S).mean())


def risk_min(inst: HUSARInstance, S) -> float:
    return float(risk_sum_by_scenario(inst, S).min())


def risk_cvar(inst: HUSARInstance, S, tau: float = TAU) -> float:
    return cvar_lower_tail(risk_sum_by_scenario(inst, S), tau)


def risk_term(inst: HUSARInstance, S, agg: str = "min", tau: float = TAU) -> float:
    """R(S)：风险聚合项（与覆盖项分离，利用平移等变性）。"""
    return risk_aggregate(risk_sum_by_scenario(inst, S), agg, tau)


# =====================================================================
# 总目标与全量评价
# =====================================================================
def husar_objective(inst: HUSARInstance, S, agg: str = "min", tau: float = TAU) -> float:
    """J_HUSAR(S) = R(S) + lam * C_HU(S)。"""
    return risk_term(inst, S, agg, tau) + inst.lam * coverage_value_of(inst, S)


# 兼容别名（论文中的 J_HUSAR）
J_HUSAR = husar_objective


def recall_at_k(inst: HUSARInstance, S, labels: np.ndarray) -> float:
    """Recall@K = |S ∩ L| / |L|（**仅用于最终评价**；labels 绝不进入训练/状态/奖励）。"""
    S = np.asarray(list(S), dtype=np.int64) if len(S) else np.zeros(0, np.int64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    return float(labels[S].sum()) / n_pos


def constraint_violations(inst: HUSARInstance, S, K: int) -> int:
    """约束违反次数（必须为 0）：重复、越界、数量不等于 K。"""
    S = list(int(i) for i in S)
    bad = 0
    if len(S) != len(set(S)):
        bad += 1
    if any((i < 0 or i >= inst.n) for i in S):
        bad += 1
    if len(S) != K:
        bad += 1
    return bad


def all_metrics(inst: HUSARInstance, S, agg: str = "min", tau: float = TAU,
                labels: np.ndarray | None = None, K: int | None = None) -> dict:
    """统一评价输出（论文表格与图表的唯一来源）。

    注意：对同一个 S，不同 agg 下的 `J_<agg>` 是**不同目标**的取值；跨方法比较必须
    固定同一 agg 列，禁止混用（论文口径纪律）。
    """
    S = np.asarray(list(S), dtype=np.int64) if len(S) else np.zeros(0, np.int64)
    rsum = risk_sum_by_scenario(inst, S)
    comp = coverage_components(inst, S)
    cov = inst.omega_city * comp["cov_city"] + inst.omega_prov * comp["cov_prov"]
    cov_term = inst.lam * cov
    J = {f"J_{a}": risk_aggregate(rsum, a, tau) + cov_term for a in AGG_KINDS}
    out = {
        "n": int(len(S)),
        # 风险项（不含覆盖）
        "risk_mean": float(rsum.mean()),
        "risk_min": float(rsum.min()),
        "risk_cvar_tau": cvar_lower_tail(rsum, tau),
        "risk_max": float(rsum.max()),
        "risk_std": float(rsum.std(ddof=0)),
        "worst_scenario": int(rsum.argmin()),
        # 覆盖项
        "coverage_utility": float(cov),                 # C_HU(S)
        "hierarchical_saturation_utility": cov_term,    # lam * C_HU(S)
        "city_coverage": comp["cov_city"],              # 加权城市覆盖值
        "province_coverage": comp["cov_prov"],
        "cities_covered": comp["cities_covered"],
        "provs_covered": comp["provs_covered"],
        # 汇总目标（三种口径分别列示）
        "J_objective": J[f"J_{agg}"],
        # 拆解：风险 + 覆盖
        "risk_part": risk_aggregate(rsum, agg, tau),
        "coverage_part": cov_term,
        "agg": agg,
        "tau": float(tau),
    }
    out.update(J)
    if labels is not None:
        out["recall_at_k"] = recall_at_k(inst, S, labels)
        out["hits"] = int(np.asarray(labels, bool)[S].sum()) if len(S) else 0
    if K is not None:
        out["constraint_violations"] = constraint_violations(inst, S, K)
    return out


# =====================================================================
# 向量化精确边际（DRL 奖励与贪心共用）
# =====================================================================
def marginal_objective(inst: HUSARInstance, st: "HUSARState", cand,
                       agg: str = "min", tau: float = TAU) -> np.ndarray:
    """精确边际 J(S∪{i}) − J(S)（i 遍历 cand）。

    agg='min' ： min_b(sums_b + q_i^(b)) − min_b sums_b + lam*ΔC(i)
    agg='mean'： mean_b q_i^(b) + lam*ΔC(i)
    agg='cvar'： CVaR_tau(sums + q_i) − CVaR_tau(sums) + lam*ΔC(i)（精确离散尾部）
    """
    cand = np.asarray(cand, dtype=np.int64)
    if len(cand) == 0:
        return np.zeros(0)
    cov_inc = inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)
    sums = st.scen_sums
    if agg == "mean":
        return inst.Q[:, cand].mean(axis=0) + cov_inc
    if agg == "min":
        cur = float(sums.min())
        new = (sums[:, None] + inst.Q[:, cand]).min(axis=0)
        return (new - cur) + cov_inc
    if agg == "cvar":
        cur = cvar_lower_tail(sums, tau)
        m = tau * inst.B
        sums2 = sums[:, None] + inst.Q[:, cand]
        if m >= inst.B - 1e-12:
            new = sums2.mean(axis=0)
        else:
            j = int(np.floor(m + 1e-12))
            frac = m - j
            part = np.partition(sums2, j, axis=0)
            if frac <= 1e-12:
                new = part[:j].mean(axis=0)
            else:
                new = (part[:j].sum(axis=0) + frac * part[j]) / m
        return new + cov_inc - cur
    raise ValueError(f"unknown aggregation: {agg}")


def marginal_proxy(inst: HUSARInstance, st: "HUSARState", cand, proxy: str = "lcb") -> np.ndarray:
    """代理边际（仅用于构造候选解）：proxy ∈ {'lcb','mean'}; 单调次模。"""
    cand = np.asarray(cand, dtype=np.int64)
    score = inst.q_lcb if proxy == "lcb" else inst.q_mean
    return score[cand] + inst.lam * cov_gain_vec(inst, st.n_sel_city, st.n_sel_prov, cand)


# =====================================================================
# 增量状态（与全量重算逐步对拍）
# =====================================================================
class HUSARState:
    """可增删的候选解状态，维护：
        n_sel_city / n_sel_prov （辖区选中计数）
        scen_sums (B,)          （已选企业情景风险分之和，**不含覆盖项**）
        cov_city / cov_prov     （加权覆盖累计值，未乘 omega/lam）
    add/remove 后可用 validate() 与全量重算对拍（实验中每轮调用）。
    """

    __slots__ = ("inst", "inS", "n_sel_city", "n_sel_prov", "scen_sums",
                 "cov_city", "cov_prov", "members")

    def __init__(self, inst: HUSARInstance, members: Iterable[int] = ()):
        self.inst = inst
        self.inS = np.zeros(inst.n, dtype=bool)
        self.n_sel_city = np.zeros(inst.n_city, dtype=float)
        self.n_sel_prov = np.zeros(inst.n_prov, dtype=float)
        self.scen_sums = np.zeros(inst.B, dtype=float)
        self.cov_city = 0.0
        self.cov_prov = 0.0
        self.members: list[int] = []
        for i in members:
            self.add(int(i))

    # ---------------------------------------------------------------- 操作
    def add(self, i: int) -> None:
        if self.inS[i]:
            raise ValueError(f"firm {i} already selected")
        inst = self.inst
        g = inst.city_idx[i]
        if self.n_sel_city[g] < inst.b_city[g]:
            self.cov_city += inst.a_city[g] / inst.b_city[g]
        self.n_sel_city[g] += 1.0
        gp = inst.prov_idx[i]
        if self.n_sel_prov[gp] < inst.b_prov[gp]:
            self.cov_prov += inst.a_prov[gp] / inst.b_prov[gp]
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
            self.cov_city -= inst.a_city[g] / inst.b_city[g]
        gp = inst.prov_idx[i]
        self.n_sel_prov[gp] -= 1.0
        if self.n_sel_prov[gp] < inst.b_prov[gp]:
            self.cov_prov -= inst.a_prov[gp] / inst.b_prov[gp]
        self.scen_sums -= inst.Q[:, i]
        self.inS[i] = False
        self.members.remove(i)

    # ---------------------------------------------------------------- 查询
    @property
    def cov_term(self) -> float:
        inst = self.inst
        return inst.lam * (inst.omega_city * self.cov_city + inst.omega_prov * self.cov_prov)

    def F(self) -> np.ndarray:
        """（B,）F_b(S) = scen_sums + 覆盖项。"""
        return self.scen_sums + self.cov_term

    def objective(self, agg: str = "min", tau: float = TAU) -> float:
        return risk_aggregate(self.scen_sums, agg, tau) + self.cov_term

    def objective_full(self, agg: str = "min", tau: float = TAU) -> float:
        """直接对含覆盖项的 F_b(S) 施加聚合（用于验证平移等变性下的等价性）。"""
        return risk_aggregate(self.F(), agg, tau)

    def members_arr(self) -> np.ndarray:
        return np.asarray(self.members, dtype=np.int64)

    def copy(self) -> "HUSARState":
        s = HUSARState.__new__(HUSARState)
        s.inst = self.inst
        s.inS = self.inS.copy()
        s.n_sel_city = self.n_sel_city.copy()
        s.n_sel_prov = self.n_sel_prov.copy()
        s.scen_sums = self.scen_sums.copy()
        s.cov_city = self.cov_city
        s.cov_prov = self.cov_prov
        s.members = list(self.members)
        return s

    # ---------------------------------------------------------------- 校验
    def validate(self) -> None:
        """与全量重算对拍（容差 1e-9 相对）。任何不一致直接抛错。"""
        inst = self.inst
        ref = coverage_components(inst, self.members)
        assert abs(ref["cov_city"] - self.cov_city) <= 1e-9 * max(1.0, abs(ref["cov_city"]))
        assert abs(ref["cov_prov"] - self.cov_prov) <= 1e-9 * max(1.0, abs(ref["cov_prov"]))
        assert np.array_equal(ref["n_sel_city"], self.n_sel_city)
        assert np.array_equal(ref["n_sel_prov"], self.n_sel_prov)
        ref_sums = risk_sum_by_scenario(inst, self.members)
        assert np.max(np.abs(ref_sums - self.scen_sums)) <= 1e-8 * max(1.0, np.abs(ref_sums).max())
        assert self.inS.sum() == len(self.members)


# =====================================================================
# 完备性检查：目标评价器的自洽性质（供训练与测试调用）
# =====================================================================
def additive_decomposition_check(inst: HUSARInstance, S, agg: str = "min",
                                 tau: float = TAU, tol: float = 1e-9) -> bool:
    """验证 R({F_b(S)}) == R({R_b(S)}) + lam*C_HU(S)（平移等变性）。"""
    a = risk_aggregate(scen_values(inst, S), agg, tau)
    b = risk_aggregate(risk_sum_by_scenario(inst, S), agg, tau) + inst.lam * coverage_value_of(inst, S)
    return abs(a - b) <= tol * max(1.0, abs(a))


def trajectory_reward_check(inst: HUSARInstance, seq: Sequence[int], agg: str = "min",
                            tau: float = TAU, tol: float = 1e-9) -> tuple[float, float]:
    """γ=1 时轨迹增量奖励之和应严格等于 J(S_K) − J(∅)。返回 (sum_r, J(S_K) − J(∅))。"""
    st = HUSARState(inst, [])
    prev = st.objective(agg, tau)
    total = 0.0
    for i in seq:
        st.add(int(i))
        cur = st.objective(agg, tau)
        total += cur - prev
        prev = cur
    return total, st.objective(agg, tau) - HUSARState(inst, []).objective(agg, tau)
