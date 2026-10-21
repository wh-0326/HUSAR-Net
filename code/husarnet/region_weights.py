# -*- coding: utf-8 -*-
"""
HUSAR 层级区域权重 a_g 的构造（事前、非目标期标签）。

论文口径
--------
    a_g = 1 + kappa * (clip( rho_g^gamma * (1 + beta * u_g), a_lo, a_cap ) - 1)

其中（全部只用训练期信息）：
  * rho_g : 区域先验风险 = 训练期（<=2024）经验贝叶斯（EB）平滑的辖区处罚披露率，
            作中位数归一化（中位数辖区 => 1）；
  * u_g   : 区域估计不确定性 = 风险模型 Bootstrap 重拟合下"辖区平均风险分"的标准差，
            作中位数归一化；
  * gamma : 风险弹性；beta : 不确定性敏感度；kappa : 总体强度
            （kappa=0 时 a_g ≡ 1，**严格退化为原等权多尺度目标**）；
  * a_lo, a_cap : 截断界限（非负有界，防止极端辖区主导覆盖项）。

纪律
----
1. rho_g 仅由 <=2024 的 firm-year 标签与 <=2024 特征计算；u_g 仅由风险模型的 Bootstrap
   重拟合分布（其本身只用 <=2024 数据训练）计算。**目标年（2025）真实处罚标签不参与
   a_g 的任何环节**；由 tests/test_leakage.py 静态断言 + 运行时打乱标签不变性检查。
2. 权重只依赖辖区（不含企业身份），因此不改变覆盖项的非负性与单调次模结构。
3. 超参数在训练/验证期（合成开发实例 + 2024 一次性验证）确定，不在 2025 上回选。
"""
from __future__ import annotations

import numpy as np

# 默认超参数（在合成开发实例 + 2024 验证上确定；2025 不参与）
GAMMA_RISK = 1.0
BETA_UNC = 0.5
KAPPA = 1.0
A_LO = 0.25
A_CAP = 4.0

DEFAULT_PARAMS = dict(gamma=GAMMA_RISK, beta=BETA_UNC, kappa=KAPPA, a_lo=A_LO, a_cap=A_CAP)


# =====================================================================
# 核心映射
# =====================================================================
def build_weights(prior: np.ndarray, unc: np.ndarray, gamma: float = GAMMA_RISK,
                  beta: float = BETA_UNC, kappa: float = KAPPA,
                  a_lo: float = A_LO, a_cap: float = A_CAP,
                  center: str = "median") -> np.ndarray:
    """由辖区先验风险 prior 与不确定性 unc 生成区域权重 a_g（非负有界，均值归一）。"""
    prior = np.asarray(prior, dtype=float)
    unc = np.asarray(unc, dtype=float)
    if prior.shape != unc.shape:
        raise ValueError("prior / unc 形状不一致")
    if np.any(prior < 0) or np.any(unc < 0):
        raise ValueError("prior / unc 必须非负")
    if kappa == 0.0:
        return np.ones_like(prior)

    cen = np.median if center == "median" else np.mean
    p_n = prior / max(float(cen(prior)), 1e-12)
    u_n = unc / max(float(cen(unc)), 1e-12)

    raw = np.power(p_n, float(gamma)) * (1.0 + float(beta) * u_n)
    raw = np.clip(raw, a_lo, a_cap)
    raw = raw / max(raw.mean(), 1e-12)
    a = 1.0 + float(kappa) * (raw - 1.0)
    return np.maximum(a, 0.0)


def empirical_bayes_rates(Y: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Poisson–Gamma 经验贝叶斯后验均值（Clayton–Kaldor / Marshall 收缩）。

    Y_g / N_g 为辖区事件数 / 观测数；r̄ = ΣY/ΣN；组间真实率方差 s² 用矩估计
    （观测组间方差减去二项抽样方差）；后验均值 rho_g = (r̄/s² + Y_g)/(1/s² + N_g)。
    N_g = 0 的辖区退回 r̄。
    """
    Y = np.asarray(Y, dtype=float)
    N = np.asarray(N, dtype=float)
    totY, totN = Y.sum(), N.sum()
    if totN <= 0:
        raise ValueError("训练期观测总数为 0")
    rbar = totY / totN
    r_g = np.divide(Y, N, out=np.full_like(N, np.nan), where=N > 0)
    ok = N > 0
    if ok.sum() >= 2:
        w_ok = N[ok] / N[ok].sum()
        r_obs = r_g[ok]
        var_obs = float((w_ok * (r_obs - rbar) ** 2).sum() / max(1e-12, 1.0 - (w_ok ** 2).sum()))
        var_bin = float((w_ok * r_obs * (1.0 - r_obs) / N[ok]).sum())
        s2 = max(var_obs - var_bin, 1e-8)
    else:
        s2 = 1e-8
    rho = (rbar / s2 + Y) / (1.0 / s2 + N)
    rho[N <= 0] = rbar
    return rho


def group_bootstrap_uncertainty(Q: np.ndarray, group_idx: np.ndarray, n_group: int) -> np.ndarray:
    """辖区平均风险的 Bootstrap 标准差：对每个情景 b 计算辖区均值，再在情景维取标准差。"""
    Q = np.asarray(Q, dtype=float)
    group_idx = np.asarray(group_idx, dtype=np.int64)
    cnt = np.bincount(group_idx, minlength=n_group).astype(float)
    cnt_safe = np.where(cnt > 0, cnt, 1.0)
    means = np.empty((Q.shape[0], n_group))
    for b in range(Q.shape[0]):
        means[b] = np.bincount(group_idx, weights=Q[b], minlength=n_group) / cnt_safe
    u = means.std(axis=0, ddof=0)
    u[cnt == 0] = np.nan
    return u


def _group_mean(values: np.ndarray, group_idx: np.ndarray, n_group: int) -> np.ndarray:
    cnt = np.bincount(group_idx, minlength=n_group).astype(float)
    s = np.bincount(group_idx, weights=np.asarray(values, float), minlength=n_group)
    out = s / np.where(cnt > 0, cnt, 1.0)
    out[cnt == 0] = np.nan
    return out


# This minimal release omits the project-specific real-data file loader.

# =====================================================================
# 合成实例：训练面板 + 目标面板两套情景矩阵
# =====================================================================
def synthetic_region_weights(inst, seed: int = 0, n_train_scen: int = 60,
                             params: dict | None = None, conflict: float = 0.5,
                             family: str = "A", rho: float | None = None) -> dict:
    """为合成 HUSARInstance 生成区域权重。

    模拟真实链路：用**独立训练面板**情景矩阵（同一生成器、不同随机种子）估计
    (i) 辖区先验风险（辖区均值风险分）与 (ii) 辖区不确定性（情景维标准差）；
    目标实例的 Q **不参与**权重构造，避免"用目标年信息构造权重"的同义问题。
    """
    from synthetic import make_scenarios  # 本地模块（与目标实例同一生成器）

    n_city = inst.n_city
    n_prov = inst.n_prov
    hier = {"city_idx": inst.city_idx, "prov_idx": inst.prov_idx,
            "n_city": np.bincount(inst.city_idx, minlength=n_city).astype(float),
            "n_prov": np.bincount(inst.prov_idx, minlength=n_prov).astype(float)}
    Qtr = make_scenarios(hier, inst.n, n_train_scen, seed=seed + 7777, family=family,
                         conflict=conflict, rho=rho)
    firm_prior = Qtr.mean(axis=0)
    prior_city = _group_mean(firm_prior, inst.city_idx, n_city)
    prior_prov = _group_mean(firm_prior, inst.prov_idx, n_prov)
    unc_city = group_bootstrap_uncertainty(Qtr, inst.city_idx, n_city)
    unc_prov = group_bootstrap_uncertainty(Qtr, inst.prov_idx, n_prov)
    for arr in (prior_city, prior_prov, unc_city, unc_prov):
        arr[np.isnan(arr)] = np.nanmedian(arr)
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    a_city = build_weights(prior_city, unc_city, **p)
    a_prov = build_weights(prior_prov, unc_prov, **p)
    return {"a_city": a_city, "a_prov": a_prov,
            "diag": {"rho_city": prior_city, "rho_prov": prior_prov,
                     "unc_city": unc_city, "unc_prov": unc_prov},
            "params": p, "unc_source": "synthetic_train_panel"}
