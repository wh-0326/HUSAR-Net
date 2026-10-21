# -*- coding: utf-8 -*-
"""
合成层级空间实例生成器（HUSAR-Net 专用，情景族 A / B / C）。

声明
----
本模块生成的全部数据均为**人工构造**，不是真实监管观测，也不用于任何监管效果推断；
只服务于算法求解质量、泛化与计算行为的对照实验。

结构：企业 → 地级单元（城市） → 省级单元（省）；城市规模服从 Zipf(a) 分布。

情景模型（与 v0.2/v0.3 生成器同族，见 01_audit_and_novelty.md）
    基础风险      p_i = clip(base_rate * exp(nu_i) * exp(h_c) * exp(h_p), p_lo, p_hi)
    情景乘子      z_{b,i} = sqrt(rho) * c_{b,c(i)} + sqrt(1-rho) * e_{b,i}
    情景风险      Q[b,i] = clip(p_i * exp(conflict * sigma * z_{b,i}), 0, 1)
族：A 独立扰动（rho=0）；B 城市相关扰动（rho=0.7）；C 高情景冲突（轮转群体冲击）。

与 v0.3 生成器的差异（本次新增）：
  1. 返回 `HUSARInstance`，并附加**由独立训练面板构造的区域权重** a_g（region_weights.py）；
  2. 提供 `split_seeds()`：训练/验证/测试种子互斥，保证 Protocol I 的实例级泛化评估。

注意：合成实例上的 `a_g` 与真实实例同构（先验风险 × bootstrap 不确定性），但数值不可与
真实数据混用或直接比较。
"""
from __future__ import annotations

import numpy as np

from objective import HUSARInstance
from region_weights import synthetic_region_weights

FAMILIES = ("A", "B", "C")


# ------------------------------------------------------------------ 层次结构
def make_hierarchy(n_firms: int, n_cities: int, n_provs: int, seed: int,
                   zipf_a: float = 1.1) -> dict:
    """按 Zipf(n_cities, a) 生成城市规模，并把城市轮转分配到省份。"""
    if not 1 <= n_cities <= n_firms or n_provs < 1:
        raise ValueError("require 1 <= n_cities <= n_firms and n_provs >= 1")
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_cities + 1)
    w = 1.0 / np.power(ranks, zipf_a)
    w = w / w.sum()
    raw = w * n_firms
    size = np.floor(raw).astype(int)
    size = np.maximum(size, 1)
    diff = int(n_firms - size.sum())
    if diff > 0:
        order = np.argsort(-(raw - size))
        size[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(size)[::-1]
        k = 0
        while diff < 0:
            j = order[k % n_cities]
            if size[j] > 1:
                size[j] -= 1
                diff += 1
            k += 1
    city_idx = np.repeat(np.arange(n_cities), size)
    rng.shuffle(city_idx)
    prov_of_city = np.arange(n_cities) % n_provs
    prov_idx = prov_of_city[city_idx]
    n_city = np.bincount(city_idx, minlength=n_cities).astype(float)
    n_prov = np.bincount(prov_idx, minlength=n_provs).astype(float)
    return {"city_idx": city_idx, "prov_idx": prov_idx,
            "n_city": n_city, "n_prov": n_prov, "prov_of_city": prov_of_city}


# ------------------------------------------------------------------ 情景矩阵
def make_scenarios(hier: dict, n_firms: int, n_scen: int, seed: int, family: str = "A",
                   conflict: float = 0.5, rho: float | None = None,
                   base_rate: float = 0.045, s_ind: float = 0.6,
                   sigma_mult: float = 0.55, n_groups: int = 8,
                   p_lo: float = 0.002, p_hi: float = 0.60) -> np.ndarray:
    """生成 (B, n) 情景风险矩阵，取值落在 [0,1]。"""
    if family not in FAMILIES:
        raise ValueError(family)
    if rho is None:
        rho = {"A": 0.0, "B": 0.7, "C": 0.5}[family]
    rng = np.random.default_rng(seed + 10007)
    city_idx = hier["city_idx"]
    n_cities = len(hier["n_city"])
    n_provs = len(hier["n_prov"])

    nu = rng.normal(0.0, s_ind, n_firms)
    h_city = rng.normal(0.0, 0.55, n_cities)
    h_prov = rng.normal(0.0, 0.30, n_provs)
    p = base_rate * np.exp(nu + h_city[city_idx] + h_prov[hier["prov_idx"]])
    p = np.clip(p, p_lo, p_hi)

    c_city = rng.normal(0.0, 1.0, size=(n_scen, n_cities))
    e_ind = rng.normal(0.0, 1.0, size=(n_scen, n_firms))
    z = np.sqrt(rho) * c_city[:, city_idx] + np.sqrt(1.0 - rho) * e_ind
    Q = p[None, :] * np.exp(conflict * sigma_mult * z)

    if family == "C":
        group_of_city = np.arange(n_cities) % n_groups
        g_i = group_of_city[city_idx]
        amp = 0.30 * conflict
        for b in range(n_scen):
            Q[b, g_i == (b % n_groups)] *= (1.0 + amp * 4.0)
        Q = Q + amp * 0.5 * conflict

    return np.clip(Q, 0.0, 1.0)


# ------------------------------------------------------------------ 实例
def make_instance(family: str = "A", n: int = 1000, n_scen: int = 150,
                  n_cities: int | None = None, n_provs: int = 31, seed: int = 1,
                  conflict: float = 0.5, rho: float | None = None, zipf_a: float = 1.1,
                  with_weights: bool = True, weight_params: dict | None = None,
                  lam: float = 0.10, omega_city: float = 0.5,
                  omega_prov: float = 0.5) -> HUSARInstance:
    """构造一个带区域权重的合成 HUSARInstance。"""
    if n_cities is None:
        n_cities = int(np.clip(round(n / 19), 20, 400))
    hier = make_hierarchy(n, n_cities, n_provs, seed=seed + 101, zipf_a=zipf_a)
    Q = make_scenarios(hier, n, n_scen, seed=seed, family=family, conflict=conflict, rho=rho)
    q_lcb = np.percentile(Q, 10, axis=0)
    name = f"syn_{family}_n{n}_B{n_scen}_c{n_cities}_s{seed}"
    meta = {"source": "synthetic", "family": family, "seed": seed,
            "conflict": conflict, "n_scen": n_scen, "n_cities": n_cities,
            "n_provs": n_provs, "zipf_a": zipf_a,
            "generator": "husarnet.synthetic.make_instance"}
    a_city = a_prov = None
    if with_weights:
        w = synthetic_region_weights(HUSARInstance(
            Q=Q, city_idx=hier["city_idx"], prov_idx=hier["prov_idx"],
            n_firm_city=hier["n_city"], n_firm_prov=hier["n_prov"],
            lam=lam, omega_city=omega_city, omega_prov=omega_prov),
            seed=seed, family=family, conflict=conflict, rho=rho,
            params=weight_params)
        a_city, a_prov = w["a_city"], w["a_prov"]
        meta["weight_params"] = w["params"]
    inst = HUSARInstance(
        Q=Q, city_idx=hier["city_idx"], prov_idx=hier["prov_idx"],
        n_firm_city=hier["n_city"], n_firm_prov=hier["n_prov"],
        a_city=a_city, a_prov=a_prov, q_mean=Q.mean(axis=0), q_lcb=q_lcb,
        lam=lam, omega_city=omega_city, omega_prov=omega_prov,
        name=name, meta=meta)
    return inst


def make_axis_instance(n: int, n_scen: int = 150, n_cities: int = 260, family: str = "B",
                       seed: int = 1, with_weights: bool = True) -> HUSARInstance:
    """规模轴实例（E6 用）：城市数按 n 比例，情景数可 30/150。"""
    return make_instance(family=family, n=n, n_scen=n_scen, n_cities=n_cities, seed=seed,
                         with_weights=with_weights, conflict=0.7)


# ------------------------------------------------------------------ 种子纪律
def split_seeds(n_train: int, n_val: int, n_test: int, base: int = 1000) -> dict:
    """训练/验证/测试实例种子互斥（Protocol I：测试实例在训练与调参中从未出现）。"""
    s = np.arange(base, base + n_train + n_val + n_test)
    return {"train": s[:n_train].tolist(),
            "val": s[n_train:n_train + n_val].tolist(),
            "test": s[n_train + n_val:].tolist()}


SCALE_GRID = [
    # (标签, N, B, 城市数, 省份数)
    ("n1000_B150", 1000, 150, 60, 10),
    ("n5000_B150", 5000, 150, 260, 31),
    ("n10000_B150", 10000, 150, 350, 31),
    ("n50000_B150", 50000, 150, 400, 31),
    ("n1000_B30", 1000, 30, 60, 10),
    ("n5000_B30", 5000, 30, 260, 31),
]


# ------------------------------------------------------------------ 实例描述量
def conflict_descriptor(inst: HUSARInstance, K: int, seed: int = 0) -> dict:
    """情景冲突强度：各情景 top-K 集合两两 Jaccard 均值（越低越冲突）。"""
    rng = np.random.default_rng(seed)
    Q = inst.Q
    B = inst.B
    sets = [set(np.argpartition(-Q[b], K)[:K].tolist()) for b in range(B)]
    jac = []
    for _ in range(min(200, B * (B - 1) // 2)):
        a, b = rng.integers(0, B, 2)
        if a == b:
            continue
        inter = len(sets[a] & sets[b])
        jac.append(inter / (2 * K - inter))
    return {"conflict_jaccard": float(np.mean(jac)) if jac else None}
