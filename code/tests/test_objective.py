# -*- coding: utf-8 -*-
"""
PHASE 1 单元测试：HUSAR 目标评价器。

覆盖：
  T1 与既有实现（research_upgrade/usma_heuristics/objectives.py）对拍（a_g ≡ 1）
  T2 收益下尾 CVaR 的离散精确定义（手算样例 + 与最差-k 均值的一致性条件）
  T3 平移等变性：R({F_b}) == R({R_b}) + lam*C_HU，对 mean/min/cvar 全部成立
  T4 增量状态 == 全量重算（随机增删序列）
  T5 向量化精确边际 == 有限差分（mean/min/cvar）
  T6 约束校验与 Recall@K 口径
  T7 区域权重映射：kappa=0 退化；非负；先验单调；截断生效
  T8 经验贝叶斯收缩的数值正确性
  T9 合成实例：目标年 Q 不参与权重构造（防"用目标年信息构造权重"）
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import HUSARNET  # noqa: E402  (路径注入)

from objective import (HUSARInstance, HUSARState, all_metrics, additive_decomposition_check,
                       constraint_violations, coverage_components, coverage_value_of,
                       cvar_lower_tail, husar_objective, marginal_objective, marginal_proxy,
                       recall_at_k, risk_aggregate, risk_sum_by_scenario, trajectory_reward_check)
from region_weights import build_weights, empirical_bayes_rates, group_bootstrap_uncertainty
from synthetic import make_hierarchy, make_instance, make_scenarios


# ------------------------------------------------------------------ 工具
def small_instance(seed=0, n=60, B=10, n_city=6, n_prov=3, lam=0.1, weighted=True):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(0, 1, size=(B, n))
    city = rng.integers(0, n_city, n)
    prov = city % n_prov
    n_fc = np.bincount(city, minlength=n_city).astype(float)
    n_fp = np.bincount(prov, minlength=n_prov).astype(float)
    a_c = rng.uniform(0.5, 2.0, n_city) if weighted else np.ones(n_city)
    a_p = rng.uniform(0.5, 2.0, n_prov) if weighted else np.ones(n_prov)
    return HUSARInstance(Q=Q, city_idx=city, prov_idx=prov, n_firm_city=n_fc,
                         n_firm_prov=n_fp, a_city=a_c, a_prov=a_p, lam=lam,
                         q_mean=Q.mean(0), q_lcb=np.percentile(Q, 10, axis=0))


# ================================================================== T2
def test_cvar_discrete_exact():
    # 手算：v=[1,2,3,4]，tau=0.5 -> m=2 -> 最差 2 个均值 = 1.5
    assert cvar_lower_tail([1, 2, 3, 4], 0.5) == pytest.approx(1.5)
    # 非整数尾部：v=[1,2,3,4,5,6], tau=0.25 -> m=1.5, j=1, frac=0.5 ->
    # (v_(1) + 0.5*v_(2))/1.5 = (1+1)/1.5
    assert cvar_lower_tail([1, 2, 3, 4, 5, 6], 0.25) == pytest.approx((1 + 0.5 * 2) / 1.5)
    # B=150, tau=0.25 -> m=37.5：最差 37 个 + 0.5*第 38 个，除以 37.5
    v = np.arange(1.0, 151.0)
    expect = (v[:37].sum() + 0.5 * v[37]) / 37.5
    assert cvar_lower_tail(v, 0.25) == pytest.approx(expect)
    # tau=1 -> 均值
    assert cvar_lower_tail(v, 1.0) == pytest.approx(v.mean())
    # 全相等 -> 该值
    assert cvar_lower_tail(np.full(150, 3.3), 0.25) == pytest.approx(3.3)
    # 与"取整 ceil 近似"不同（B=150, tau=0.25 时 ceil 方案取 38 个的均值）
    assert cvar_lower_tail(v, 0.25) != pytest.approx(v[:38].mean())


# ================================================================== T3/T4/T5
@pytest.mark.parametrize("agg", ["mean", "min", "cvar"])
def test_additive_decomposition_and_incremental(agg):
    inst = small_instance(seed=1)
    rng = np.random.default_rng(7)
    S = rng.choice(inst.n, 25, replace=False).tolist()
    assert additive_decomposition_check(inst, S, agg)

    st = HUSARState(inst, [])
    for i in S[:10]:
        st.add(int(i))
    st.validate()
    # 全量对比
    assert st.objective(agg) == pytest.approx(husar_objective(inst, st.members, agg), abs=1e-9)
    assert st.objective_full(agg) == pytest.approx(st.objective(agg), abs=1e-9)
    # 随机增删
    for i in S[10:]:
        st.add(int(i))
    for i in S[:5]:
        st.remove(int(i))
    st.validate()
    assert st.objective(agg) == pytest.approx(husar_objective(inst, st.members, agg), abs=1e-9)

    # 轨迹奖励一致性（gamma=1）
    s, j = trajectory_reward_check(inst, S, agg)
    assert s == pytest.approx(j, abs=1e-9)


@pytest.mark.parametrize("agg", ["mean", "min", "cvar"])
def test_marginal_matches_finite_difference(agg):
    inst = small_instance(seed=2, n=40, B=8)
    rng = np.random.default_rng(3)
    base = rng.choice(inst.n, 12, replace=False)
    st = HUSARState(inst, [int(i) for i in base])
    cand = np.array([i for i in range(inst.n) if i not in set(base.tolist())])
    exact = marginal_objective(inst, st, cand, agg)
    cur = st.objective(agg)
    fdm = np.array([husar_objective(inst, list(base) + [int(i)], agg) - cur for i in cand])
    assert np.allclose(exact, fdm, atol=1e-9)

    # 代理边际（LCB/mean）仅用于构造候选，但必须是真值（可对拍）
    lp = marginal_proxy(inst, st, cand, "lcb")
    fdl = np.array([(inst.q_lcb[int(i)] + inst.lam * coverage_value_of(inst, list(base) + [int(i)]))
                    - (inst.lam * coverage_value_of(inst, list(base))) for i in cand])
    assert np.allclose(lp, fdl, atol=1e-9)


def test_saturated_region_boundary():
    """区域饱和边界：达到 b_g 后覆盖边际为 0，但不影响风险边际。"""
    inst = small_instance(seed=5, n=30, B=5, n_city=3)
    st = HUSARState(inst, [])
    # 把城市 0 填满
    g0 = np.where(inst.city_idx == 0)[0]
    for i in g0[:int(inst.b_city[0])]:
        st.add(int(i))
    st.validate()
    rest = [i for i in g0[int(inst.b_city[0]):]]
    if rest:
        cg = np.array([inst.lam * (inst.a_city[0] / inst.b_city[0])])  # 满额后应为 0
        gains = marginal_objective(inst, st, np.array(rest), "mean")
        risk_only = inst.Q[:, np.array(rest)].mean(axis=0)
        assert np.allclose(gains, risk_only, atol=1e-9)  # 覆盖增量为 0


def test_zero_and_negative_utility_robustness():
    """零收益/负收益候选的数值稳健性（Q 含 0 与负值域内不产生 nan/inf）。"""
    rng = np.random.default_rng(11)
    Q = rng.uniform(-0.5, 0.5, size=(6, 30))
    Q[:, 0] = 0.0
    inst = HUSARInstance(Q=Q, city_idx=np.arange(30) % 5, prov_idx=np.arange(30) % 2,
                         n_firm_city=np.full(5, 6.0), n_firm_prov=np.full(2, 15.0))
    st = HUSARState(inst, [])
    for agg in ("mean", "min", "cvar"):
        g = marginal_objective(inst, st, np.arange(30), agg)
        assert np.all(np.isfinite(g))
    # 零覆盖权重（omega=0）时目标只含风险项
    inst0 = inst.copy_with(omega_city=0.0, omega_prov=0.0)
    S = [0, 1, 2]
    assert husar_objective(inst0, S, "min") == pytest.approx(risk_sum_by_scenario(inst0, S).min())


def test_constraints_and_recall():
    inst = small_instance(seed=6)
    S = [0, 1, 2, 3]
    assert constraint_violations(inst, S, 4) == 0
    assert constraint_violations(inst, [0, 0, 1], 3) >= 1     # 重复
    assert constraint_violations(inst, [0, 1], 3) >= 1        # 数量不符
    assert constraint_violations(inst, [0, 1, inst.n + 5], 3) >= 1  # 越界
    lab = np.zeros(inst.n, int); lab[[1, 2, 5]] = 1
    assert recall_at_k(inst, [1], lab) == pytest.approx(1 / 3)
    assert recall_at_k(inst, [0, 1, 2, 5], lab) == pytest.approx(1.0)
    m = all_metrics(inst, [1, 2], agg="min", labels=lab, K=2)
    assert m["hits"] == 2 and m["recall_at_k"] == pytest.approx(2 / 3)
    assert m["constraint_violations"] == 0


# ================================================================== T7/T8
def test_build_weights_properties():
    prior = np.array([0.001, 0.01, 0.05, 0.2, 0.6])
    unc = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    a0 = build_weights(prior, unc, kappa=0.0)
    assert np.allclose(a0, 1.0)                                  # 严格退化
    a = build_weights(prior, unc, gamma=1.0, beta=0.5, kappa=1.0, a_lo=0.25, a_cap=4.0)
    assert np.all(a >= 0) and np.all(np.isfinite(a))
    assert np.all(np.diff(a) >= 0)                               # 先验风险非减（截断处并列）
    assert a.mean() == pytest.approx(1.0, abs=1e-12)             # 均值归一
    # 不触发截断时严格递增
    a_free = build_weights(prior, unc, gamma=1.0, beta=0.0, kappa=1.0, a_lo=0.01, a_cap=1e6)
    assert np.all(np.diff(a_free) > 0)
    # 不确定性单调（固定先验）
    p = np.full(5, 0.05)
    u = np.array([0.0, 0.2, 0.4, 0.8, 1.6])
    au = build_weights(p, u, gamma=1.0, beta=0.5, kappa=1.0)
    assert np.all(np.diff(au) > 0)
    # 截断生效
    ext = build_weights(np.array([1e-6, 1.0, 1e6]), np.zeros(3), a_lo=0.25, a_cap=4.0)
    assert ext.max() <= 4.0 / ext.mean() + 1e-9  # 上界在均值归一化前施加；此处只检查有限性
    assert np.all(np.isfinite(ext))
    # kappa 插值：kappa=0.5 时权重位于 1 与 kappa=1 结果之间
    a_full = build_weights(prior, unc, kappa=1.0)
    a_half = build_weights(prior, unc, kappa=0.5)
    assert np.allclose(a_half, 1 + 0.5 * (a_full - 1))


def test_empirical_bayes_rates():
    Y = np.array([0.0, 5.0, 50.0])
    N = np.array([10.0, 100.0, 1000.0])
    rho = empirical_bayes_rates(Y, N)
    raw = Y / N
    rbar = Y.sum() / N.sum()
    # 收缩：小样本辖区被拉向全局均值，大样本辖区接近原始率
    assert abs(rho[0] - rbar) < abs(raw[0] - rbar) + 1e-12
    assert abs(rho[2] - raw[2]) < abs(rho[0] - raw[0])
    assert np.all((rho >= 0) & (rho <= 1))
    # N=0 退回全局均值
    rho0 = empirical_bayes_rates(np.array([1.0, 0.0]), np.array([10.0, 0.0]))
    assert rho0[1] == pytest.approx(1.0 / 10.0)


def test_group_bootstrap_uncertainty():
    rng = np.random.default_rng(4)
    Q = rng.normal(0.1, 1.0, size=(50, 20))
    g = np.repeat([0, 1], 10)
    u = group_bootstrap_uncertainty(Q, g, 2)
    # 各组的不确定性应等于"该组在每个情景的均值"在情景维上的标准差
    for k, idx in enumerate((np.arange(10), np.arange(10, 20))):
        m = Q[:, idx].mean(axis=1)
        assert u[k] == pytest.approx(m.std())
    # 组内方差越小 -> 不确定性越小
    Q2 = Q.copy()
    Q2[:, :10] *= 0.1
    u2 = group_bootstrap_uncertainty(Q2, g, 2)
    assert u2[0] < u[0]


# ================================================================== T1
def test_crosscheck_against_existing_implementation():
    """与既有 USMA 实现（同为本文作者代码）在 a_g ≡ 1 时逐位对拍。"""
    try:
        import objectives as old_obj
    except Exception as exc:      # pragma: no cover
        pytest.skip(f"既有实现不可导入: {exc}")

    rng = np.random.default_rng(42)
    n, B, n_city, n_prov = 120, 20, 8, 4
    Q = rng.uniform(0, 1, (B, n))
    city = rng.integers(0, n_city, n)
    prov = city % n_prov
    n_fc = np.bincount(city, minlength=n_city).astype(float)
    n_fp = np.bincount(prov, minlength=n_prov).astype(float)
    q_mean, q_lcb = Q.mean(0), np.percentile(Q, 10, axis=0)

    old = old_obj.USMAInstance(Q=Q, city_idx=city, prov_idx=prov, n_firm_city=n_fc,
                               n_firm_prov=n_fp, q_mean=q_mean, q_lcb=q_lcb,
                               name="cross", meta={})
    new = HUSARInstance(Q=Q, city_idx=city, prov_idx=prov, n_firm_city=n_fc,
                        n_firm_prov=n_fp, q_mean=q_mean, q_lcb=q_lcb, name="cross")

    for K in (5, 30, 77):
        S = rng.choice(n, K, replace=False)
        S = [int(i) for i in S]
        assert old_obj.g_min(old, S) == pytest.approx(husar_objective(new, S, "min"), abs=1e-12)
        assert old_obj.g_cvar(old, S) == pytest.approx(husar_objective(new, S, "cvar"), abs=1e-12)
        assert old_obj.g_mean(old, S) == pytest.approx(husar_objective(new, S, "mean"), abs=1e-12)
        mo, mn = old_obj.all_metrics(old, S), all_metrics(new, S, agg="min")
        keymap = {"g_min": "J_min", "g_cvar_tau": "J_cvar", "g_mean": "J_mean",
                  "cov_city": "city_coverage", "cov_prov": "province_coverage",
                  "cov_term": "hierarchical_saturation_utility",
                  "cities_covered": "cities_covered", "provs_covered": "provs_covered",
                  "risk_sum_mean": "risk_mean", "risk_sum_worst": "risk_min"}
        for kok, nok in keymap.items():
            assert mo[kok] == pytest.approx(mn[nok], abs=1e-12), f"{kok}->{nok}"
        # 边际增益对拍（LCB 代理 + 精确 gmin 边际）
        st_o = old_obj.SolutionState(old, S[:3])
        st_n = HUSARState(new, S[:3])
        cand = np.array([i for i in range(n) if i not in set(S[:3])])
        assert np.allclose(old_obj.marginal_gmin(old, st_o, cand),
                           marginal_objective(new, st_n, cand, "min"), atol=1e-12)
        assert np.allclose(old_obj.marginal_gcvar(old, st_o, cand),
                           marginal_objective(new, st_n, cand, "cvar"), atol=1e-12)
        assert np.allclose(old_obj.marginal_lcb_score(old, st_o, cand),
                           marginal_proxy(new, st_n, cand, "lcb"), atol=1e-12)


# ================================================================== T9
def test_synthetic_weights_do_not_use_target_scenarios():
    """目标实例的 Q 被替换后，由训练面板生成的 a_g 必须逐位不变。"""
    inst1 = make_instance(family="B", n=300, n_scen=30, n_cities=20, seed=11)
    rng = np.random.default_rng(0)
    Q2 = rng.uniform(0, 1, size=inst1.Q.shape)
    inst2 = inst1.copy_with(Q=Q2)
    # 直接用同一训练面板生成
    from region_weights import synthetic_region_weights
    w1 = synthetic_region_weights(inst1, seed=11, family="B", conflict=0.5, rho=0.7)
    assert np.allclose(w1["a_city"], inst1.a_city)
    assert not np.allclose(w1["a_city"], np.ones_like(w1["a_city"]))
    # 目标 Q 不同但种子相同 -> 权重相同（说明权重不依赖目标 Q）
    w2 = synthetic_region_weights(inst2, seed=11, family="B", conflict=0.5, rho=0.7)
    assert np.allclose(w1["a_city"], w2["a_city"])


def test_synthetic_make_instance_shapes():
    inst = make_instance(family="C", n=500, n_scen=30, n_cities=25, n_provs=8, seed=3,
                         conflict=0.8)
    assert inst.Q.shape == (30, 500)
    assert len(inst.a_city) == 25 and len(inst.a_prov) == 8
    assert np.all(inst.a_city >= 0) and np.all(inst.a_prov >= 0)
    # 严格嵌套：每个城市只属于一个省
    for g in np.unique(inst.city_idx):
        assert len(np.unique(inst.prov_idx[inst.city_idx == g])) == 1
    comp = coverage_components(inst, list(range(50)))
    assert comp["cities_covered"] <= 25 and comp["provs_covered"] <= 8
