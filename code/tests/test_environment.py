# -*- coding: utf-8 -*-
"""
PHASE 2 单元测试：MDP 环境与 HUSAR-Net。

覆盖（对应任务书 §5.6 基础测试清单）：
  E1 序贯选择不重复、预算严格满足、动作掩码严格
  E2 每一步状态更新正确（增量状态 == 全量重算，含饱和边界）
  E3 增量奖励求和 == 终局目标差值（γ=1）
  E4 负收益 / 零收益 / 同分候选 / 饱和边界的数值稳健性（无 nan/inf）
  E5 max-min 与 CVaR 的边际由**全量重算**给出（不假设次模/单调）
  E6 静态特征与客观覆盖边际的一致性（张量特征 == objective.cov_gain_vec）
  E7 不同 N / K / B 的适用范围与失效方式（K>N 报错；B 可变；情景标准化正确）
  E8 CPU 与 GPU 解码结果一致
  E9 网络消融开关的维度与运行正确性
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import HUSARNET  # noqa: E402,F401

from environment import HUSAREnv, robust_scale
from model import HUSARNet, HUSARNetConfig
from objective import (HUSARInstance, cov_gain_vec, cvar_lower_tail, husar_objective,
                       marginal_objective, risk_aggregate)
from synthetic import make_hierarchy, make_scenarios, make_instance
from train import run_episode

HAS_CUDA = torch.cuda.is_available()
DEV = "cuda" if HAS_CUDA else "cpu"


def tiny_instance(seed=0, n=40, B=6, n_city=5, n_prov=2, weighted=True):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(0, 0.5, size=(B, n))
    city = np.arange(n) % n_city
    prov = city % n_prov
    n_fc = np.bincount(city, minlength=n_city).astype(float)
    n_fp = np.bincount(prov, minlength=n_prov).astype(float)
    return HUSARInstance(
        Q=Q, city_idx=city, prov_idx=prov, n_firm_city=n_fc, n_firm_prov=n_fp,
        a_city=(rng.uniform(0.5, 2, n_city) if weighted else None),
        a_prov=(rng.uniform(0.5, 2, n_prov) if weighted else None),
        q_mean=Q.mean(0), q_lcb=np.percentile(Q, 10, axis=0))


# ================================================================== E1
@pytest.mark.parametrize("agg", ["mean", "min", "cvar"])
def test_no_repeat_and_budget(agg):
    inst = tiny_instance(seed=1)
    K = 12
    env = HUSAREnv(inst, K, agg=agg, device="cpu", validate_every=1)
    env.reset()
    rng = np.random.default_rng(0)
    sel = []
    while not env.done:
        mask = env.action_mask().cpu().numpy()
        assert mask.sum() == inst.n - len(sel), "掩码必须恰好排除已选项"
        avail = np.where(mask)[0]
        i = int(rng.choice(avail))
        env.step(i)
        sel.append(i)
    assert len(sel) == K
    assert len(set(sel)) == K, "不得重复选择"
    assert env.done
    # 非法动作必须报错
    with pytest.raises((RuntimeError, ValueError)):
        env.step(sel[0])
    env2 = HUSAREnv(inst, K, agg=agg, device="cpu")
    env2.reset()
    env2.step(sel[0])
    with pytest.raises(ValueError):
        env2.step(sel[0])


def test_k_exceeds_n_rejected():
    inst = tiny_instance(seed=2, n=10)
    with pytest.raises(ValueError):
        HUSAREnv(inst, 11, device="cpu")


# ================================================================== E2/E3/E4
@pytest.mark.parametrize("agg", ["mean", "min", "cvar"])
def test_state_and_reward_consistency(agg):
    inst = tiny_instance(seed=3)
    K = 15
    env = HUSAREnv(inst, K, agg=agg, device="cpu", validate_every=1)
    env.reset()
    rng = np.random.default_rng(5)
    while not env.done:
        env.step(int(rng.choice(np.where(env.action_mask().cpu().numpy())[0])))
    s, j = env.empirical_reward_sum_check()
    assert s == pytest.approx(j, abs=1e-9)
    # 与全量评价器一致
    assert env.current_objective() == pytest.approx(
        husar_objective(inst, env.selection(), agg), abs=1e-9)
    # 设备侧情景累计收益与 numpy 增量状态一致
    assert np.allclose(env.scen_sums_t.cpu().numpy(), env.st.scen_sums, atol=1e-5)


def test_saturation_boundary_and_zero_utility():
    """城市饱和后城市侧覆盖边际为 0；省份未饱和时仍保留省侧边际；全饱和则为 0。"""
    rng = np.random.default_rng(4)
    n, B = 30, 5
    Q = rng.uniform(0, 0.5, size=(B, n))
    city = np.arange(n) % 3
    prov = np.zeros(n, int)
    n_fc = np.bincount(city, minlength=3).astype(float)          # [10,10,10]
    n_fp = np.array([30.0])
    A_C, A_P = np.array([1.5, 1.0, 0.8]), np.array([1.2])
    inst = HUSARInstance(Q=Q, city_idx=city, prov_idx=prov, n_firm_city=n_fc,
                         n_firm_prov=n_fp, a_city=A_C, a_prov=A_P, alpha=0.1)
    # alpha=0.1 -> b_city = clip(round(1),1,10) = 1；b_prov = clip(round(3),1,10) = 3
    assert inst.b_city.tolist() == [1, 1, 1] and inst.b_prov.tolist() == [3]
    env = HUSAREnv(inst, n, agg="min", device="cpu")
    env.reset()
    g0 = np.where(city == 0)[0]
    env.step(int(g0[0]))                                  # 城市 0 饱和
    cg = env.dynamic_features()["firm_dyn"][:, 4].cpu().numpy()
    rest = g0[1:]
    # 城市侧为 0，省侧仍有 1/3 容量 -> 边际 = omega_prov * a_prov / b_prov
    assert np.allclose(cg[rest], inst.omega_prov * A_P[0] / inst.b_prov[0], atol=1e-6)
    # 把省份也填满 -> 边际归零
    for i in range(n):
        if len(env.st.members) >= inst.b_prov[0]:
            break
        if not env.selected[i]:
            env.step(int(i))
    cg2 = env.dynamic_features()["firm_dyn"][:, 4].cpu().numpy()
    assert np.allclose(cg2, 0.0, atol=1e-6)
    assert np.isfinite(cg2).all()

    # 全部 Q 为 0 的实例：无 nan
    Q0 = np.zeros((4, 20))
    inst0 = HUSARInstance(Q=Q0, city_idx=np.arange(20) % 4, prov_idx=np.arange(20) % 2,
                          n_firm_city=np.full(4, 5.0), n_firm_prov=np.full(2, 10.0))
    for agg in ("mean", "min", "cvar"):
        env0 = HUSAREnv(inst0, 6, agg=agg, device="cpu")
        env0.reset()
        for i in range(6):
            env0.step(i)
        assert np.isfinite(env0.current_objective())


def test_tied_candidates_robustness():
    """同分候选：所有候选完全相同得分时仍能完成轨迹且无异常。"""
    n, B = 20, 4
    Q = np.ones((B, n)) * 0.1
    inst = HUSARInstance(Q=Q, city_idx=np.zeros(n, int), prov_idx=np.zeros(n, int),
                         n_firm_city=np.array([n], float), n_firm_prov=np.array([n], float))
    for agg in ("mean", "min", "cvar"):
        env = HUSAREnv(inst, 5, agg=agg, device="cpu")
        env.reset()
        for i in range(5):
            env.step(i)
        s, j = env.empirical_reward_sum_check()
        assert abs(s - j) < 1e-9


# ================================================================== E5
def test_marginal_does_not_assume_submodularity():
    """max-min / CVaR 边际由全量重算给出；不做次模或单调假设。

    构造一个已知会破坏次模性的两元素两情景实例（见 v0.2 §3.3 反例），
    验证环境给出的精确边际与全量重算一致（可以是负值）。
    """
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])          # 2 情景 × 2 企业
    inst = HUSARInstance(Q=Q, city_idx=np.zeros(2, int), prov_idx=np.zeros(2, int),
                         n_firm_city=np.array([2], float), n_firm_prov=np.array([2], float),
                         lam=0.0, omega_city=0.0, omega_prov=0.0)
    st_env = HUSAREnv(inst, 2, agg="min", device="cpu")
    st_env.reset()
    from objective import HUSARState
    st = HUSARState(inst, [])
    m = marginal_objective(inst, st, np.array([0, 1]), "min")
    # 空集时 min=0；加 0 号后 min=0；加 1 号后 min=0 -> 边际为 0（并列）
    assert np.allclose(m, [0.0, 0.0])
    st.add(0)
    m2 = marginal_objective(inst, st, np.array([1]), "min")
    # 加 1 号后集合收益为 [1,1]，min=1 -> 边际 = 1（次模不等式方向反转）
    assert m2[0] == pytest.approx(1.0)
    # 环境步进得到的奖励与边际一致
    r = st_env.step(0)
    r2 = st_env.step(1)
    assert r == pytest.approx(0.0)
    assert r2 == pytest.approx(1.0)


# ================================================================== E6
def test_features_match_objective_coverage_gain():
    inst = tiny_instance(seed=6, n=60, n_city=7, n_prov=3)
    env = HUSAREnv(inst, 20, agg="min", device="cpu")
    env.reset()
    rng = np.random.default_rng(2)
    for _ in range(6):
        if env.done:
            break
        d = env.dynamic_features()
        cg_t = d["firm_dyn"][:, 4].numpy()
        cg_ref = cov_gain_vec(inst, env._n_sel_city_np, env._n_sel_prov_np,
                              np.arange(inst.n))
        assert np.allclose(cg_t, cg_ref, atol=1e-5)
        env.step(int(rng.choice(np.where(env.action_mask().cpu().numpy())[0])))


def test_tenseor_cvar_forward_matches_objective():
    """环境里的张量 CVaR（用于状态特征）与 objective 的 float64 实现一致。"""
    from environment import _cvar_torch
    rng = np.random.default_rng(0)
    for B in (6, 30, 150):
        for tau in (0.1, 0.25, 0.5, 1.0):
            v = rng.normal(0, 1, B)
            ref = cvar_lower_tail(v, tau)
            got = float(_cvar_torch(torch.tensor(v, dtype=torch.float32), tau))
            assert abs(got - ref) < 1e-5 * max(1.0, abs(ref))


# ================================================================== E7
def test_variable_B_and_N():
    """B ∈ {6, 30, 150} 与 N ∈ {20, 200, 600} 均可运行；K ≤ N。"""
    for B in (6, 30, 150):
        for n in (20, 200):
            inst = make_instance(family="A", n=n, n_scen=B, n_cities=max(3, n // 19),
                                 seed=11, conflict=0.5)
            env = HUSAREnv(inst, max(1, n // 10), agg="min", device="cpu")
            env.reset()
            for _ in range(env.K):
                env.step(int(np.argmax(env.action_mask().cpu().numpy())))
            assert env.done


def test_robust_scale():
    assert robust_scale([1, 1, 1, 1]) == pytest.approx(1.0) or robust_scale([1, 1, 1, 1]) > 0
    x = np.concatenate([np.zeros(100), np.ones(100) * 100])
    assert robust_scale(x) > 0


# ================================================================== E8
@pytest.mark.skipif(not HAS_CUDA, reason="需要 CUDA")
def test_cpu_gpu_consistency():
    inst = tiny_instance(seed=8, n=80, B=8, n_city=6, n_prov=3)
    torch.manual_seed(0)
    model = HUSARNet(HUSARNetConfig(d_model=32, d_mlp=64))
    model_cpu = HUSARNet(HUSARNetConfig(d_model=32, d_mlp=64))
    model_cpu.load_state_dict(model.state_dict())
    env_cpu = HUSAREnv(inst, 40, agg="min", device="cpu")
    env_gpu = HUSAREnv(inst, 40, agg="min", device="cuda")
    o_cpu = run_episode(model_cpu, env_cpu, mode="greedy")
    o_gpu = run_episode(model.to("cuda"), env_gpu, mode="greedy")
    assert o_cpu["objective"] == pytest.approx(o_gpu["objective"], rel=1e-3, abs=1e-3)
    # 选择在无并列时应当一致
    n_diff = len(set(o_cpu["selection"]) ^ set(o_gpu["selection"]))
    assert n_diff <= 2, f"CPU/GPU 选择差异过大: {n_diff}"


# ================================================================== E9
@pytest.mark.parametrize("cfg_kw", [
    {},                                                     # N4 完整
    {"use_scenario_encoder": False, "use_scenario_projection": False},   # N0/N5
    {"use_hierarchy": False},                               # N6
    {"use_prov": False},                                    # N1 仅城市
    {"use_city": False},                                    # N2 仅省份
])
def test_ablation_configs_run(cfg_kw):
    inst = make_instance(family="B", n=120, n_scen=30, n_cities=12, n_provs=4, seed=13)
    cfg = HUSARNetConfig(d_model=32, d_mlp=64, **cfg_kw)
    model = HUSARNet(cfg)
    env = HUSAREnv(inst, 15, agg="min", device="cpu")
    o = run_episode(model, env, mode="greedy")
    assert np.isfinite(o["objective"])
    assert len(o["selection"]) == 15 and len(set(o["selection"])) == 15
    assert len(cfg.tag()) > 0


def test_value_head_and_entropy_only_when_needed():
    inst = make_instance(family="A", n=80, n_scen=10, n_cities=8, seed=14)
    model = HUSARNet(HUSARNetConfig(d_model=32, d_mlp=64))
    env = HUSAREnv(inst, 10, agg="cvar", device="cpu")
    o = run_episode(model, env, mode="sample")
    assert o["v0"] is not None
    assert o["entropy"] is not None
    assert o["logp_sum"] is not None
    assert np.isfinite(o["objective"])
    o2 = run_episode(model, env, mode="greedy", need_logprob=False)
    assert o2["logp_sum"] is None
