# -*- coding: utf-8 -*-
"""
HUSAR-Net 序贯决策环境（MDP）。

MDP 定义
--------
    S_0 = ∅；S_{t+1} = S_t ∪ {a_t}，t = 0..K-1；动作 a_t ∈ V \\ S_t（严格可行掩码）。
    奖励（目标一致增量）： r_t = J_HUSAR(S_{t+1}) − J_HUSAR(S_t)，γ = 1
    于是 Σ_t r_t = J_HUSAR(S_K) − J_HUSAR(∅)（tests 验证）。

状态（本环境提供给策略网络的全部信息，不含目标期标签）
------------------------------------------------------
静态（每个实例预计算一次）：
  * 企业风险分布特征：情景均值/标准差/最小/分位/最大（由 Bootstrap 情景矩阵 Q 派生）；
  * **跨企业联合情景标准化**：每个情景内企业风险分的 z 分数 (Q[b,i] − μ_b)/σ_b，
    以及原始风险分按实例稳健尺度标准化后的值（供情景集合编码器使用）；
  * 层级空间静态属性：城市/省份在册规模、饱和阈值 b_g、区域权重 a_g。
动态（每一步）：
  * 企业层：是否已选、其城市/省份的当前饱和度、加选该企业的覆盖边际 ΔC_i、
    当前绑定情景（argmin_b 累计收益）下该企业的风险分；
  * 层级层：城市/省份当前选中计数、是否已达饱和、剩余候选数；
  * 全局层：进度、当前各情景累计风险收益（含排序分位）、目标类型（mean/min/CVaR）、
    尾部水平 τ 与风险—空间权衡 λ。

环境不含企业身份、不含任何目标期真实标签；奖励由 objective.py 的唯一评价器计算，
增量状态每步 validate()（可选开启，测试时开启）。
"""
from __future__ import annotations

import numpy as np
import torch

from objective import (HUSARInstance, HUSARState, cov_gain_vec, cvar_lower_tail,
                       risk_aggregate)

DVEC_DYN = 10          # 企业层动态特征维数
GVEC_DYN = 18          # 全局动态特征维数（2 进度 + 5 统计 + 5 分位 + 3 目标 + 3 参数）


def _to_quantile(x: np.ndarray) -> np.ndarray:
    """把一维分数映射到 [0,1] 的分位位置（平均秩处理并列，无标签信息）。"""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / max(1.0, n - 1)


# =====================================================================
def _cvar_torch(values: torch.Tensor, tau: float) -> torch.Tensor:
    """收益下尾 CVaR（离散精确尾部）的张量实现，与 objective.cvar_lower_tail 一致。"""
    B = values.numel()
    m = tau * B
    if m >= B - 1e-12:
        return values.mean()
    j = int(np.floor(m + 1e-12))
    frac = m - j
    srt = torch.sort(values).values
    if frac <= 1e-12:
        return srt[:j].mean()
    return (srt[:j].sum() + frac * srt[j]) / m


def robust_scale(x: np.ndarray) -> float:
    """稳健尺度：IQR/1.349（近似正态 sd），退化时退回 std 或 1。"""
    x = np.asarray(x, dtype=float).ravel()
    q75, q25 = np.percentile(x, [75, 25])
    s = (q75 - q25) / 1.349
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.std(x))
    return float(s) if s > 1e-12 else 1.0


# =====================================================================
class InstanceTensors:
    """把一个 HUSARInstance 预处理为策略网络使用的张量（CPU 或 CUDA）。"""

    def __init__(self, inst: HUSARInstance, device="cpu", agg: str = "min",
                 tau: float = 0.25):
        self.inst = inst
        self.device = torch.device(device)
        self.agg = agg
        self.tau = float(tau)
        Q = inst.Q
        self.scale = robust_scale(Q)                      # 风险分稳健尺度（实例级）
        # --- 情景内跨企业标准化（联合情景结构）
        mu_b = Q.mean(axis=1, keepdims=True)
        sd_b = Q.std(axis=1, keepdims=True)
        self.z_scen = (Q - mu_b) / np.maximum(sd_b, 1e-9)      # (B, n)
        self.raw_scen = Q / self.scale                          # (B, n) 尺度标准化
        # --- 企业层风险分布特征
        q = Q
        feats = np.stack([
            q.mean(0), q.std(0), q.min(0),
            np.percentile(q, 10, axis=0), np.percentile(q, 50, axis=0),
            np.percentile(q, 90, axis=0), q.max(0),
        ], axis=1) / self.scale                                  # (n, 7)
        # --- 层级空间静态属性
        ci, pi = inst.city_idx, inst.prov_idx
        city_sz = inst.n_firm_city[ci]
        prov_sz = inst.n_firm_prov[pi]
        sp = np.stack([
            inst.a_city[ci], inst.b_city[ci] / 10.0, np.log1p(city_sz) / 6.0,
            inst.a_prov[pi], inst.b_prov[pi] / 10.0, np.log1p(prov_sz) / 8.0,
            (city_sz <= 1.0).astype(float),
        ], axis=1)                                               # (n, 7)
        # --- 实例内相对风险位置（静态、无标签）：全样本分位 + 所属城市内分位
        rank_inst = _to_quantile(feats[:, 0])                     # q_mean 的实例内分位
        rank_inst_lcb = _to_quantile(feats[:, 4])                 # 10% 分位的实例内分位
        rank_inst_min = _to_quantile(feats[:, 2])                 # 最差情景的实例内分位
        rank_city_lcb = np.zeros(inst.n, dtype=float)
        for g in range(len(inst.n_firm_city)):
            idx = np.where(ci == g)[0]
            if len(idx):
                rank_city_lcb[idx] = _to_quantile(feats[idx, 4])
        rel = np.stack([rank_inst, rank_inst_lcb, rank_inst_min, rank_city_lcb], axis=1)
        sp = np.concatenate([sp, rel], axis=1)                    # (n, 11)
        self.static = torch.tensor(np.concatenate([feats, sp], axis=1),
                                   dtype=torch.float32, device=self.device)
        self.d_static = self.static.shape[1]

        # --- 供环境使用的 torch 常量
        self.Q_t = torch.tensor(Q, dtype=torch.float32, device=self.device)          # (B,n)
        self.z_t = torch.tensor(self.z_scen, dtype=torch.float32, device=self.device)
        self.raw_t = torch.tensor(self.raw_scen, dtype=torch.float32, device=self.device)
        self.city_t = torch.tensor(ci, dtype=torch.long, device=self.device)
        self.prov_t = torch.tensor(pi, dtype=torch.long, device=self.device)
        self.b_city_t = torch.tensor(inst.b_city, dtype=torch.float32, device=self.device)
        self.b_prov_t = torch.tensor(inst.b_prov, dtype=torch.float32, device=self.device)
        self.a_city_t = torch.tensor(inst.a_city, dtype=torch.float32, device=self.device)
        self.a_prov_t = torch.tensor(inst.a_prov, dtype=torch.float32, device=self.device)
        self.n_firm_city_t = torch.tensor(inst.n_firm_city, dtype=torch.float32,
                                          device=self.device)
        # --- 预先构造的静态系数张量（避免每步重新分配/拷贝）
        self.gain_city_coef = torch.tensor(inst.omega_city * inst.a_city
                                           / np.maximum(inst.b_city, 1e-9),
                                           dtype=torch.float32, device=self.device)
        self.gain_prov_coef = torch.tensor(inst.omega_prov * inst.a_prov
                                           / np.maximum(inst.b_prov, 1e-9),
                                           dtype=torch.float32, device=self.device)
        self.a_city_firm = self.a_city_t[self.city_t]
        self.a_prov_firm = self.a_prov_t[self.prov_t]
        self.nfirm_city_firm = self.n_firm_city_t[self.city_t]

        # --- 归一化常数
        qbar = float(Q.mean())
        self.qbar = qbar if qbar > 1e-9 else 1.0
        self.scen_ref = max(1.0, self.scale)
        self.B, self.n = inst.B, inst.n
        # 由环境设置的额外全局常数（tau, lam, K/n）
        self.extra_t = torch.zeros(3, dtype=torch.float32, device=self.device)
        # 预计算静态常数（避免每步重算）
        self.a_city_over_b_firm = self.a_city_firm / self.b_city_t[self.city_t]
        self.b_city_over_10_firm = self.b_city_t[self.city_t] / 10.0
        self.scen_denom = torch.tensor([self.scen_ref], dtype=torch.float32,
                                       device=self.device)
        self.cvar_mode = "quantile"
        self.agg_onehot = {
            "mean": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            "min": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            "cvar": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)}
        self.cvar_j, self.cvar_frac, self.cvar_m = 0, 0.0, 1.0
        self.qidx = torch.linspace(0, max(0, self.B - 1), steps=5,
                                   device=self.device).round().long()

    def set_extra(self, tau: float, lam: float, k_over_n: float, K: int = 1) -> None:
        self.extra_t = torch.tensor([tau, lam, k_over_n], dtype=torch.float32,
                                    device=self.device)
        self.scen_denom = torch.tensor([self.scen_ref * max(1.0, np.sqrt(K))],
                                       dtype=torch.float32, device=self.device)
        m = tau * self.B
        if m >= self.B - 1e-12:
            self.cvar_mode = "mean"
            self.cvar_j, self.cvar_frac, self.cvar_m = self.B - 1, 0.0, float(self.B)
        else:
            self.cvar_mode = "quantile"
            self.cvar_j = int(np.floor(m + 1e-12))
            self.cvar_frac = float(m - self.cvar_j)
            self.cvar_m = float(m)
        self.qidx = torch.linspace(0, max(0, self.B - 1), steps=5,
                                   device=self.device).round().long()

    # ------------------------------------------------------------------
    def city_sat(self, n_sel_city: torch.Tensor) -> torch.Tensor:
        return torch.clamp(n_sel_city / self.b_city_t, max=1.0)

    def prov_sat(self, n_sel_prov: torch.Tensor) -> torch.Tensor:
        return torch.clamp(n_sel_prov / self.b_prov_t, max=1.0)


# =====================================================================
class HUSAREnv:
    """单实例序贯选择环境（K 步）。"""

    def __init__(self, inst: HUSARInstance, K: int, agg: str = "min", tau: float = 0.25,
                 device="cpu", validate_every: int = 0, dtype=torch.float32):
        if K > inst.n:
            raise ValueError("预算 K 不能超过候选企业数")
        self.inst = inst
        self.K = int(K)
        self.agg = agg
        self.tau = float(tau)
        self.device = torch.device(device)
        self.validate_every = int(validate_every)
        self.dtype = dtype
        self.T = InstanceTensors(inst, device=device, agg=agg, tau=tau)
        self.T.set_extra(tau, inst.lam, K / max(1, inst.n), K)
        self._n_sel_city_np = np.zeros(inst.n_city)
        self._n_sel_prov_np = np.zeros(inst.n_prov)
        self.n_sel_city = torch.zeros(inst.n_city, dtype=dtype, device=self.device)
        self.n_sel_prov = torch.zeros(inst.n_prov, dtype=dtype, device=self.device)
        self.st = HUSARState(inst, [])
        self.t = 0
        self.selected = torch.zeros(inst.n, dtype=torch.bool, device=self.device)
        self.rewards: list[float] = []
        self.scen_sums_t = torch.zeros(inst.B, dtype=dtype, device=self.device)
        self._prog_table = torch.stack([
            torch.arange(self.K + 1, dtype=dtype, device=self.device) / max(1, self.K),
            (self.K - torch.arange(self.K + 1, dtype=dtype, device=self.device))
            / max(1, self.K)], dim=1)

    # ------------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        inst = self.inst
        self.st = HUSARState(inst, [])
        self.t = 0
        self.selected = torch.zeros(inst.n, dtype=torch.bool, device=self.device)
        self.n_sel_city = torch.zeros(inst.n_city, dtype=self.dtype, device=self.device)
        self.n_sel_prov = torch.zeros(inst.n_prov, dtype=self.dtype, device=self.device)
        self._n_sel_city_np = np.zeros(inst.n_city)
        self._n_sel_prov_np = np.zeros(inst.n_prov)
        self.rewards = []
        self.scen_sums_t = torch.zeros(inst.B, dtype=self.dtype, device=self.device)

    @property
    def done(self) -> bool:
        return self.t >= self.K

    # ------------------------------------------------------------------ 状态
    def action_mask(self) -> torch.Tensor:
        """True = 可选（未选）。严格无重复、无隐藏配额。"""
        return ~self.selected

    def current_objective(self) -> float:
        return self.st.objective(self.agg, self.tau)

    def static_features(self) -> torch.Tensor:
        return self.T.static

    def dynamic_features(self) -> dict:
        """返回 (firm_dyn (n,11), global_dyn (18,))。张量运算 + 预计算常数，最小化算子数。"""
        T = self.T
        # 企业层（10 维；city_full/prov_full 与饱和度冗余，已并入 sat）
        sel = self.selected.to(self.dtype)
        city_sat = T.city_sat(self.n_sel_city)[T.city_t]
        prov_sat = T.prov_sat(self.n_sel_prov)[T.prov_t]
        n_left_city = T.nfirm_city_firm - self.n_sel_city[T.city_t]
        gc = (self.n_sel_city[T.city_t] < T.b_city_t[T.city_t]).to(self.dtype)
        gp = (self.n_sel_prov[T.prov_t] < T.b_prov_t[T.prov_t]).to(self.dtype)
        cov_gain = torch.addcmul(gc * T.gain_city_coef[T.city_t], gp,
                                 T.gain_prov_coef[T.prov_t])
        sums = self.scen_sums_t
        bstar = torch.argmin(sums)
        q_bind = T.Q_t[bstar] / T.scen_ref
        left_frac = n_left_city / torch.clamp(T.nfirm_city_firm, min=1.0)
        firm_dyn = torch.stack([sel, city_sat, prov_sat, left_frac,
                                cov_gain, q_bind, T.a_city_firm, T.a_prov_firm,
                                T.a_city_over_b_firm, T.b_city_over_10_firm], dim=1)
        # 全局层：进度(2) + 统计(5) + 分位(5) + 目标(3) + 参数(3)
        ss = sums / T.scen_denom
        srt = torch.sort(ss).values
        if T.cvar_mode == "quantile":
            cvar = (srt[:T.cvar_j].sum() + T.cvar_frac * srt[T.cvar_j]) / T.cvar_m
        else:
            cvar = ss.mean()
        stat = torch.stack([srt[0], ss.mean(), cvar, srt[-1], ss.std(unbiased=False)])
        glob = torch.cat([self._prog_table[self.t], stat, srt[T.qidx],
                          T.agg_onehot[self.agg], T.extra_t])
        return {"firm_dyn": firm_dyn, "global_dyn": glob}

    # ------------------------------------------------------------------ 步进
    def step(self, i: int) -> float:
        if self.done:
            raise RuntimeError("episode 已结束")
        i = int(i)
        if self.selected[i]:
            raise ValueError(f"非法动作：企业 {i} 已被选择（必须使用 action mask）")
        prev = self.current_objective()
        self.st.add(i)
        # 出栈式（out-of-place）更新：避免破坏已构建计算图中的张量版本
        self.selected = self.selected.clone()
        self.selected[i] = True
        self.n_sel_city = self.n_sel_city.clone()
        self.n_sel_prov = self.n_sel_prov.clone()
        g = int(self.inst.city_idx[i])
        gp = int(self.inst.prov_idx[i])
        self._n_sel_city_np[g] += 1.0
        self._n_sel_prov_np[gp] += 1.0
        self.n_sel_city[g] += 1.0
        self.n_sel_prov[gp] += 1.0
        self.scen_sums_t = self.scen_sums_t + self.T.Q_t[:, i]
        self.t += 1
        r = self.current_objective() - prev
        self.rewards.append(r)
        if self.validate_every and (self.t % self.validate_every == 0):
            self.st.validate()
        return float(r)

    def total_reward(self) -> float:
        return float(sum(self.rewards))

    def selection(self) -> list[int]:
        return list(self.st.members)

    # ------------------------------------------------------------------ 轨迹工具
    def random_rollout(self, rng: np.random.Generator) -> list[int]:
        self.reset()
        avail = np.arange(self.inst.n)
        for _ in range(self.K):
            j = rng.integers(len(avail))
            self.step(int(avail[j]))
            avail = np.delete(avail, j)
        return self.selection()

    def empirical_reward_sum_check(self) -> tuple[float, float]:
        """返回 (Σr_t, J(S_K) − J(∅))（测试用）。"""
        j0 = HUSARState(self.inst, []).objective(self.agg, self.tau)
        return self.total_reward(), self.current_objective() - j0


# =====================================================================
# 便利函数：由实例构建环境
# =====================================================================
def make_env(inst: HUSARInstance, frac: float | None = None, K: int | None = None,
             agg: str = "min", tau: float = 0.25, device="cpu",
             validate_every: int = 0) -> HUSAREnv:
    """frac 为预算比例（如 0.05），或直接给 K。"""
    if K is None:
        if frac is None:
            raise ValueError("必须给出 K 或 frac")
        K = max(1, int(round(frac * inst.n)))
    return HUSAREnv(inst, K, agg=agg, tau=tau, device=device,
                    validate_every=validate_every)
