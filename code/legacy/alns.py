# -*- coding: utf-8 -*-
"""
LCB-ALNS —— 以 LCB 贪心为初值的情景鲁棒自适应大邻域搜索。

整体结构
--------
    LCB 贪心初始化  ->  [ 破坏(destroy) -> 修复(repair) -> 目标评价 -> 接受/回滚
                          -> 更新算子权重 ]  迭代循环  ->  输出历史最优解

破坏算子（至少 4 个）
    1. random_removal            随机移除
    2. low_contribution_removal  移除对当前鲁棒目标贡献最低的企业
    3. spatial_redundancy_removal 优先移除空间覆盖边际收益最小（辖区已饱和）的企业
    4. related_removal           集中移除同一辖区或情景风险结构相似的企业

修复算子（至少 3 个）
    1. lcb_repair            LCB 代理边际增益补齐
    2. worst_scenario_repair 原始 max-min 目标边际增益补齐
    3. swg_repair            情景加权（SWG 平滑目标）边际增益补齐

接受准则
    'strict' : 仅接受严格改进
    'sa'     : 模拟退火式接受（按时间预算退火），历史最优解单独保存

状态一致性
    所有增删都通过 objectives.SolutionState 的增量缓存完成；每轮结束调用 validate()
    与全量重算对拍（verify=True 时），确保缓存与重新计算完全一致。

消融模式（mode）
    'alns'     : 完整自适应权重
    'fixed'    : 固定算子对（不用自适应），用于区分"大邻域"与"自适应选择"的贡献
    'no_adaptive': 所有算子等概率随机选择，不做权重更新
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from objectives import (USMAInstance, SolutionState, all_metrics, cov_gain_vec,
                        marginal_gmin, marginal_gcvar, marginal_lcb_score)
from scenario_weighted_greedy import delta_exact, delta_weighted, scenario_weights

DESTROY_OPS = ("random", "low_contribution", "spatial_redundancy", "related")
REPAIR_OPS = ("lcb", "worst_scenario", "swg")


@dataclass
class ALNSConfig:
    """ALNS 超参数（所有实验共用的默认值见 config.yaml）。"""
    removal_frac: float = 0.10          # 移除比例 q
    min_removal: int = 4
    max_removal: int = 80               # 移除规模上限（控制大规模实例代价）
    candidate_pool: int | None = None   # 修复候选池大小 M（None -> max(4K, 1500)）
    accept: str = "sa"                  # 'sa' | 'strict'
    objective: str = "maxmin"           # 'maxmin' | 'cvar'
    tau: float = 0.25
    mode: str = "alns"                  # 'alns' | 'fixed' | 'no_adaptive'
    fixed_pair: tuple = ("low_contribution", "worst_scenario")
    eta_kappa: float = 1.0              # SWG 修复算子的平滑温度（相对情景收益尺度）
    eta_scale: float = 1.0              # 运行前由 outcome_scale 填入
    reaction: float = 0.20              # 权重平滑系数 rho
    segment: int = 40                   # 每 segment 次迭代更新一次权重
    sigma1: float = 33.0                # 刷新全局最优
    sigma2: float = 9.0                 # 优于当前解
    sigma3: float = 3.0                 # 被接受的解
    sa_T0_frac: float = 0.05            # T0 = frac * |v0| / ln 2
    sa_Tend_frac: float = 0.02          # Tend = Tend_frac * T0
    verify_every: int = 0               # >0 时每 N 轮与全量重算对拍
    seed: int = 20261015


class ALNS:
    def __init__(self, inst: USMAInstance, K: int, init_selection, cfg: ALNSConfig | None = None):
        self.inst = inst
        self.K = int(K)
        self.cfg = cfg or ALNSConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.st = SolutionState(inst, [int(i) for i in init_selection])
        if len(self.st.members) != self.K:
            raise ValueError(f"init selection size {len(self.st.members)} != K={self.K}")
        self.st.validate()
        self.n_iter = 0
        self.n_rejected = 0
        self.n_full_eval = 0
        self.n_cand_eval = 0
        self.eta = 1.0     # 由 _prepare / set_eta 覆盖
        self.best = self.st.copy()
        self.init_val = self._eval(self.best)
        self.best_val = self.init_val
        self.cur_val = self.best_val
        self._rollback_members = list(self.st.members)
        # 候选池：按 LCB 代理分降序的前 M 个企业
        M = self.cfg.candidate_pool or max(4 * self.K, 1500)
        self.pool = np.argsort(-inst.q_lcb)[: min(M, inst.n)].astype(np.int64)
        # 算子权重与统计
        self.wD = {op: 1.0 for op in DESTROY_OPS}
        self.wR = {op: 1.0 for op in REPAIR_OPS}
        self.stats = {(d, r): {"calls": 0, "score": 0.0, "best": 0, "accepted": 0,
                               "sum_gain": 0.0} for d in DESTROY_OPS for r in REPAIR_OPS}
        self._seg_score = {(d, r): 0.0 for d in DESTROY_OPS for r in REPAIR_OPS}
        self._seg_calls = {(d, r): 0 for d in DESTROY_OPS for r in REPAIR_OPS}

    # ------------------------------------------------------------ 目标
    def _eval(self, st: SolutionState) -> float:
        self.n_full_eval += 1
        if self.cfg.objective == "cvar":
            return st.f_cvar(self.cfg.tau)
        return st.f_min()

    def _removal_size(self) -> int:
        cfg = self.cfg
        r = int(round(cfg.removal_frac * self.K))
        return int(np.clip(r, cfg.min_removal, min(cfg.max_removal, self.K - 1)))

    # ------------------------------------------------------------ 破坏算子
    def _destroy(self, op: str, r: int) -> list[int]:
        inst, st = self.inst, self.st
        members = np.asarray(st.members, dtype=np.int64)
        if len(members) <= r:
            return members.tolist()
        if op == "random":
            return self.rng.choice(members, size=r, replace=False).tolist()

        if op == "low_contribution":
            base = self._eval(st)
            # 逐个移除的真实损失（向量化：移除 i 后的最差情景收益）
            sums = st.scen_sums
            after = (sums[:, None] - inst.Q[:, members]).min(axis=0)
            cov_after = self._cov_after_removal(members)
            contrib = (sums.min() + st.cov_term) - (after + cov_after)
            self.n_cand_eval += len(members)
            # 贡献最小者先移除（并列时随机）
            key = contrib + 1e-12 * self.rng.random(len(members))
            return members[np.argsort(key)[:r]].tolist()

        if op == "spatial_redundancy":
            cov_loss = self._cov_loss_on_removal(members)
            key = cov_loss + 1e-9 * self.rng.random(len(members))
            return members[np.argsort(key)[:r]].tolist()

        if op == "related":
            seed = int(members[self.rng.integers(len(members))])
            q0 = inst.Q[:, seed]
            # 情景风险结构相似度（负 L2 距离）+ 同辖区加成
            d2 = np.linalg.norm(inst.Q[:, members] - q0[:, None], axis=0)
            bonus = (inst.city_idx[members] == inst.city_idx[seed]).astype(float)
            score = -d2 + 0.25 * bonus * (d2.max() + 1.0) + 1e-9 * self.rng.random(len(members))
            return members[np.argsort(-score)[:r]].tolist()

        raise ValueError(op)

    def _cov_after_removal(self, members) -> np.ndarray:
        """向量化：一次性移除 members 中每个企业后覆盖项（乘 lam 前）的损失所对应的剩余值。

        返回每个 i 对应的 lam*(C(S)-ΔC_removal(i))，即移除 i 后的覆盖项值。
        """
        inst, st = self.inst, self.st
        out = np.zeros(len(members))
        if inst.omega_city > 0:
            g = inst.city_idx[members]
            act = st.n_sel_city[g] <= inst.b_city[g]
            loss = np.where(act, inst.v_city[g] / inst.b_city[g], 0.0)
            out += inst.omega_city * (st.cov_city - loss)
        if inst.omega_prov > 0:
            gp = inst.prov_idx[members]
            act = st.n_sel_prov[gp] <= inst.b_prov[gp]
            loss = np.where(act, inst.v_prov[gp] / inst.b_prov[gp], 0.0)
            out += inst.omega_prov * (st.cov_prov - loss)
        return inst.lam * out

    def _cov_loss_on_removal(self, members) -> np.ndarray:
        """移除每个成员造成的覆盖项损失（越小越"冗余"）。"""
        inst, st = self.inst, self.st
        loss = np.zeros(len(members))
        if inst.omega_city > 0:
            g = inst.city_idx[members]
            act = st.n_sel_city[g] <= inst.b_city[g]
            loss += inst.omega_city * np.where(act, inst.v_city[g] / inst.b_city[g], 0.0)
        if inst.omega_prov > 0:
            gp = inst.prov_idx[members]
            act = st.n_sel_prov[gp] <= inst.b_prov[gp]
            loss += inst.omega_prov * np.where(act, inst.v_prov[gp] / inst.b_prov[gp], 0.0)
        return loss

    # ------------------------------------------------------------ 修复算子
    def _repair_candidates(self, extra=()) -> np.ndarray:
        st = self.st
        cand = self.pool[~st.inS[self.pool]]
        if len(extra):
            ex = np.asarray([i for i in extra if not st.inS[i]], dtype=np.int64)
            if len(ex):
                cand = np.unique(np.concatenate([cand, ex]))
        return cand

    def _repair(self, op: str, n_slots: int, extra=()) -> bool:
        """用指定算子补齐 n_slots 个名额；返回是否成功补齐。"""
        inst, st = self.inst, self.st
        for _ in range(n_slots):
            cand = self._repair_candidates(extra)
            if len(cand) == 0:
                return False
            if op == "lcb":
                m = marginal_lcb_score(inst, st, cand)
            elif op == "worst_scenario":
                if self.cfg.objective == "cvar":
                    m = marginal_gcvar(inst, st, cand, self.cfg.tau)
                else:
                    m = marginal_gmin(inst, st, cand)
            elif op == "swg":
                m = delta_exact(inst, st, cand, self.eta)
            else:
                raise ValueError(op)
            self.n_cand_eval += len(cand)
            j = int(np.argmax(m))
            if not np.isfinite(m[j]):
                return False
            st.add(int(cand[j]))
        return True

    # ------------------------------------------------------------ 权重选择
    def _pick(self, weights: dict) -> str:
        ops = list(weights.keys())
        w = np.asarray([weights[o] for o in ops], dtype=float)
        w = np.maximum(w, 1e-9)
        p = w / w.sum()
        return ops[int(self.rng.choice(len(ops), p=p))]

    def _update_weights(self):
        rho = self.cfg.reaction
        for d in DESTROY_OPS:
            for r in REPAIR_OPS:
                c = self._seg_calls[(d, r)]
                if c > 0:
                    avg = self._seg_score[(d, r)] / c
                    self.wD[d] = (1 - rho) * self.wD[d] + rho * max(avg, 1e-6)
                    self.wR[r] = (1 - rho) * self.wR[r] + rho * max(avg, 1e-6)
                self._seg_score[(d, r)] = 0.0
                self._seg_calls[(d, r)] = 0

    # ------------------------------------------------------------ 主循环
    def run(self, time_budget: float = 60.0, max_iters: int = 10 ** 9,
            track_anytime: bool = True) -> dict:
        cfg = self.cfg
        t0 = time.time()
        anytime = [(0.0, self.best_val)]
        # SA 温度：以初值为参照，允许约 5% 相对劣化有 50% 接受率
        T0 = max(cfg.sa_T0_frac * abs(self.best_val), 1e-9) / np.log(2.0)
        Tend = cfg.sa_Tend_frac * T0
        for it in range(max_iters):
            elapsed = time.time() - t0
            if elapsed >= time_budget:
                break
            self.n_iter += 1
            frac = min(1.0, elapsed / max(time_budget, 1e-9))
            T = T0 * (Tend / T0) ** frac

            # 选择算子
            if cfg.mode == "fixed":
                d_op, r_op = cfg.fixed_pair
            elif cfg.mode == "no_adaptive":
                d_op = DESTROY_OPS[int(self.rng.integers(len(DESTROY_OPS)))]
                r_op = REPAIR_OPS[int(self.rng.integers(len(REPAIR_OPS)))]
            else:
                d_op = self._pick(self.wD)
                r_op = self._pick(self.wR)

            self._seg_calls[(d_op, r_op)] += 1
            self.stats[(d_op, r_op)]["calls"] += 1
            before = self.cur_val

            removed = self._destroy(d_op, self._removal_size())
            for i in removed:
                self.st.remove(int(i))
            ok = self._repair(r_op, len(removed), extra=removed)
            if not ok or len(self.st.members) != self.K:
                # 回滚
                self.st = SolutionState(self.inst, self._rollback_members)
                self.n_rejected += 1
                continue
            self._rollback_members = list(self.st.members)

            if cfg.verify_every and self.n_iter % cfg.verify_every == 0:
                self.st.validate()
            v = self._eval(self.st)

            # 接受判定
            new_best = v > self.best_val + 1e-12
            accepted = False
            if v > self.cur_val + 1e-12:
                accepted = True
            elif cfg.accept == "sa" and np.exp((v - self.cur_val) / max(T, 1e-12)) > self.rng.random():
                accepted = True

            if new_best:
                self.best_val = v
                self.best = self.st.copy()
                self._seg_score[(d_op, r_op)] += cfg.sigma1
                self.stats[(d_op, r_op)]["best"] += 1
            elif accepted and v > before + 1e-12:
                self._seg_score[(d_op, r_op)] += cfg.sigma2
            elif accepted:
                self._seg_score[(d_op, r_op)] += cfg.sigma3

            if accepted:
                self.cur_val = v
                self.stats[(d_op, r_op)]["accepted"] += 1
                self.stats[(d_op, r_op)]["sum_gain"] += v - before
            else:
                self.n_rejected += 1
                self.st = SolutionState(self.inst, self._rollback_members)
                self.st.validate()

            if track_anytime:
                anytime.append((time.time() - t0, self.best_val))
            if (it + 1) % cfg.segment == 0:
                self._update_weights()

        self.best.validate()
        met = all_metrics(self.inst, self.best.members, tau=cfg.tau)
        out = {"method": "lcb_alns", "selection": self.best.members, "n": len(self.best.members),
               "g_min": met["g_min"], "g_cvar": met["g_cvar_tau"], "g_mean": met["g_mean"],
               "cities_covered": met["cities_covered"], "provs_covered": met["provs_covered"],
               "time": time.time() - t0, "iters": self.n_iter,
               "n_rejected": self.n_rejected,
               "n_full_eval": self.n_full_eval, "n_cand_eval": self.n_cand_eval,
               "objective_value": self.best_val,
               "mode": cfg.mode, "accept": cfg.accept, "objective": cfg.objective,
               "removal_frac": cfg.removal_frac, "eta": self.eta,
               "op_stats": {f"{d}|{r}": dict(v) for (d, r), v in self.stats.items()},
               "final_weights_D": dict(self.wD), "final_weights_R": dict(self.wR)}
        if track_anytime:
            out["anytime"] = anytime
        return out

    # ------------------------------------------------------------
    def _init_rollback(self):
        self._rollback_members = list(self.st.members)

    def set_eta(self, eta: float):
        self.eta = float(eta)


def _prepare(inst: USMAInstance, K: int, init_selection, cfg: ALNSConfig, eta: float | None):
    a = ALNS(inst, K, init_selection, cfg)
    a._init_rollback()
    if eta is None:
        from scenario_weighted_greedy import outcome_scale, eta_from_kappa
        scale = outcome_scale(inst, a.best.members)
        eta = eta_from_kappa(scale, cfg.eta_kappa)
        cfg.eta_scale = scale
    a.set_eta(eta)
    return a


def run_alns(inst: USMAInstance, K: int, init_selection=None, cfg: ALNSConfig | None = None,
             time_budget: float = 60.0, max_iters: int = 10 ** 9, eta: float | None = None,
             track_anytime: bool = True) -> dict:
    """便捷入口：默认以 LCB 贪心解为初值。"""
    cfg = cfg or ALNSConfig()
    if init_selection is None:
        from baselines import celf_greedy
        init_selection = celf_greedy(inst, K, proxy="lcb")["selection"]
    a = _prepare(inst, K, init_selection, cfg, eta)
    out = a.run(time_budget=time_budget, max_iters=max_iters, track_anytime=track_anytime)
    out["init_g_min"] = a.init_val
    out["seed"] = cfg.seed
    out["K"] = K
    return out
