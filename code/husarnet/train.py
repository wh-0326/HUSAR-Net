# -*- coding: utf-8 -*-
"""
HUSAR-Net 训练（REINFORCE + 基线 + 梯度裁剪）。

目标一致的奖励
--------------
    r_t = J_HUSAR(S_{t+1}) − J_HUSAR(S_t)，γ = 1
    轨迹收益 G = Σ_t r_t = J_HUSAR(S_K) − J_HUSAR(∅)

策略梯度
--------
    ∇L = −(G − b) · Σ_t ∇log π_θ(a_t | s_t)
基线 b 支持：
  * 'rollout'：同一实例上策略的**贪心 rollout** 收益（Kool et al. 2019 的 rollout baseline）；
  * 'value'  ：学习到的价值头 V(s_0)（额外 MSE 损失）；
  * 'mean'   ：同批次实例平均收益（方差较大，仅作对照）；
  * 'none'   ：无基线。
另可加熵正则（鼓励探索）与优势裁剪。

训练实例：合成情景族 A/B/C，n、B、K 比例、情景冲突在配置范围内抽样；
实例种子与验证/测试种子互斥（Protocol I）。**任何目标期真实标签都不参与训练**。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from environment import HUSAREnv, InstanceTensors
from objective import HUSARState, husar_objective
from synthetic import make_instance


# =====================================================================
@dataclass
class TrainConfig:
    # 训练规模
    iters: int = 600
    instances_per_update: int = 8
    lr: float = 3e-4
    grad_clip: float = 1.0
    entropy_coef: float = 0.02
    reward_scale: str = "per_k"        # per_k：G/K（跨预算可比）；none：原始 G
    baseline: str = "rollout"          # rollout | value | mean | none
    value_coef: float = 0.5
    advantage_norm: bool = True
    # 任务分布
    n_min: int = 200
    n_max: int = 800
    n_scen_choices: tuple = (30, 150)
    k_frac_choices: tuple = (0.02, 0.05, 0.10, 0.15)
    agg_probs: tuple = (("min", 0.5), ("cvar", 0.35), ("mean", 0.15))
    family_probs: tuple = (("A", 0.25), ("B", 0.4), ("C", 0.35))
    # 目标条件化：覆盖层级（两尺度/仅城市/无覆盖）与风险—空间权衡 lambda
    omega_variants: tuple = (("both", 0.70), ("city", 0.15), ("none", 0.15))
    lam_choices: tuple = ((0.05, 0.2), (0.10, 0.6), (0.20, 0.2))
    conflict_range: tuple = (0.3, 1.0)
    zipf_range: tuple = (1.0, 1.3)
    # 验证
    val_instances: int = 8
    val_every: int = 25
    seed: int = 0
    val_seed_base: int = 900000
    train_seed_base: int = 100000
    # 环境
    device: str = "cuda"
    tau: float = 0.25
    use_weights: bool = True
    log_every: int = 10
    out_dir: str | None = None
    tag: str = "husarnet"

    def to_json(self) -> str:
        d = asdict(self)
        d["device"] = str(d.get("device"))
        d["n_scen_choices"] = list(d.get("n_scen_choices", []))
        d["k_frac_choices"] = list(d.get("k_frac_choices", []))
        d["agg_probs"] = [list(x) for x in d.get("agg_probs", [])]
        d["family_probs"] = [list(x) for x in d.get("family_probs", [])]
        d["omega_variants"] = [list(x) for x in d.get("omega_variants", [])]
        d["lam_choices"] = [list(x) for x in d.get("lam_choices", [])]
        return json.dumps(d, ensure_ascii=False, indent=2)


# =====================================================================
def sample_train_instance(cfg: TrainConfig, rng: np.random.Generator, idx: int):
    """按任务分布抽一个训练实例（种子与验证/测试互斥）。"""
    n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
    n_scen = int(rng.choice(cfg.n_scen_choices))
    fams, ps = zip(*cfg.family_probs)
    fam = str(rng.choice(fams, p=np.array(ps) / np.sum(ps)))
    conflict = float(rng.uniform(*cfg.conflict_range))
    zipf = float(rng.uniform(*cfg.zipf_range))
    k_frac = float(rng.choice([float(x) for x in cfg.k_frac_choices]))
    seed = cfg.train_seed_base + idx
    ovs, ovp = zip(*cfg.omega_variants)
    ov = str(rng.choice(ovs, p=np.array(ovp) / np.sum(ovp)))
    om_c, om_p = {"both": (0.5, 0.5), "city": (0.5, 0.0), "none": (0.0, 0.0)}[ov]
    lams, lps = zip(*cfg.lam_choices)
    lam = float(rng.choice(lams, p=np.array(lps) / np.sum(lps)))
    inst = make_instance(family=fam, n=n, n_scen=n_scen, seed=seed, conflict=conflict,
                         zipf_a=zipf, with_weights=cfg.use_weights, lam=lam,
                         omega_city=om_c, omega_prov=om_p)
    aggs, ap = zip(*cfg.agg_probs)
    agg = str(rng.choice(aggs, p=np.array(ap) / np.sum(ap)))
    K = max(1, int(round(k_frac * n)))
    return inst, K, agg


VAL_FRACS = (0.02, 0.05, 0.10, 0.15)


def sample_val_instances(cfg: TrainConfig, n_inst: int):
    """固定的验证实例集（种子在验证区、且与训练/测试互斥）。

    预算比例循环覆盖真实实验的 {2%,5%,10%,15%}；目标与情景族循环覆盖。
    """
    out = []
    rng = np.random.default_rng(20260101)
    for i in range(n_inst):
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
        n_scen = int(rng.choice(cfg.n_scen_choices))
        fam = ["A", "B", "C"][i % 3]
        seed = cfg.val_seed_base + i
        inst = make_instance(family=fam, n=n, n_scen=n_scen, seed=seed,
                             conflict=float(rng.uniform(*cfg.conflict_range)),
                             zipf_a=float(rng.uniform(*cfg.zipf_range)),
                             with_weights=cfg.use_weights)
        agg = ["min", "min", "cvar", "min"][i % 4] if i >= 4 else "min"
        frac = VAL_FRACS[i % len(VAL_FRACS)]
        K = max(1, int(round(frac * n)))
        out.append((inst, K, agg))
    return out


# =====================================================================
def run_episode(model, env: HUSAREnv, mode: str = "sample", temperature: float = 1.0,
                gen: torch.Generator | None = None, need_logprob: bool = True,
                cache=None, grads: bool = False) -> dict:
    """grads=False（默认）时在 torch.no_grad 下运行——**推理与实验必须用默认值**，
    否则每一步都会保留计算图（在 n=50,000、K=5,000 的实例上会造成显存爆炸）。
    训练时显式传入 grads=True。"""
    if not grads:
        with torch.no_grad():
            return _run_episode_impl(model, env, mode, temperature, gen, need_logprob,
                                     cache)
    return _run_episode_impl(model, env, mode, temperature, gen, need_logprob, cache)


_SAMPLE_SEED_BASE = 20261016        # 采样解码的确定性种子序列（保证可复现）
_sample_counter = 0


def _next_generator(device) -> torch.Generator:
    """按固定基数递增的采样发生器：同一次运行内的第 k 次采样调用有确定种子。"""
    global _sample_counter
    g = torch.Generator(device=device)
    g.manual_seed(_SAMPLE_SEED_BASE + _sample_counter)
    _sample_counter += 1
    return g


def _run_episode_impl(model, env: HUSAREnv, mode: str = "sample", temperature: float = 1.0,
                      gen: torch.Generator | None = None, need_logprob: bool = True,
                      cache=None) -> dict:
    """跑一个 episode；返回轨迹收益、log-prob 和、熵、以及 value（若需要）。

    cache 可复用同一实例的静态编码（例如 rollout 基线复用采样 episode 的缓存），
    避免重复计算最贵的情景/层级静态编码。
    """
    env.reset()
    T = env.T
    if gen is None:
        gen = _next_generator(env.device)
    if cache is None:
        cache = model.precompute(T)
    logps: list[torch.Tensor] = []
    ents: list[torch.Tensor] = []
    v0 = None
    while not env.done:
        logits, v, _ = model.logits(cache, T, env)
        if v0 is None:
            v0 = v
        mask = env.action_mask()
        logits = logits.masked_fill(~mask, float("-inf"))
        if mode == "sample":
            logits = logits / max(temperature, 1e-6)
        logp = torch.log_softmax(logits, dim=0)
        if mode == "sample":
            i = int(torch.multinomial(torch.exp(logp), 1, generator=gen).item())
        else:
            i = int(torch.argmax(logp).item())
        if need_logprob:
            logps.append(logp[i])
            p = torch.exp(logp)
            ents.append(-(p * torch.log(torch.clamp(p, min=1e-12))).sum())
        env.step(i)
    G = env.total_reward()
    return {"G": G, "logp_sum": torch.stack(logps).sum() if logps else None,
            "entropy": torch.stack(ents).mean() if ents else None, "v0": v0,
            "selection": env.selection(), "objective": env.current_objective(),
            "cache": cache}


# =====================================================================
@torch.no_grad()
def validate(model, cfg: TrainConfig, val_set, with_reference: bool = True) -> dict:
    """固定验证实例上的贪心解码评估。

    报告两项：
      * val_gain_mean  = J(S_K) − J(∅)（绝对增益，受覆盖项主导，判别力弱）；
      * val_ratio_vs_lcb = J_model / J_lcb 的均值（以 LCB 贪心解为参照的**相对质量**，
        作为检查点选择的主指标，判别力显著更强）。
    """
    model.eval()
    rows = []
    for (inst, K, agg) in val_set:
        env = HUSAREnv(inst, K, agg=agg, tau=cfg.tau, device=cfg.device)
        T = env.T
        cache = model.precompute(T)
        env.reset()
        while not env.done:
            logits, _, _ = model.logits(cache, T, env)
            mask = env.action_mask()
            logits = logits.masked_fill(~mask, float("-inf"))
            env.step(int(torch.argmax(logits).item()))
        J = env.current_objective()
        st = HUSARState(inst, [])
        rec = {"J": J, "J0": st.objective(agg, cfg.tau), "K": K, "agg": agg,
               "family": inst.meta.get("family"), "n": inst.n}
        if with_reference:
            from baseline_suite import run_baseline
            rec["J_lcb"] = run_baseline("celf_lcb", inst, K, agg=agg)["J_objective"]
        rows.append(rec)
    Js = np.array([r["J"] - r["J0"] for r in rows])
    ratios = [r["J"] / r["J_lcb"] for r in rows if r.get("J_lcb")]
    out = {"val_gain_mean": float(Js.mean()), "val_gain_median": float(np.median(Js)),
           "n_val": len(rows), "rows": rows}
    if ratios:
        out["val_ratio_vs_lcb"] = float(np.mean(ratios))
        out["val_ratio_worst"] = float(np.min(ratios))
    return out


# =====================================================================
def train(cfg: TrainConfig, model=None, log_fn=print):
    from model import HUSARNet, HUSARNetConfig

    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu"
                          else "cpu")
    cfg.device = str(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if model is None:
        model = HUSARNet(HUSARNetConfig())
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 12345)
    rng = np.random.default_rng(cfg.seed)

    val_set = sample_val_instances(cfg, cfg.val_instances)
    history = []
    # 检查点选择主指标：验证集上相对 LCB 贪心的解质量比（单调、判别力强）；
    # 该指标不可用时退回绝对增益。
    best = {"score": -np.inf, "iter": -1}
    t_start = time.time()
    n_param = sum(p.numel() for p in model.parameters())

    for it in range(1, cfg.iters + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        # --- 采样 M 个实例（梯度累积；基线必须在**同一批实例**上评价）
        batch = [sample_train_instance(cfg, rng, (it - 1) * cfg.instances_per_update + m)
                 for m in range(cfg.instances_per_update)]
        logps, Glist, ent_sum, v0s, caches = [], [], None, [], []
        envs = []
        for (inst, K, agg) in batch:
            env = HUSAREnv(inst, K, agg=agg, tau=cfg.tau, device=device)
            envs.append(env)
            out = run_episode(model, env, mode="sample", gen=gen, grads=True)
            caches.append(out["cache"])
            logps.append(out["logp_sum"])
            Glist.append(out["G"])
            if out["v0"] is not None:
                v0s.append(out["v0"])
            ent_sum = out["entropy"] if ent_sum is None else ent_sum + out["entropy"]
        if cfg.reward_scale == "per_k":
            Ks = torch.tensor([max(1, b[1]) for b in batch], dtype=torch.float32,
                              device=device)
        else:
            Ks = torch.ones(len(batch), dtype=torch.float32, device=device)
        G_t = torch.tensor(Glist, dtype=torch.float32, device=device) / Ks
        lp_t = torch.stack(logps)

        # --- 基线（同一批实例；rollout 复用采样 episode 的静态缓存）
        value_loss = None
        if cfg.baseline == "rollout":
            with torch.no_grad():
                bs = []
                for env, cache in zip(envs, caches):
                    o = run_episode(model, env, mode="greedy", need_logprob=False,
                                    cache=cache)
                    bs.append(o["G"])
                b_t = torch.tensor(bs, dtype=torch.float32, device=device) / Ks
        elif cfg.baseline == "mean":
            b_t = G_t.mean().expand_as(G_t)
        elif cfg.baseline == "value":
            v0_t = torch.stack(v0s)
            b_t = v0_t.detach()
            value_loss = nn.functional.mse_loss(v0_t, G_t.detach())
        elif cfg.baseline == "none":
            b_t = torch.zeros_like(G_t)
        else:
            raise ValueError(cfg.baseline)

        adv = G_t - b_t
        if cfg.advantage_norm and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = -(adv.detach() * lp_t).mean()
        if cfg.entropy_coef > 0:
            loss = loss - cfg.entropy_coef * (ent_sum / cfg.instances_per_update)
        if value_loss is not None:
            loss = loss + cfg.value_coef * value_loss

        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # --- 日志与验证
        if it % cfg.log_every == 0 or it == 1:
            ent_val = (float(ent_sum.item()) / cfg.instances_per_update
                       if torch.is_tensor(ent_sum) else None)
            rec = {"iter": it, "G_mean": float(np.mean(Glist)), "G_std": float(np.std(Glist)),
                   "loss": float(loss.item()), "grad_norm": float(gn),
                   "entropy": ent_val, "elapsed": time.time() - t_start}
            history.append(rec)
            log_fn(f"[{it:>4}/{cfg.iters}] G={rec['G_mean']:.3f}±{rec['G_std']:.3f} "
                   f"loss={rec['loss']:.4f} |g|={rec['grad_norm']:.3f} "
                   f"ent={ent_val:.3f} t={rec['elapsed']:.0f}s" if ent_val is not None
                   else f"[{it:>4}/{cfg.iters}] G={rec['G_mean']:.3f}±{rec['G_std']:.3f} "
                        f"loss={rec['loss']:.4f} |g|={rec['grad_norm']:.3f} "
                        f"t={rec['elapsed']:.0f}s")
        if it % cfg.val_every == 0 or it == cfg.iters:
            vm = validate(model, cfg, val_set)
            score = vm.get("val_ratio_vs_lcb", vm["val_gain_mean"])
            log_fn(f"    VAL iter={it} gain_mean={vm['val_gain_mean']:.4f} "
                   f"ratio_vs_lcb={vm.get('val_ratio_vs_lcb', float('nan')):.4f}")
            history.append({"iter": it, "val_gain_mean": vm["val_gain_mean"],
                            "val_ratio_vs_lcb": vm.get("val_ratio_vs_lcb"),
                            "val_ratio_worst": vm.get("val_ratio_worst")})
            if score > best["score"]:
                best = {"score": float(score), "val_gain_mean": vm["val_gain_mean"],
                        "val_ratio_vs_lcb": vm.get("val_ratio_vs_lcb"), "iter": it}
                if cfg.out_dir:
                    torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                                "net_cfg": asdict(model.cfg),
                                "iter": it, "val": json.loads(json.dumps(vm))}, os.path.join(cfg.out_dir,
                                                                     "best.pt"))
        if cfg.out_dir and (it % (cfg.val_every * 4) == 0):
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "net_cfg": asdict(model.cfg), "iter": it},
                       os.path.join(cfg.out_dir, "last.pt"))

    summary = {"n_param": n_param, "best": best, "elapsed": time.time() - t_start,
               "history": history, "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "peak_vram_MB": (torch.cuda.max_memory_allocated() / 1e6
                                if torch.cuda.is_available() else None)}
    if cfg.out_dir:
        with open(os.path.join(cfg.out_dir, "train_summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as fh:
            fh.write(cfg.to_json())
    return model, summary
