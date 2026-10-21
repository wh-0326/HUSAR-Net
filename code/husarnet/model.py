# -*- coding: utf-8 -*-
"""
HUSAR-Net 网络结构（Hierarchical Uncertainty-Aware Spatial Allocation Network）。

四个模块与论文结构一一对应
--------------------------
(A) EnterpriseRiskEncoder
    由企业风险分的情景分布（均值/标准差/分位/极值）与层级静态属性编码企业表示。
(B) ScenarioSetEncoder（跨企业联合情景编码）
    对每个企业，其 B 维 Bootstrap 情景剖面被视为一个**集合**：元素 b 的特征为
    [原始风险分(实例尺度标准化), 情景内跨企业 z 分数]，经共享 MLP 与**可学习查询 token 的
    注意力池化**（置换不变、B 可变），表示企业在情景间的联合波动结构，而非仅均值/标准差；
    另有一条对情景剖面的低秩投影分支（共享基底，跨企业可比）。
(C) HierarchicalSpatialEncoder
    按企业→城市→省份隶属关系聚合当前**未选**企业表示（增量维护），并把辖区饱和度、
    阈值 b_g、区域权重 a_g、剩余候选规模与未饱和城市比例回传至企业表示。
    该层级是**行政区嵌套**，不是地理距离传播。
(D) SequentialPolicyDecoder
    输出合法动作的掩码 softmax 分布 π_θ(a_t | s_t)。
(E) Feasibility & cache
    可行性由环境的严格掩码与 objective.HUSARState 增量缓存保证；网络不含额外配额。

消融开关（E3）
-------------
    use_scenario_encoder : N5（关闭联合情景编码）
    use_hierarchy        : N6（关闭层级状态反馈）
    use_city / use_prov  : N1 / N2（单尺度 vs 两尺度）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from environment import DVEC_DYN, GVEC_DYN

D_STATIC = 18          # 企业静态特征维数（7 风险分布 + 11 层级空间与相对位置）


# =====================================================================
@dataclass
class HUSARNetConfig:
    d_model: int = 64
    d_scen_hidden: int = 32
    n_scen_queries: int = 8
    d_hier: int = 32
    d_mlp: int = 128
    n_mlp_layers: int = 2
    use_scenario_encoder: bool = True
    use_scenario_projection: bool = True
    d_scen_proj: int = 16
    use_hierarchy: bool = True
    use_city: bool = True
    use_prov: bool = True
    chunk: int = 4096                 # 情景编码分块（内存控制，不改变数学）

    def tag(self) -> str:
        return (f"d{self.d_model}_q{self.n_scen_queries}_scen{int(self.use_scenario_encoder)}"
                f"_proj{int(self.use_scenario_projection)}_hier{int(self.use_hierarchy)}"
                f"_city{int(self.use_city)}_prov{int(self.use_prov)}")


# =====================================================================
class ScenarioSetEncoder(nn.Module):
    """集合式情景编码器（B 可变、元素置换不变）。输出 (n, d_model)。"""

    def __init__(self, cfg: HUSARNetConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_scen_hidden
        self.elem = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, d))
        self.queries = nn.Parameter(torch.randn(cfg.n_scen_queries, d) * (d ** -0.5))
        self.out = nn.Sequential(nn.Linear(cfg.n_scen_queries * d, cfg.d_model),
                                 nn.ReLU(), nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, z: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        B, n = raw.shape
        outs = []
        for s in range(0, n, self.cfg.chunk):
            e = min(n, s + self.cfg.chunk)
            feats = torch.stack([raw[:, s:e], z[:, s:e]], dim=-1)      # (B, c, 2)
            h = self.elem(feats)                                       # (B, c, d)
            score = torch.einsum("kd,bcd->kbc", self.queries, h)       # (K, B, c)
            w = torch.softmax(score, dim=1)
            ctx = torch.einsum("kbc,bcd->kcd", w, h)                   # (K, c, d)
            ctx = ctx.permute(1, 0, 2).reshape(e - s, -1)
            outs.append(self.out(ctx))
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


class ScenarioProjection(nn.Module):
    """情景剖面的低秩投影（共享基底的线性因子模型）。输出 (n, d_scen_proj)。"""

    def __init__(self, cfg: HUSARNetConfig):
        super().__init__()
        self.proj = nn.Linear(1, cfg.d_scen_proj, bias=False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.transpose(0, 1).unsqueeze(-1)          # (n, B, 1)
        return self.proj(x).mean(dim=1)                # (n, d)


# =====================================================================
class StaticCache:
    """一个实例的静态缓存（encode 一次，K 步复用；保持计算图以便整体反传）。"""

    __slots__ = ("h", "static", "city_cnt", "prov_of_city", "prov_cnt", "prov_firms",
                 "city_sum_all", "n_city", "n_prov", "scratch_city", "scratch_prov",
                 "scratch_unsat")

    def __init__(self, h, static, city_cnt, prov_of_city, prov_cnt, prov_firms,
                 city_sum_all, n_city, n_prov):
        self.h = h
        self.static = static
        self.city_cnt = city_cnt
        self.prov_of_city = prov_of_city
        self.prov_cnt = prov_cnt
        self.prov_firms = prov_firms
        self.city_sum_all = city_sum_all
        self.n_city = n_city
        self.n_prov = n_prov
        self.scratch_city = torch.zeros(n_city, h.shape[1], dtype=h.dtype, device=h.device)
        self.scratch_prov = torch.zeros(n_prov, h.shape[1], dtype=h.dtype, device=h.device)
        self.scratch_unsat = torch.zeros(n_prov, dtype=h.dtype, device=h.device)


class HUSARNet(nn.Module):
    def __init__(self, cfg: HUSARNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or HUSARNetConfig()
        c = self.cfg
        d_in = D_STATIC + (c.d_model if c.use_scenario_encoder else 0) \
            + (c.d_scen_proj if c.use_scenario_projection else 0)
        self.scen_enc = ScenarioSetEncoder(c) if c.use_scenario_encoder else None
        self.scen_proj = ScenarioProjection(c) if c.use_scenario_projection else None
        layers: list[nn.Module] = [nn.Linear(d_in, c.d_mlp), nn.ReLU()]
        for _ in range(c.n_mlp_layers - 1):
            layers += [nn.Linear(c.d_mlp, c.d_mlp), nn.ReLU()]
        layers += [nn.Linear(c.d_mlp, c.d_model)]
        self.enterprise = nn.Sequential(*layers)

        d_ctx = c.d_hier * int(c.use_city) + c.d_hier * int(c.use_prov)
        if c.use_hierarchy:
            self.city_mlp = (nn.Sequential(nn.Linear(c.d_model + 5, c.d_hier), nn.ReLU(),
                                           nn.Linear(c.d_hier, c.d_hier))
                             if c.use_city else None)
            self.prov_mlp = (nn.Sequential(
                nn.Linear((c.d_hier if c.use_city else c.d_model) + 4, c.d_hier), nn.ReLU(),
                nn.Linear(c.d_hier, c.d_hier)) if c.use_prov else None)
            self.hier_norm = nn.LayerNorm(d_ctx) if d_ctx > 0 else None
        else:
            self.city_mlp = self.prov_mlp = self.hier_norm = None
            d_ctx = 0

        self.policy = nn.Sequential(
            nn.Linear(c.d_model + d_ctx + DVEC_DYN, c.d_mlp), nn.ReLU(),
            nn.Linear(c.d_mlp, max(16, c.d_mlp // 2)), nn.ReLU(),
            nn.Linear(max(16, c.d_mlp // 2), 1))
        self.value = nn.Sequential(
            nn.Linear(c.d_model + GVEC_DYN, c.d_mlp), nn.ReLU(),
            nn.Linear(c.d_mlp, 1))

    # ------------------------------------------------------------------ 静态
    def precompute(self, T) -> StaticCache:
        parts = [T.static]
        if self.scen_enc is not None:
            parts.append(self.scen_enc(T.z_t, T.raw_t))
        if self.scen_proj is not None:
            parts.append(self.scen_proj(T.raw_t))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        h = self.enterprise(x)
        city_cnt = torch.bincount(T.city_t, minlength=T.inst.n_city).float()
        # 城市 -> 省份 的映射（由企业的省市归属推出；严格嵌套下唯一）
        prov_of_city = torch.zeros(T.inst.n_city, dtype=torch.long, device=h.device)
        prov_of_city[T.city_t] = T.prov_t
        prov_cnt = torch.bincount(prov_of_city, minlength=T.inst.n_prov).float()
        prov_firms = torch.zeros(T.inst.n_prov, dtype=h.dtype, device=h.device)
        prov_firms = prov_firms.index_add(0, T.prov_t,
                                          torch.ones(T.inst.n, dtype=h.dtype,
                                                     device=h.device))
        # 城市层企业表示之和（本 episode 内 h 不变，预计算）
        city_sum_all = torch.zeros(T.inst.n_city, h.shape[1], dtype=h.dtype, device=h.device)
        city_sum_all = city_sum_all.index_add(0, T.city_t, h)
        return StaticCache(h, T.static, city_cnt, prov_of_city, prov_cnt, prov_firms,
                           city_sum_all, T.inst.n_city, T.inst.n_prov)

    # ------------------------------------------------------------------ 层级
    def hierarchical_context(self, cache: StaticCache, T, selected: torch.Tensor,
                             n_sel_city: torch.Tensor, n_sel_prov: torch.Tensor,
                             ctx_out: torch.Tensor | None = None):
        c = self.cfg
        if not c.use_hierarchy:
            return None
        h = cache.h
        city_t, prov_t = T.city_t, T.prov_t
        # 已选企业的城市/省份表示和（精确增量，O(K·d)）
        sel_sum_city = torch.zeros(cache.n_city, h.shape[1], dtype=h.dtype,
                                   device=h.device)
        if selected.any():
            sel_sum_city = sel_sum_city.index_add(0, city_t[selected], h[selected])
        left_city = torch.clamp(cache.city_cnt - n_sel_city, min=0.0)
        city_sum_left = cache.city_sum_all - sel_sum_city
        city_emb = city_sum_left / torch.clamp(left_city, min=1.0).unsqueeze(1)
        sat = torch.clamp(n_sel_city / T.b_city_t, max=1.0).unsqueeze(1)
        cf = torch.cat([city_emb, sat, (T.a_city_t / 4.0).unsqueeze(1),
                        (T.b_city_t / 10.0).unsqueeze(1),
                        (left_city / torch.clamp(cache.city_cnt, min=1.0)).unsqueeze(1),
                        (n_sel_city / 10.0).unsqueeze(1)], dim=1)
        city_ctx = self.city_mlp(cf) if c.use_city else None
        out = []
        if c.use_city:
            out.append(city_ctx[city_t])
        if c.use_prov:
            base = city_ctx if c.use_city else city_emb
            prov_sum = torch.zeros(cache.n_prov, base.shape[1], dtype=base.dtype,
                                   device=base.device)
            prov_sum = prov_sum.index_add(0, cache.prov_of_city, base)
            prov_emb = prov_sum / torch.clamp(cache.prov_cnt, min=1.0).unsqueeze(1)
            psat = torch.clamp(n_sel_prov / T.b_prov_t, max=1.0).unsqueeze(1)
            # 未饱和城市比例（辖区层级的动态反馈）
            city_sat_flag = (n_sel_city >= T.b_city_t).to(base.dtype)
            unsat = torch.zeros(cache.n_prov, dtype=base.dtype, device=base.device)
            unsat = unsat.index_add(0, cache.prov_of_city, 1.0 - city_sat_flag)
            pf = torch.cat([prov_emb, psat, (T.a_prov_t / 4.0).unsqueeze(1),
                            (n_sel_prov / 10.0).unsqueeze(1),
                            (unsat / torch.clamp(cache.prov_cnt, min=1.0)).unsqueeze(1)],
                           dim=1)
            prov_ctx = self.prov_mlp(pf)
            out.append(prov_ctx[prov_t])
        res = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        return self.hier_norm(res) if self.hier_norm is not None else res

    # ------------------------------------------------------------------ 策略
    def logits(self, cache: StaticCache, T, env) -> tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor]:
        """返回 (logits (n,), value (), ctx (n,d) 或 None)。"""
        d = env.dynamic_features()
        ctx = self.hierarchical_context(cache, T, env.selected, env.n_sel_city,
                                        env.n_sel_prov)
        parts = [cache.h] if ctx is None else [cache.h, ctx]
        parts.append(d["firm_dyn"])
        logits = self.policy(torch.cat(parts, dim=1)).squeeze(-1)
        v = self.value(torch.cat([cache.h.mean(dim=0), d["global_dyn"]])).squeeze(-1)
        return logits, v, ctx
