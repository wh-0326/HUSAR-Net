# -*- coding: utf-8 -*-
"""Fast inference decoder for HUSAR-Net (same policy, same maths, lean loop).

Why this file exists
--------------------
The reference decoder (`inference.decode`) rebuilds several temporaries every step and
records reward book-keeping that is only needed during training:

  * `HUSAREnv.step` evaluates the objective twice per step (training reward) - not needed
    at inference time, the final objective is computed once at the end;
  * `sel_sum_city` is rebuilt from scratch with a full `index_add` over `n` rows;
  * `selected/n_sel_city/n_sel_prov` are cloned (out-of-place) every step;
  * every step allocates fresh tensors and synchronises (`argmax(...).item()`), which
    serialises CPU and GPU.

This decoder keeps the identical policy and state definition, but
  1. keeps all episode state in pre-allocated GPU buffers and updates it in place;
  2. maintains the per-city representation sum incrementally (O(d) per step instead of
     O(n_city*d)) and skips the training-only reward book-keeping;
  3. never synchronises inside the loop - the chosen index stays on the device and the
     state update is expressed as tensor ops;
  4. optionally emits `block` actions per forward pass (a quality/speed knob).

The numerical output is checked against the reference decoder in
`scripts/verify_fast_decode.py` (selection and objective must match).
"""
from __future__ import annotations

import time

import torch

DVEC_DYN = 10
GVEC_DYN = 18


class FastDecoder:
    def __init__(self, model, T, K: int, agg: str = "min", tau: float = 0.25,
                 device=None):
        self.model = model
        self.T = T
        self.K = int(K)
        self.agg = agg
        self.tau = float(tau)
        self.device = torch.device(device) if device is not None else T.static.device
        model.eval()
        self.cache = model.precompute(T)
        self.h = self.cache.h
        self.d = self.h.shape[1]
        self.n = T.inst.n
        self.n_city = T.inst.n_city
        self.n_prov = T.inst.n_prov
        self.B = T.inst.B
        # static per-firm tensors used every step
        self.city_t = T.city_t
        self.prov_t = T.prov_t
        self.Q_t = T.Q_t
        # episode state (allocated once)
        self.selected = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.n_sel_city = torch.zeros(self.n_city, dtype=torch.float32, device=self.device)
        self.n_sel_prov = torch.zeros(self.n_prov, dtype=torch.float32, device=self.device)
        self.sel_sum_city = torch.zeros(self.n_city, self.d, dtype=self.h.dtype,
                                        device=self.device)
        self.scen_sums = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.members: list[int] = []
        self._one = torch.ones(1, dtype=torch.float32, device=self.device)
        # selection buffer kept on device: no host sync inside the loop
        self.sel_buf = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self._t = 0

    # ------------------------------------------------------------------ state
    def reset(self):
        self.selected.zero_()
        self.n_sel_city.zero_()
        self.n_sel_prov.zero_()
        self.sel_sum_city.zero_()
        self.scen_sums.zero_()
        self.sel_buf.zero_()
        self._t = 0
        self.members = []

    # ------------------------------------------------------------------ one step
    def _logits(self):
        model, T, cfg = self.model, self.T, self.model.cfg
        n_sel_city, n_sel_prov = self.n_sel_city, self.n_sel_prov
        city_t, prov_t = self.city_t, self.prov_t

        # ---- firm-level dynamic features (10 dims, same order as env.dynamic_features)
        nsc = n_sel_city.index_select(0, city_t)
        nsp = n_sel_prov.index_select(0, prov_t)
        city_sat = torch.clamp(nsc / T.b_city_t[city_t], max=1.0)
        prov_sat = torch.clamp(nsp / T.b_prov_t[prov_t], max=1.0)
        n_left_city = T.nfirm_city_firm - nsc
        gc = (nsc < T.b_city_t[city_t]).to(torch.float32)
        gp = (nsp < T.b_prov_t[prov_t]).to(torch.float32)
        cov_gain = gc * T.gain_city_coef[city_t] + gp * T.gain_prov_coef[prov_t]
        bstar = torch.argmin(self.scen_sums)
        # index_select keeps the row lookup on device (advanced indexing would sync)
        q_bind = torch.index_select(self.Q_t, 0, bstar).squeeze(0) / T.scen_ref
        left_frac = n_left_city / torch.clamp(T.nfirm_city_firm, min=1.0)
        firm_dyn = torch.stack([self.selected.to(torch.float32), city_sat, prov_sat,
                                left_frac, cov_gain, q_bind, T.a_city_firm,
                                T.a_prov_firm, T.a_city_over_b_firm,
                                T.b_city_over_10_firm], dim=1)

        # ---- city / province context
        ctx = None
        if cfg.use_hierarchy:
            left = torch.clamp(self.cache.city_cnt - n_sel_city, min=0.0)
            city_sum_left = self.cache.city_sum_all - self.sel_sum_city
            city_emb = city_sum_left / torch.clamp(left, min=1.0).unsqueeze(1)
            sat = torch.clamp(n_sel_city / T.b_city_t, max=1.0).unsqueeze(1)
            cf = torch.cat([city_emb, sat, (T.a_city_t / 4.0).unsqueeze(1),
                            (T.b_city_t / 10.0).unsqueeze(1),
                            (left / torch.clamp(self.cache.city_cnt, min=1.0)).unsqueeze(1),
                            (n_sel_city / 10.0).unsqueeze(1)], dim=1)
            city_ctx = model.city_mlp(cf) if cfg.use_city else None
            out = []
            if cfg.use_city:
                out.append(city_ctx.index_select(0, city_t))
            if cfg.use_prov:
                base = city_ctx if cfg.use_city else city_emb
                prov_sum = torch.zeros(self.n_prov, base.shape[1], dtype=base.dtype,
                                       device=base.device)
                prov_sum = prov_sum.index_add(0, self.cache.prov_of_city, base)
                prov_emb = prov_sum / torch.clamp(self.cache.prov_cnt, min=1.0).unsqueeze(1)
                psat = torch.clamp(n_sel_prov / T.b_prov_t, max=1.0).unsqueeze(1)
                city_sat_flag = (n_sel_city >= T.b_city_t).to(base.dtype)
                unsat = torch.zeros(self.n_prov, dtype=base.dtype, device=base.device)
                unsat = unsat.index_add(0, self.cache.prov_of_city, 1.0 - city_sat_flag)
                pf = torch.cat([prov_emb, psat, (T.a_prov_t / 4.0).unsqueeze(1),
                                (n_sel_prov / 10.0).unsqueeze(1),
                                (unsat / torch.clamp(self.cache.prov_cnt,
                                                     min=1.0)).unsqueeze(1)], dim=1)
                prov_ctx = model.prov_mlp(pf)
                out.append(prov_ctx.index_select(0, prov_t))
            ctx = torch.cat(out, dim=1) if len(out) > 1 else out[0]
            if model.hier_norm is not None:
                ctx = model.hier_norm(ctx)

        parts = [self.h] if ctx is None else [self.h, ctx]
        parts.append(firm_dyn)
        logits = model.policy(torch.cat(parts, dim=1)).squeeze(-1)
        return logits

    def _apply(self, idx: torch.Tensor):
        """In-place state update for the chosen index/indices (device tensor)."""
        k = idx.numel()
        self.selected.index_fill_(0, idx, True)
        g = self.city_t.index_select(0, idx)
        gp = self.prov_t.index_select(0, idx)
        self.n_sel_city.index_add_(0, g, torch.ones(k, dtype=torch.float32,
                                                    device=self.device))
        self.n_sel_prov.index_add_(0, gp, torch.ones(k, dtype=torch.float32,
                                                     device=self.device))
        self.sel_sum_city.index_add_(0, g, self.h.index_select(0, idx))
        self.scen_sums.add_(self.Q_t.index_select(1, idx).sum(dim=1))
        self.sel_buf[self._t:self._t + k] = idx
        self._t += k

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, inst, n_samples: int = 16, temperature: float = 1.0,
               seed: int = 20261016, agg: str = "min", tau: float = 0.25) -> dict:
        """Stochastic decoding: `n_samples` trajectories, best one by the shared objective.

        Runs in eager mode (the RNG makes the whole loop non-capturable); each step still
        uses the lean state update, so the cost is n_samples x the deterministic loop.
        """
        from objective import husar_objective
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        best_sel, best_J, best_env = None, float("-inf"), None
        t0 = time.perf_counter()
        for _ in range(n_samples):
            self.reset()
            while self._t < self.K:
                logits = self._logits()
                logits = logits.masked_fill(self.selected, float("-inf")) / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs, 1, generator=gen)
                self._apply(idx)
            sel = [int(v) for v in self.sel_buf.tolist()]
            J = husar_objective(inst, sel, agg, tau)
            if J > best_J:
                best_J, best_sel, best_env = J, sel, (self.n_sel_city.clone(),
                                                      self.n_sel_prov.clone(),
                                                      self.scen_sums.clone())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.members = best_sel
        return {"selection": best_sel, "objective": best_J, "time": dt,
                "n_samples": n_samples, "n_network_calls": n_samples * self.K}

    # ------------------------------------------------------------------ decode
    def _run(self, block: int = 1):
        """One full episode, no timing / no synchronisation (graph-capturable)."""
        self.reset()
        while self._t < self.K:
            logits = self._logits()
            logits = logits.masked_fill(self.selected, float("-inf"))
            b = min(block, self.K - self._t)
            if b == 1:
                idx = torch.argmax(logits).reshape(1)
            else:
                idx = torch.topk(logits, b).indices
            self._apply(idx)

    @torch.no_grad()
    def greedy(self, block: int = 1, timed: bool = False) -> dict:
        t0 = time.perf_counter()
        self._run(block)
        if timed and self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.members = [int(v) for v in self.sel_buf.tolist()]
        return {"selection": self.members, "time": dt,
                "n_network_calls": (self.K + block - 1) // block, "block": block}

    # ------------------------------------------------------------------ cuda graph
    @torch.no_grad()
    def capture(self, block: int = 1, warmup: int = 3):
        """Capture the whole K-step decode as one CUDA graph (fixed K and block)."""
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        self.block = block
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._run(block)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run(block)
        return self

    @torch.no_grad()
    def replay(self) -> dict:
        t0 = time.perf_counter()
        self._graph.replay()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.members = [int(v) for v in self.sel_buf.tolist()]
        return {"selection": self.members, "time": dt,
                "n_network_calls": (self.K + self.block - 1) // self.block,
                "block": self.block, "graph": True}
