# -*- coding: utf-8 -*-
"""
HUSAR-Net 评价与统计（论文表格与统计检验的唯一来源）。

纪律
----
本模块是唯一允许接触目标年真实标签的算法侧模块，且标签只用于**评价**：
`recall_at_k` / `hits` / 覆盖率等回溯指标。禁止把本节任何量回传给选择或训练。

内容
----
  * evaluate_selection : 统一指标（调用 objective.all_metrics）
  * paired_compare     : 同一实例×预算/种子的配对比较（Wilcoxon、均值/中位数差、效应量）
  * bootstrap_ci       : 均值的 bootstrap 置信区间（固定种子）
  * summarize_table    : 生成可直接写入论文的表格行
"""
from __future__ import annotations

import numpy as np

from objective import all_metrics


# =====================================================================
def evaluate_selection(inst, selection, agg: str = "min", tau: float = 0.25,
                       labels=None, K: int | None = None) -> dict:
    """统一评价出口（标签只在此处进入，且只用于回溯指标）。"""
    return all_metrics(inst, selection, agg=agg, tau=tau, labels=labels, K=K)


# =====================================================================
def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 20261016) -> tuple[float, float, float]:
    """均值的 bootstrap 置信区间。返回 (mean, lo, hi)。"""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def wilcoxon_p(a, b) -> float | None:
    """配对 Wilcoxon 符号秩检验 p 值（样本过少或全零差时返回 None）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    if d.size < 5 or np.allclose(d, 0):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return None


def paired_compare(rows: list[dict], key_a: str, key_b: str, metric: str = "J_min",
                   rel: bool = True, ref_key: str = "ref") -> dict:
    """在同一批 (实例 × 预算/种子) 记录上做配对比较。

    rows 中每条记录含 metric 与 ref_key（参照值，用于计算相对差）。
    返回：实例数、平均绝对差、平均相对差(%)、a 优于 b 的计数、Wilcoxon p、Cohen's d。
    """
    a = np.array([r[key_a][metric] for r in rows if key_a in r], dtype=float)
    b = np.array([r[key_b][metric] for r in rows if key_b in r], dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b
    rel_d = np.array([(r[key_a][metric] - r[key_b][metric]) / abs(r["ref"])
                      for r in rows[:n]], dtype=float) * 100 if rel else d
    sd = float(np.std(d, ddof=0))
    return {"n": int(n), "mean_abs_diff": float(d.mean()),
            "mean_rel_diff_pct": float(rel_d.mean()), "win": int((d > 0).sum()),
            "lose": int((d < 0).sum()), "tie": int((d == 0).sum()),
            "wilcoxon_p": wilcoxon_p(a, b),
            "cohens_d": float(d.mean() / sd) if sd > 0 else 0.0}


# =====================================================================
def summarize_table(records: list[dict], group_keys=("method",),
                    metric_keys=("J_min", "J_cvar", "recall_at_k", "cities_covered",
                                 "provs_covered", "coverage_utility", "n_network_calls",
                                 "time", "constraint_violations")) -> list[dict]:
    """把多条运行记录汇总为论文表格行（按 group_keys 分组，报告均值/标准差/计数）。"""
    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        return []
    keys = [k for k in group_keys if k in df.columns]
    mkeys = [k for k in metric_keys if k in df.columns]
    g = df.groupby(keys, dropna=False)
    out = []
    for name, sub in g:
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row["n_runs"] = len(sub)
        for k in mkeys:
            v = sub[k].astype(float)
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_std"] = float(v.std(ddof=0))
        out.append(row)
    return out
