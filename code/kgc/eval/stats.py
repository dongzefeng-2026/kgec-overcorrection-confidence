"""作文聚类的重抽样统计（返修 R2.3）。

同一篇作文的句子不独立，因此所有区间估计和检验都以聚类（作文）为重抽样单位：
- cluster_bootstrap：有放回地抽聚类，拼接其句子下标后重算语料级统计量（M²、GLEU
  等语料级指标不能对句子分数取平均，必须传入整个下标序列重算）；
- paired_bootstrap：两个系统使用同一组重抽样，得到差值的区间；
- paired_randomization：聚类层面的近似随机化检验——每个聚类以 1/2 概率交换两系统
  的输出，p = (#{|d*| >= |d|} + 1) / (R + 1)；
- holm：Holm–Bonferroni 校正。

聚类单元由 kgc.data.essay_link 给出（已链接句 = 作文；未链接句 = .m2 中的连续段）。
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

Stat = Callable[[Sequence[int]], float]


def cluster_index(clusters: Sequence[str]) -> List[List[int]]:
    """按首次出现顺序把句子下标分组。"""
    order: Dict[str, List[int]] = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    return list(order.values())


def _resample(groups: List[List[int]], rng: random.Random) -> List[int]:
    idx: List[int] = []
    for _ in range(len(groups)):
        idx.extend(groups[rng.randrange(len(groups))])
    return idx


def cluster_bootstrap(stat: Stat, clusters: Sequence[str], n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """返回 (点估计, 下界, 上界)，百分位区间。"""
    groups = cluster_index(clusters)
    full = [i for g in groups for i in g]
    point = stat(full)
    rng = random.Random(seed)
    vals = sorted(stat(_resample(groups, rng)) for _ in range(n_boot))
    return point, vals[int(alpha / 2 * n_boot)], vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def paired_bootstrap(stat_a: Stat, stat_b: Stat, clusters: Sequence[str], n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """差值 a - b 的点估计与百分位区间；两系统共用每一次重抽样。"""
    groups = cluster_index(clusters)
    full = [i for g in groups for i in g]
    point = stat_a(full) - stat_b(full)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(groups, rng)
        diffs.append(stat_a(idx) - stat_b(idx))
    diffs.sort()
    return point, diffs[int(alpha / 2 * n_boot)], diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def paired_randomization(stat_pair: Callable[[Sequence[bool]], Tuple[float, float]],
                         clusters: Sequence[str], n_perm: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """聚类层面的配对近似随机化检验。

    stat_pair(swap) 接收逐句的交换标记（同一聚类内取值相同），返回交换后
    (系统A统计量, 系统B统计量)。返回 (观测差值, 双侧 p)。
    """
    groups = cluster_index(clusters)
    n = len(clusters)
    a, b = stat_pair([False] * n)
    obs = a - b
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        swap = [False] * n
        for g in groups:
            if rng.random() < 0.5:
                for i in g:
                    swap[i] = True
        a2, b2 = stat_pair(swap)
        if abs(a2 - b2) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni 校正后的 p 值（保持单调）。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
