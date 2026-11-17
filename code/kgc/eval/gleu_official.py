"""GLEU（Napoles et al. 2015/2016 多参考版），复用 vendored 官方实现。

与官方 run_gleu.py 相同：每次迭代为每句随机选一个参考（seed = j*101），
迭代 500 次取均值。为支持作文聚类 bootstrap，缓存逐句、逐参考的充分统计量。

参考句由 .m2 文件逐 annotator 应用编辑得到（noop annotator 的参考即源句）。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from kgc.data.m2_io import M2Sentence

_VENDOR = (Path(__file__).resolve().parents[2]
           / "vendor/Standard_Korean_GEC/metric")
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
from gleu import GLEU  # noqa: E402

N_ANN = 2


def apply_edits(s: M2Sentence, ann: int) -> str:
    """把某 annotator 的编辑应用到源句，得到该 annotator 的参考句。"""
    toks = s.source.split()
    es = sorted((e for e in s.edits if e.annotator == ann and not e.is_noop),
                key=lambda e: (e.start, e.end), reverse=True)
    for e in es:
        corr = e.corrs[0] if e.corrs else "-NONE-"
        toks[e.start:e.end] = [] if corr == "-NONE-" else corr.split()
    return " ".join(toks)


def references(m2: List[M2Sentence]) -> List[List[str]]:
    """[[ref_ann0, ref_ann1], ...]，按句。

    某句缺少某个 annotator（如 Kor-Lang8 的单参考句）时用该句已有的第一个参考填补，
    使随机选参考时不会把源句误当参考；KoLLA 每句都有两个 annotator，不受影响。
    """
    out = []
    for s in m2:
        anns = sorted(s.annotator_groups())
        out.append([apply_edits(s, a if a in anns else anns[0]) for a in range(N_ANN)])
    return out


class GleuCache:
    """逐句逐参考的 GLEU 充分统计量缓存。"""

    def __init__(self, m2: List[M2Sentence]):
        g = GLEU(4)
        g.order = 4
        g.all_s_ngrams = [[g.get_ngram_counts(s.source.split(), n) for n in range(1, 5)]
                          for s in m2]
        refs = references(m2)
        # 复刻 load_references 的内部状态（官方实现只接受文件路径）
        from collections import Counter
        g.refs = [[r.split() for r in rs] for rs in refs]
        g.rlens = [[len(r.split()) for r in rs] for rs in refs]
        g.all_rngrams_freq = [Counter() for _ in range(4)]
        g.all_r_ngrams = []
        for refset in g.refs:
            all_ngrams = []
            g.all_r_ngrams.append(all_ngrams)
            for n in range(1, 5):
                ngrams = g.get_ngram_counts(refset[0], n)
                all_ngrams.append(ngrams)
                for k in ngrams.keys():
                    g.all_rngrams_freq[n - 1][k] += 1
                for ref in refset[1:]:
                    new = g.get_ngram_counts(ref, n)
                    for nn in new.elements():
                        if new[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new[nn]
        self.g = g
        self.n = len(m2)

    def sentence_stats(self, hyps: Sequence[str]) -> List[List[List[int]]]:
        """stats[i][r] = 第 i 句对第 r 个参考的 10 维统计量。"""
        out = []
        for i, h in enumerate(hyps):
            self.g.load_hypothesis_sentence(h.split())
            out.append([list(self.g.gleu_stats(i, r_ind=r)) for r in range(N_ANN)])
        return out

    @staticmethod
    def corpus(stats: List[List[List[int]]], n_iter: int = 500,
               order: Optional[Sequence[int]] = None) -> float:
        idx = list(range(len(stats))) if order is None else list(order)
        total = 0.0
        g = GLEU(4)
        for j in range(n_iter):
            random.seed(j * 101)
            pick = [random.randint(0, N_ANN - 1) for _ in range(len(stats))]
            acc = [0] * 10
            for i in idx:
                st = stats[i][pick[i]]
                acc = [a + b for a, b in zip(acc, st)]
            total += g.gleu(acc)
        return total / n_iter


def _picks(n: int, n_iter: int = 500):
    """与官方 run_gleu.py 相同的随机参考选择（seed = j*101）。"""
    import numpy as np
    out = np.empty((n_iter, n), dtype=np.int64)
    for j in range(n_iter):
        random.seed(j * 101)
        out[j] = [random.randint(0, N_ANN - 1) for _ in range(n)]
    return out


def corpus_np(stats, picks, order: Optional[Sequence[int]] = None) -> float:
    """corpus() 的向量化等价实现：stats 形状 (n, N_ANN, 10)，picks 形状 (n_iter, n)。

    注意：官方实现对每次迭代 j 先按全部 n 句生成参考选择、再取 order 中的句子，
    这里同样按原句下标取 picks[:, i]，重复抽到的句子沿用同一选择。
    """
    import numpy as np
    idx = np.arange(stats.shape[0]) if order is None else np.asarray(order)
    acc = stats[idx[None, :], picks[:, idx]].sum(axis=1).astype(float)   # (n_iter, 10)
    c, r = acc[:, 0], acc[:, 1]
    num, den = acc[:, 2::2], acc[:, 3::2]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(num / den).sum(axis=1) / 4
        g = np.exp(np.minimum(0.0, 1.0 - r / c) + logp)
    g[(acc == 0).any(axis=1)] = 0.0
    return float(g.mean())
