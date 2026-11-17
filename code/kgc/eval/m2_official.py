"""官方 NUS M² scorer 的逐句统计封装（返修主指标）。

原 m2score.py 是句级 F0.5 再平均：假设与参考皆无编辑的句子记 1 分，
因而奖励"不改"。返修改用官方 M² scorer 的语料级口径：
- 编辑抽取与匹配完全复用 vendored 的 levenshtein.py（NUS m2scorer 的 py3 移植），
  不做任何改写；
- 多参考：逐句在各 annotator 中选使**累计** F 最大者（官方贪心规则，含平局规则）；
- 单参考：每个 annotator 单独累加，再对 annotator 取平均。

为支持作文聚类 bootstrap，先缓存逐句、逐 annotator 的 (correct, proposed, gold)，
重抽样时只在缓存上重跑官方的贪心选择，无需重新对齐。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from kgc.data.m2_io import M2Sentence

_VENDOR = (Path(__file__).resolve().parents[2]
           / "vendor/Standard_Korean_GEC/metric/m2scorer/scripts")
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
import levenshtein as _lev  # noqa: E402

Stats = Tuple[int, int, int]          # (correct, proposed, gold)


def gold_by_annotator(s: M2Sentence) -> Dict[int, list]:
    """转成官方 scorer 的 gold 结构：{ann: [(start, end, original, [corr, ...])]}。

    与官方 load_annotation 一致：noop 只保留 annotator 键、不产生编辑；
    -NONE- 表示删除，即空串。
    """
    toks = s.source.split()
    out: Dict[int, list] = {}
    for e in s.edits:
        out.setdefault(e.annotator, [])
        if e.is_noop:
            continue
        corrs = ["" if c == "-NONE-" else c.strip() for c in e.corrs]
        out[e.annotator].append((e.start, e.end, " ".join(toks[e.start:e.end]), corrs))
    return out or {0: []}


def sentence_stats(s: M2Sentence, hyp: str, max_unchanged_words: int = 2
                   ) -> Dict[int, Stats]:
    """单句、逐 annotator 的官方统计量。图构造与官方 batch_multi_pre_rec_f1 相同。"""
    src_tok, hyp_tok = s.source.split(), hyp.split()
    lm1, bp1 = _lev.levenshtein_matrix(src_tok, hyp_tok, 1, 1, 1)
    lm2, bp2 = _lev.levenshtein_matrix(src_tok, hyp_tok, 1, 1, 2)
    V1, E1, d1, ed1 = _lev.edit_graph(lm1, bp1)
    V2, E2, d2, ed2 = _lev.edit_graph(lm2, bp2)
    V, E, dist, edits = _lev.merge_graph(V1, V2, E1, E2, d1, d2, ed1, ed2)
    V, E, dist, edits = _lev.transitive_arcs(V, E, dist, edits, max_unchanged_words, False)
    out: Dict[int, Stats] = {}
    for ann, gold in gold_by_annotator(s).items():
        local = _lev.set_weights(E, dist, edits, gold, False, False)
        seq = _lev.best_edit_seq_bf(V, E, local, edits, False)
        correct = _lev.matchSeq(seq, gold, False, False)
        out[ann] = (len(correct), len(seq), len(gold))
    return out


def _prf(c: float, p: float, g: float, beta: float = 0.5) -> Tuple[float, float, float]:
    prec = c / p if p else 1.0
    rec = c / g if g else 1.0
    den = beta * beta * prec + rec
    f = (1 + beta * beta) * prec * rec / den if den else 0.0
    return prec, rec, f


def corpus_multi(stats: Sequence[Dict[int, Stats]], beta: float = 0.5,
                 order: Optional[Sequence[int]] = None):
    """官方多参考语料级 P/R/F。order 为句子下标序列（bootstrap 重抽样时传入，可重复）。

    返回 (P, R, F, chosen)；chosen[k] 为第 k 个位置所选 annotator。
    """
    idx = range(len(stats)) if order is None else order
    sq = beta * beta
    C = P = G = 0.0
    chosen = []
    for i in idx:
        best = None
        for ann, (c, p, g) in stats[i].items():
            cl, pl, gl = C + c, P + p, G + g
            f = _prf(cl, pl, gl, beta)[2]
            key = (f, cl, -(pl + sq * gl))
            if best is None or key > best[0]:
                best = (key, ann, (c, p, g))
        _, ann, (c, p, g) = best
        C, P, G = C + c, P + p, G + g
        chosen.append(ann)
    return (*_prf(C, P, G, beta), chosen)


def corpus_single(stats: Sequence[Dict[int, Stats]], ann: int, beta: float = 0.5,
                  order: Optional[Sequence[int]] = None):
    """单参考语料级：只用 annotator `ann`（某句缺该 annotator 时按空参考计）。"""
    idx = range(len(stats)) if order is None else order
    C = P = G = 0.0
    for i in idx:
        c, p, g = stats[i].get(ann, (0, _proposed(stats[i]), 0))
        C, P, G = C + c, P + p, G + g
    return _prf(C, P, G, beta)


def _proposed(st: Dict[int, Stats]) -> int:
    return max(p for _c, p, _g in st.values()) if st else 0


def corpus_stats_for_runs(m2: List[M2Sentence], hyps: Sequence[str]
                          ) -> List[Dict[int, Stats]]:
    return [sentence_stats(s, h) for s, h in zip(m2, hyps)]
