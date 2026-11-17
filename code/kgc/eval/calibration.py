"""编辑级置信度、校准与选择性纠错。

置信度（本文方法）：
    conf(e) = |{samples 中含有编辑 e}| / N
即某条编辑在自洽采样中的复现频率。与序列 logprob 相比，它与 GEC 的编辑粒度对齐，
可直接作为"是否向学习者展示该反馈"的阈值。

术语纪律（关系到结论的可辩护性）：
- supported edit  ：与任一参考的编辑精确一致；
- unsupported edit：未与任何参考匹配。**不等于"错误"**——参考不完备时，
  合理的修正也可能 unsupported。故风险曲线一律称 unsupported rate，
  不称 error rate；仅在多参考设定下讨论其接近 false correction 的程度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from kgc.data.m2_io import M2Sentence
from kgc.eval.m2score import BETA, _runs, align_hyp_edits, merge_edits


@dataclass
class ScoredEdit:
    sent_id: int
    group: Optional[str]
    start: int
    end: int
    corr: str
    conf: float          # 自洽置信度 [0,1]
    supported: bool      # 是否被任一参考支持


def _canon(source: str, hyp: str) -> List[Tuple[int, int, str]]:
    """假设句 -> canonical 编辑列表 [(start, end, corr)]。"""
    items = merge_edits([(e.start, e.end, [" ".join(e.corr_tokens)])
                         for e in align_hyp_edits(source, hyp)])
    return [(s, e, c[0]) for s, e, c in items]


def _norm_corr(c: str) -> str:
    """校正串归一：-NONE-（删除）记为空串，空白归一。系统侧删除本就是空串。"""
    return "" if c == "-NONE-" else " ".join(c.split())


def _gold_edits(s: M2Sentence) -> List[Tuple[int, int, List[str]]]:
    """参考编辑：逐 annotator 取其相邻编辑的全部连续段（含单条编辑），再跨 annotator 取并集。

    返修更正：原实现先跨 annotator 合并同 span 编辑、再把相邻编辑拼接，会把两位标注者
    各自的编辑拼成谁都没做过的复合编辑，并吞掉原本的单条编辑（例：annotator 0 在 (1,1)
    插入、annotator 1 把 (0,1) 改为"저녁에"，拼接后后者不复存在），系统性低估 supported 率。
    现在拼接只在同一 annotator 内进行，且保留每一段连续子序列，与系统侧的合并粒度对齐。
    """
    by_span: Dict[Tuple[int, int], List[str]] = {}
    for _ann, es in s.annotator_groups().items():
        items = sorted(((e.start, e.end, [_norm_corr(c) for c in e.corrs])
                        for e in es if not e.is_noop), key=lambda t: (t[0], t[1]))
        for (a, b), cs, _members in _runs(items):
            by_span.setdefault((a, b), []).extend(_norm_corr(c) for c in cs)
    return [(a, b, list(dict.fromkeys(cs))) for (a, b), cs in by_span.items()]


CONF_SOURCES = ("esc", "seqlp", "mintok", "verbalized", "random")


def score_edits(s: M2Sentence, greedy: str, samples: Sequence[str],
                sent_id: int, group: Optional[str] = None,
                source: str = "esc", rec: Optional[dict] = None
                ) -> List[ScoredEdit]:
    """对 greedy 输出的每条编辑计算置信度与是否被参考支持。

    source 决定置信度来源，用于 C1 的对照：
      esc        —— 本文方法：编辑在 N 条采样中的复现频率（编辑级）
      seqlp      —— 长度归一化序列对数概率，句内所有编辑共享（句级）
      mintok     —— 该编辑替换串所覆盖 token 的最小概率（编辑级，但仅用解码信号）
      verbalized —— 模型自述把握度，句内共享（句级）
      random     —— 均匀随机，风险曲线无信息量的参照点
    seqlp/verbalized 为句级信号，天然无法区分同句内不同编辑，这正是要量化的差距。
    """
    g_edits = _canon(s.source, greedy)
    gold = _gold_edits(s)
    rec = rec or {}

    if source == "esc":
        counts: Dict[Tuple[int, int, str], int] = {}
        for smp in samples:
            for key in set(_canon(s.source, smp)):
                counts[key] = counts.get(key, 0) + 1
        n = max(1, len(samples))
        conf_of = lambda st, en, c: (counts.get((st, en, c), 0) / n
                                     if samples else 1.0)
    elif source == "seqlp":
        v = _seq_conf(rec)
        conf_of = lambda st, en, c: v
    elif source == "verbalized":
        v = _verbal_conf(rec)
        conf_of = lambda st, en, c: v
    elif source == "mintok":
        tokmap = _token_probs_by_char(rec)
        conf_of = lambda st, en, c: _span_min_prob(tokmap, greedy, c)
    elif source == "random":
        import random as _r
        rng = _r.Random(sent_id)
        conf_of = lambda st, en, c: rng.random()
    else:
        raise ValueError(f"未知置信度来源: {source}")

    out = []
    for st, en, corr in g_edits:
        sup = any(st == gs and en == ge and _norm_corr(corr) in gc for gs, ge, gc in gold)
        out.append(ScoredEdit(sent_id, group, st, en, corr,
                              float(conf_of(st, en, corr)), sup))
    return out


def _seq_conf(rec: dict) -> float:
    """exp(累计 logprob / token 数) —— 长度归一化的序列概率。"""
    lp, n = rec.get("greedy_cumlogprob"), rec.get("greedy_ntok")
    if lp is None or not n:
        return 1.0
    import math
    return math.exp(max(-20.0, lp / n))


def _verbal_conf(rec: dict) -> float:
    """解析模型自述的 0-100 把握度；解析失败按 0.5 处理并可在论文中披露失败率。"""
    import re
    m = re.search(r"\d+(?:\.\d+)?", str(rec.get("verbalized_raw", "")))
    if not m:
        return 0.5
    return min(max(float(m.group()) / 100.0, 0.0), 1.0)


def _token_probs_by_char(rec: dict) -> List[Tuple[int, int, float]]:
    """把逐 token logprob 展开为 [(char_start, char_end, prob)]，用于按字符区间定位。"""
    import math
    out, pos = [], 0
    for item in rec.get("greedy_tokens") or []:
        try:
            text, lp = item[0] or "", item[1]
        except (TypeError, IndexError):
            continue
        end = pos + len(text)
        if lp is not None:
            out.append((pos, end, math.exp(max(-20.0, float(lp)))))
        pos = end
    return out


def _span_min_prob(tokmap, hyp: str, corr: str) -> float:
    """取替换串在假设句中所处字符区间内、各 token 概率的最小值。

    定位不到（替换串为空或未出现）时回退到全句最小值；tokmap 为空时回退 1.0。
    """
    if not tokmap:
        return 1.0
    idx = hyp.find(corr) if corr else -1
    if idx < 0:
        return min(p for _a, _b, p in tokmap)
    a, b = idx, idx + len(corr)
    ps = [p for s, e, p in tokmap if s < b and e > a]
    return min(ps) if ps else min(p for _a, _b, p in tokmap)


# ---------- 校准指标 ----------

def ece(edits: Sequence[ScoredEdit], n_bins: int = 10) -> float:
    """Expected Calibration Error（等宽分箱）。"""
    if not edits:
        return float("nan")
    tot = len(edits)
    acc = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [e for e in edits
                  if (e.conf > lo or (b == 0 and e.conf >= lo)) and e.conf <= hi]
        if not bucket:
            continue
        avg_conf = sum(e.conf for e in bucket) / len(bucket)
        avg_acc = sum(1 for e in bucket if e.supported) / len(bucket)
        acc += len(bucket) / tot * abs(avg_conf - avg_acc)
    return acc


def brier(edits: Sequence[ScoredEdit]) -> float:
    if not edits:
        return float("nan")
    return sum((e.conf - (1.0 if e.supported else 0.0)) ** 2
               for e in edits) / len(edits)


# ---------- 选择性纠错 ----------

def risk_coverage(edits: Sequence[ScoredEdit],
                  n_points: int = 101) -> List[Tuple[float, float, float]]:
    """返回 [(threshold, coverage, unsupported_rate)]。

    coverage        = 保留编辑数 / 全部编辑数（即向学习者展示的反馈比例）
    unsupported_rate= 保留编辑中未被参考支持的比例（风险代理量）
    """
    if not edits:
        return []
    total = len(edits)
    out = []
    for i in range(n_points):
        thr = i / (n_points - 1)
        kept = [e for e in edits if e.conf >= thr]
        cov = len(kept) / total
        risk = (sum(1 for e in kept if not e.supported) / len(kept)
                if kept else 0.0)
        out.append((thr, cov, risk))
    return out


def aurc(edits: Sequence[ScoredEdit]) -> float:
    """Area Under Risk-Coverage curve（越低越好）。

    按置信度降序逐条纳入，风险取累计 unsupported 比例，
    对 coverage 做梯形积分并归一化到 [0,1] 覆盖区间。
    """
    if not edits:
        return float("nan")
    order = sorted(edits, key=lambda e: -e.conf)
    n = len(order)
    bad = 0
    prev_cov, prev_risk, area = 0.0, 0.0, 0.0
    for i, e in enumerate(order, start=1):
        if not e.supported:
            bad += 1
        cov, risk = i / n, bad / i
        area += (cov - prev_cov) * (risk + prev_risk) / 2
        prev_cov, prev_risk = cov, risk
    return area


def optimal_aurc(base_rate: float) -> float:
    """给定 unsupported 基础率 r，理想置信度排序器可达的最小 AURC。

    完美排序下：coverage <= 1-r 时风险为 0；此后 risk(c) = (c-(1-r))/c。
    积分得 AURC* = r + (1-r)·ln(1-r)。用于扣除"错误多本来就更难"的天然代价。
    """
    r = min(max(base_rate, 0.0), 1.0)
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return 1.0
    import math
    return r + (1 - r) * math.log(1 - r)


def excess_aurc(edits: Sequence[ScoredEdit]) -> float:
    """E-AURC = AURC - AURC*(同基础率)。

    这是跨层/跨组比较置信度**判别力**的正确量：直接比 AURC 会把
    "弱写作者错误更多"误读成"对弱写作者校准更差"。E-AURC 越低越好，
    0 表示置信度排序已达该基础率下的理论最优。
    """
    if not edits:
        return float("nan")
    base = 1 - sum(1 for e in edits if e.supported) / len(edits)
    return aurc(edits) - optimal_aurc(base)


def coverage_at(edits: Sequence[ScoredEdit], target_cov: float) -> Tuple[float, float]:
    """给定目标覆盖率，返回 (阈值, 该覆盖率下的 unsupported_rate)。"""
    if not edits:
        return float("nan"), float("nan")
    order = sorted(edits, key=lambda e: -e.conf)
    k = max(1, round(target_cov * len(order)))
    kept = order[:k]
    thr = kept[-1].conf
    risk = sum(1 for e in kept if not e.supported) / len(kept)
    return thr, risk


def by_group(edits: Sequence[ScoredEdit]) -> Dict[str, List[ScoredEdit]]:
    out: Dict[str, List[ScoredEdit]] = {}
    for e in edits:
        out.setdefault(e.group or "UNASSIGNED", []).append(e)
    return out
