"""M2 评测器（py3 干净实现，语义对齐 NUS CoNLL-2014 m2scorer）。

流程：
1. 将系统输出（hypothesis 文本）与 gold 源句 token 对齐（difflib LCS），
   差异区段生成 hypothesis edits（span 落在源 token 索引上）；
2. 对每个 gold annotator：hyp edit 与 gold edit 做 span + 校正 token
   精确匹配（type 参与与否由 use_type 控制，见注）；
3. F-beta（默认 0.5）。

多参考（KoLLA 双 annotator）模式：
- mode='max'    ：hyp edit 匹配任一 annotator 即算正确（多参考主流做法）；
- mode='per-ann'：逐 annotator 算 P/R/F 再平均（单参考 vs 多参考敏感性分析的原料）。

注（type 层）：hyp edit 的类型需韩语 ERRANT 式标注器（KAGAS）推断，本模块
不承担类型分类；use_type=False 时 type 不参与匹配，error-type 分面分析
由外层（matched edit 的 gold type 桶）完成。
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from kgc.data.m2_io import Edit, M2Sentence

BETA = 0.5


@dataclass
class HypEdit:
    start: int                 # 源 token span（含）
    end: int                   # 不含
    corr_tokens: List[str]     # 校正 token 序列


def align_hyp_edits(source: str, hyp: str) -> List[HypEdit]:
    """源句与假设句 token 对齐（difflib LCS），差异 -> hyp edits。"""
    src = source.split()
    hys = hyp.split()
    if src == hys:
        return []
    sm = difflib.SequenceMatcher(None, src, hys, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # replace/delete: 覆盖源 [i1,i2)；insert: 0 宽，锚定在 i1 处
        if tag == "insert":
            edits.append(HypEdit(start=i1, end=i1, corr_tokens=hys[j1:j2]))
        else:
            edits.append(HypEdit(start=i1, end=i2, corr_tokens=hys[j1:j2]))
    return edits


def merge_edits(items):
    """相接编辑合并（canonicalization，ERRANT/NUS 语义）。

    items: List[(start, end, corrs: List[str])]。相邻编辑（前.end == 后.start）
    合并为 (start, 后.end)；corr 为两侧候选的笛卡尔积空格拼接：
    - 单候选: 替换(0,1,'X') + 插入(1,1,'Y') => (0,1,'X Y')（difflib 合并粒度对齐）；
    - 多候选: (0,1,'A'|'B') + (1,2,'C'|'D') => (0,2,'A C','A D','B C','B D')。
    链式迭代至不动点；排序使 insert@s 先于同点 replace 处理。
    """
    es = sorted(items, key=lambda e: (e[0], e[1]))
    while True:
        out = []
        changed = False
        for e in es:
            if out and out[-1][1] == e[0]:
                ps, _, pc = out[-1]
                comb = [f"{a} {b}" for a in pc for b in e[2]]
                out[-1] = (ps, e[1], list(dict.fromkeys(comb)))
                changed = True
            else:
                out.append(e)
        es = out
        if not changed:
            return es


def _norm(c: str) -> str:
    """参考中的删除记作 -NONE-，系统侧删除为空串；匹配前统一为空串（返修更正）。"""
    return "" if c == "-NONE-" else " ".join(c.split())


def _match(he: HypEdit, ge: Edit) -> bool:
    """span 精确相等 + 校正文本精确匹配（多候选任一成立）。type 层由外部负责。"""
    if he.start != ge.start or he.end != ge.end:
        return False
    return " ".join(he.corr_tokens) in [_norm(c) for c in ge.corrs]


def score_sentence(s: M2Sentence, hyp: str,
                   mode: str = "best") -> Dict[str, float]:
    """单句打分。返回 {'P','R','F'}。

    参考口径（三者语义不同，C2 的评测偏差必须用 best vs per-ann）：
    - best   ：**标准多参考协议** —— 逐句对每个 annotator 独立打分后取 F 最高者。
               「系统是否匹配了某一种合法改法」。多参考分数在数学上恒 >= 任一
               单参考分数，故偏差恒非负。
    - per-ann：逐 annotator 独立打分再平均 —— 模拟只有一个参考时的评测结果。
    - union  ：把各 annotator 的编辑取并集当作单一 gold（**非标准**，仅作敏感性
               对照）。它允许系统混搭不同标注者的编辑，且在「标注者对该句改不改
               都不一致」时反而低于 per-ann 平均，会产生看似为负的偏差。

    空编辑语义（NUS）：某 annotator 无编辑且 hyp 也无编辑 => 该对记全对。
    """
    hedits = align_hyp_edits(s.source, hyp)
    ann_groups: Dict[int, List[Edit]] = s.annotator_groups()
    # 去掉 noop 占位，纯编辑才参与；ann 无任何编辑 => 空 gold 集
    ann_edits = {a: [e for e in es if not e.is_noop]
                 for a, es in ann_groups.items()}
    if not ann_edits:
        return _fbeta(hedits, [])

    per_ann = [_fbeta(hedits, ges) for ges in ann_edits.values()]

    if mode == "per-ann":
        return _macro(per_ann)
    if mode == "best":
        return max(per_ann, key=lambda d: d["F"])
    if mode in ("union", "max"):   # max 为旧名，保留兼容
        return _fbeta(hedits, [e for ges in ann_edits.values() for e in ges])
    raise ValueError(f"未知参考口径: {mode}")


def _runs(items):
    """由相邻编辑生成"连读"候选：[(span, corrs, 构成成员下标集)]。

    仅用于**生成匹配候选**，不替换原始单条编辑——早期实现直接把 gold
    合并掉，导致 gold 里本就存在的单条编辑（如 (0,1,'저녁에')）被吞进
    (0,1,'저녁에 열 시에') 而无法匹配，系统性低估 supported 率。
    长度 1 的候选即原编辑本身。
    """
    out = []
    n = len(items)
    for i in range(n):
        s0, e0, c0 = items[i]
        out.append(((s0, e0), list(c0), frozenset([i])))
        cur_end, cur_corrs, members = e0, list(c0), [i]
        for j in range(i + 1, n):
            sj, ej, cj = items[j]
            if sj != cur_end:            # 不相邻则停止延伸
                break
            cur_corrs = [f"{a} {b}" for a in cur_corrs for b in cj]
            cur_corrs = list(dict.fromkeys(cur_corrs))[:16]   # 限爆炸
            members = members + [j]
            cur_end = ej
            out.append(((s0, cur_end), list(cur_corrs), frozenset(members)))
    return out


def _fbeta(hedits: List[HypEdit], golds: List[Edit], beta: float = BETA):
    """在原始编辑粒度上计分，允许"连读"候选跨越粒度差异。

    P 的分母是 hyp 原始编辑数，R 的分母是 gold 原始编辑数；一次匹配
    可消耗两侧各若干条（当粒度不同时）。先试长度 1 的匹配，再试更长的
    连读，保证同粒度的精确匹配优先。
    无编辑句语义（NUS）：两侧皆空 => 全对；一侧空另一侧非空 => 全错。
    """
    by_span: Dict[tuple, List[str]] = {}
    for ge in golds:
        by_span.setdefault((ge.start, ge.end), []).extend(_norm(c) for c in ge.corrs)
    gold_items = sorted(((s, e, list(dict.fromkeys(cs)))
                         for (s, e), cs in by_span.items()),
                        key=lambda t: (t[0], t[1]))
    hyp_items = sorted(((he.start, he.end, [" ".join(he.corr_tokens)])
                        for he in hedits), key=lambda t: (t[0], t[1]))
    n_gold, n_hyp = len(gold_items), len(hyp_items)

    g_cands, h_cands = _runs(gold_items), _runs(hyp_items)
    used_g, used_h = set(), set()
    # 先精确同粒度（成员数均为 1），再允许跨粒度连读
    for max_len in (1, 99):
        for hspan, hcorrs, hmem in h_cands:
            if len(hmem) > max_len or hmem & used_h:
                continue
            for gspan, gcorrs, gmem in g_cands:
                if len(gmem) > max_len or gmem & used_g or gspan != hspan:
                    continue
                if any(hc in gcorrs for hc in hcorrs):
                    used_h |= hmem
                    used_g |= gmem
                    break

    p = len(used_h) / n_hyp if n_hyp else (1.0 if n_gold == 0 else 0.0)
    r = len(used_g) / n_gold if n_gold else (1.0 if n_hyp == 0 else 0.0)
    if p == 0.0 and r == 0.0:
        return {"P": 0.0, "R": 0.0, "F": 0.0}
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
    return {"P": p, "R": r, "F": f}


def _macro(fs: List[Dict[str, float]]) -> Dict[str, float]:
    if not fs:
        return {"P": 1.0, "R": 1.0, "F": 1.0}
    return {k: sum(f[k] for f in fs) / len(fs) for k in ("P", "R", "F")}


def score_corpus(m2: Iterable[M2Sentence], hyps: Iterable[str],
                 mode: str = "best") -> Dict[str, float]:
    """整语料 macro：句子级指标均值（句子粒度加权一致，文档级自举在外层）。"""
    acc = {"P": 0.0, "R": 0.0, "F": 0.0}
    n = 0
    for s, h in zip(m2, hyps):
        sc = score_sentence(s, h, mode=mode)
        for k in acc:
            acc[k] += sc[k]
        n += 1
    return {k: v / n for k, v in acc.items()} if n else acc
