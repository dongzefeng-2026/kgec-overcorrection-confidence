"""编辑级多数投票（Goto et al., BEA 2026）复现，以及 ESC 阈值的完整操作曲线（返修 R2.6）。

三种输出均用官方 M²（多参考）、GLEU 与无需修改句误改率评估：
- greedy：原贪心输出；
- ESC@τ：只应用 ESC 置信度 >= τ 的贪心编辑，τ ∈ {0, 0.1, ..., 1.0}（τ=0 即贪心）；
  τ = 0.6 即"多数票"（N=10 时 >5 票）作用在贪心编辑上；
- MV-all：Goto et al. 的做法——在 N 个采样候选的全部编辑中保留出现次数过半者
  （不要求出现在贪心输出中），按票数从高到低贪心地放入互不重叠的编辑后生成句子。
目的：说明多数投票是本文风险–覆盖曲线上的一个固定点；本文另外提供校准度量、
完整操作曲线和按学习者分层的分析。
用法: python majority_vote.py [--runs DIR ...]
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.kolla import load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import _canon  # noqa: E402
from kgc.eval.gleu_official import GleuCache, _picks, corpus_np  # noqa: E402
from kgc.eval.m2_official import corpus_multi, corpus_stats_for_runs  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TAUS = [round(0.1 * k, 1) for k in range(11)]


def apply(source: str, edits):
    """把互不重叠的编辑 [(start, end, corr)] 应用到源句 token 上。"""
    toks = source.split()
    for st, en, corr in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        toks[st:en] = [] if corr in ("", "-NONE-") else corr.split()
    return " ".join(toks)


def non_overlapping(cands):
    """cands: [(votes, (start, end, corr))]，按票数降序选取互不重叠的编辑。"""
    chosen = []
    for _v, (st, en, c) in sorted(cands, key=lambda x: (-x[0], x[1])):
        clash = False
        for s2, e2, _c2 in chosen:
            if st == en or s2 == e2:                       # 插入：同一锚点只保留一个
                if (st == en and s2 <= st <= e2 and (s2 < st < e2 or s2 == e2 == st)) or \
                   (s2 == e2 and st <= s2 <= en and (st < s2 < en or st == en == s2)):
                    clash = True
            elif st < e2 and s2 < en:
                clash = True
        if not clash:
            chosen.append((st, en, c))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "results/runs_v2")])
    ap.add_argument("--out", default=str(ROOT / "results/revision/majority_vote.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = kept_indices(m2)
    noedit = [i for i in kept if all(e.is_noop for e in m2[i].edits)]
    gc, picks = GleuCache(m2), _picks(len(m2))

    def evaluate(hyps):
        st = corpus_stats_for_runs(m2, hyps)
        P, R, F, _ = corpus_multi(st, order=kept)
        g = corpus_np(np.array(gc.sentence_stats(hyps)), picks, kept)
        fa = sum(hyps[i].split() != m2[i].source.split() for i in noedit) / len(noedit)
        return {"P": P, "R": R, "F": F, "GLEU": g, "FA": fa}

    files = sorted(f for d in a.runs for f in glob.glob(f"{d}/*.jsonl")
                   if "smoke" not in f and "baseline" not in f)
    res = {}
    for f in files:
        recs = {}
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            recs[r["sent_id"]] = r
        if len(recs) != len(m2):
            continue
        name = sysname(f)
        curve = {}
        greedy_edits, sample_votes = {}, {}
        for i in range(len(m2)):
            s, r = m2[i], recs[i]
            g_ed = _canon(s.source, r["greedy"])
            votes = Counter()
            for smp in r.get("samples", []):
                for e in set(_canon(s.source, smp)):
                    votes[e] += 1
            greedy_edits[i], sample_votes[i] = g_ed, votes
        n_s = max(1, len(recs[0].get("samples", [])))
        for tau in TAUS:
            hyps = []
            for i in range(len(m2)):
                keep = [e for e in greedy_edits[i] if sample_votes[i][e] / n_s >= tau - 1e-9]
                hyps.append(apply(m2[i].source, keep))
            curve[str(tau)] = evaluate(hyps)
        mv_hyps = [apply(m2[i].source, non_overlapping([(v, e) for e, v in sample_votes[i].items()
                                                         if v > n_s / 2]))
                   for i in range(len(m2))]
        res[name] = {"esc_curve": curve, "mv_all": evaluate(mv_hyps)}
        best = max(TAUS, key=lambda t: curve[str(t)]["F"])
        g0, mv, c6 = curve["0.0"], res[name]["mv_all"], curve["0.6"]
        print(f"{name:28s} greedy F={g0['F']:.4f} P={g0['P']:.3f} FA={g0['FA']:.3f} | "
              f"ESC@0.6 F={c6['F']:.4f} P={c6['P']:.3f} FA={c6['FA']:.3f} | "
              f"MV-all F={mv['F']:.4f} P={mv['P']:.3f} FA={mv['FA']:.3f} | best τ={best} F={curve[str(best)]['F']:.4f}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
