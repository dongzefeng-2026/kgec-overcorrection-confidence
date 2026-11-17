"""选择性纠错与置信度估计器比较（返修 R1.8、R2.6；原 Table 4/5a）。

对每个配置（KoLLA 排除审计条目后的 1,393 句）：
- ESC 的 ECE、Brier、AURC、E-AURC、各覆盖率下的 unsupported rate、Δ50（全覆盖→50% 覆盖的相对降幅），
  E-AURC 与 Δ50 给作文聚类 bootstrap 95% CI；
- 五种估计器（ESC / 序列 logprob / 最小 token 概率 / 口头自评 / 随机）的 ECE、E-AURC、Δ50；
- 句内排序：同一句中既有被支持又有不被支持编辑时，(支持, 不支持) 编辑对中置信度排对的比例
  （平局记 0.5）。句级分数（序列 logprob、口头自评）在句内恒为平局，按构造等于 0.5——这正是
  "编辑级置信度能否区分同一句内的编辑"的直接检验。
跨配置汇总：每个模型先平均两种提示条件，再对模型做 Wilcoxon 符号秩检验（精确分布；零差值
用 Pratt 法，并报告 Wilcox 法）与符号检验。
用法: python selective.py [--runs DIR ...] [--boot 1000]
"""
import argparse
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.essay_link import link_sentences  # noqa: E402
from kgc.data.kolla import load_essays, load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import (CONF_SOURCES, aurc, brier, coverage_at, ece,  # noqa: E402
                                  excess_aurc, score_edits)
from kgc.eval.stats import cluster_index  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def delta50(edits):
    base = 1 - sum(e.supported for e in edits) / len(edits)
    _thr, r50 = coverage_at(edits, 0.5)
    return (base - r50) / base * 100 if base else float("nan")


def within_sentence(edits):
    """返回 (排对得分和, 编辑对数)。"""
    by = defaultdict(list)
    for e in edits:
        by[e.sent_id].append(e)
    score = n = 0
    for es in by.values():
        pos = [e.conf for e in es if e.supported]
        neg = [e.conf for e in es if not e.supported]
        for p in pos:
            for q in neg:
                n += 1
                score += 1.0 if p > q else (0.5 if p == q else 0.0)
    return score, n


def boot_ci(edits, clusters_of, fn, n_boot, seed):
    """以作文为单位重抽样编辑。"""
    groups = defaultdict(list)
    for e in edits:
        groups[clusters_of[e.sent_id]].append(e)
    keys = list(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [e for _ in keys for e in groups[keys[rng.randrange(len(keys))]]]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def wilcoxon_report(diffs):
    out = {"n": len(diffs), "median": statistics.median(diffs),
           "positive": sum(d > 0 for d in diffs), "negative": sum(d < 0 for d in diffs)}
    for zm in ("pratt", "wilcox"):
        try:
            r = sps.wilcoxon(diffs, zero_method=zm, alternative="two-sided", method="exact"
                             if zm == "wilcox" else "auto")
            out[f"W_{zm}"], out[f"p_{zm}"] = float(r.statistic), float(r.pvalue)
        except ValueError as ex:
            out[f"p_{zm}"] = None
            out[f"err_{zm}"] = str(ex)
    k, nz = out["positive"], out["positive"] + out["negative"]
    out["p_sign"] = float(sps.binomtest(k, nz).pvalue) if nz else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "results/runs_v2")])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results/revision/selective.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = kept_indices(m2)
    links = link_sentences(load_essays(ROOT / "data/kolla_v2"), m2)
    cl = {i: links[i]["cluster"] for i in range(len(m2))}

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
        row = {}
        for src in CONF_SOURCES:
            E = [e for i in kept for e in score_edits(m2[i], recs[i]["greedy"], recs[i].get("samples", []),
                                                         i, source=src, rec=recs[i])]
            ws, wn = within_sentence(E)
            d = {"n_edits": len(E), "supported": sum(e.supported for e in E) / len(E),
                 "ECE": ece(E), "Brier": brier(E), "AURC": aurc(E), "E_AURC": excess_aurc(E),
                 "risk": {str(c): coverage_at(E, c)[1] for c in (1.0, 0.9, 0.8, 0.5)},
                 "delta50": delta50(E), "within_acc": ws / wn if wn else None, "within_pairs": wn}
            if src == "esc":
                d["E_AURC_ci"] = boot_ci(E, cl, excess_aurc, a.boot, 1)
                d["delta50_ci"] = boot_ci(E, cl, delta50, a.boot, 2)
            row[src] = d
        res[name] = row
        e = row["esc"]
        print(f"{name:28s} n={e['n_edits']:5d} supp={e['supported']:.3f} ECE={e['ECE']:.3f} "
              f"E-AURC={e['E_AURC']:.3f} Δ50={e['delta50']:5.1f}% [{e['delta50_ci'][0]:.1f},{e['delta50_ci'][1]:.1f}] "
              f"| within: esc={e['within_acc']:.3f} mintok={row['mintok']['within_acc']:.3f} "
              f"seqlp={row['seqlp']['within_acc']:.3f} rand={row['random']['within_acc']:.3f} (pairs {e['within_pairs']})",
              flush=True)

    # ---- 跨配置汇总 ----
    names = list(res)
    summary = {"median": {}, "tests": {}}
    for src in CONF_SOURCES:
        summary["median"][src] = {k: statistics.median(res[n][src][k] for n in names)
                                  for k in ("ECE", "AURC", "E_AURC", "delta50", "within_acc")}
    models = sorted({n.split(" (")[0] for n in names})

    def per_model(metric, src):
        out = []
        for m in models:
            v = [res[n][src][metric] for n in names if n.startswith(m + " (")]
            out.append(statistics.mean(v))
        return out

    for other in ("seqlp", "mintok", "verbalized", "random"):
        for metric in ("E_AURC", "delta50", "ECE", "within_acc"):
            d = [x - y for x, y in zip(per_model(metric, "esc"), per_model(metric, other))]
            summary["tests"][f"esc_vs_{other}:{metric}"] = wilcoxon_report(d)
    print("\n中位数（跨配置）：")
    for src, v in summary["median"].items():
        print(f"  {src:10s} ECE={v['ECE']:.3f} AURC={v['AURC']:.3f} E-AURC={v['E_AURC']:.3f} "
              f"Δ50={v['delta50']:.1f}% within={v['within_acc']:.3f}")
    print("\nESC 与其他估计器（按模型平均后的配对差，ESC − other）：")
    for k, v in summary["tests"].items():
        print(f"  {k:28s} median={v['median']:+.4f} +{v['positive']}/-{v['negative']} "
              f"p_pratt={v.get('p_pratt')} p_wilcox={v.get('p_wilcox')} p_sign={v['p_sign']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"configs": res, "summary": summary, "models": models}, open(a.out, "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
