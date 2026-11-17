"""错误类型分析（返修 R1.3 / R1.10；对应 RQ1、RQ4）。

A. 参考编辑的类型分布：按标注者、按学习者背景（外语 F / 继承语 H）与水平（B / I）；
B. 系统编辑（与校准分析同一套编辑抽取）按类型：占比、被参考支持率、ESC 平均置信度；
C. 无需修改句（两位标注者都未改）上的系统编辑——即误改——的类型构成；
D. 句级语体与敬语改动率：系统 vs 两位标注者。
用法: ~/.venvs/kgec-rev1/bin/python edit_type_analysis.py [--runs DIR ...]
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.analysis.edit_types import classify, coarse, honorific_count, speech_level  # noqa: E402
from kgc.data.groups import assign_groups  # noqa: E402
from kgc.data.kolla import load_essays, load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import score_edits  # noqa: E402
from kgc.eval.gleu_official import apply_edits  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ORDER = ["Particle", "Ending", "Particle+ending", "Spelling", "Spacing", "Lexical choice",
         "Insertion/deletion", "Word order", "Multi-word rewrite", "Punctuation"]


def pct(c: Counter):
    n = sum(c.values()) or 1
    return {k: 100 * c.get(k, 0) / n for k in ORDER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "results/runs_v2")])
    ap.add_argument("--out", default=str(ROOT / "results/revision/edit_types.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = kept_indices(m2)
    groups = [g for g, _ in assign_groups(load_essays(ROOT / "data/kolla_v2"), m2)]
    noedit = {i for i in kept if all(e.is_noop for e in m2[i].edits)}
    res = {}

    # ---- A. 参考编辑 ----
    gold = {"ann0": Counter(), "ann1": Counter(), "F": Counter(), "H": Counter(),
            "B": Counter(), "I": Counter()}
    for g in ("FB", "FI", "HB", "HI"):
        gold[g] = Counter()
    ref_reg, ref_hon = Counter(), Counter()
    for i in kept:
        s = m2[i]
        for ann in (0, 1):
            ref = apply_edits(s, ann)
            ref_reg[ann] += int(speech_level(s.source) is not None and speech_level(ref) is not None
                                and speech_level(s.source) != speech_level(ref))
            ref_hon[ann] += int(honorific_count(s.source) != honorific_count(ref))
        for e in s.edits:
            if e.is_noop:
                continue
            src_t, cor_t = s.source.split()[e.start:e.end], ([] if e.corrs[0] == "-NONE-" else e.corrs[0].split())
            t = coarse(classify(src_t, cor_t)["type"])
            gold[f"ann{e.annotator}"][t] += 1
            g = groups[i]
            if g:
                gold[g][t] += 1
                gold[g[0]][t] += 1
                gold[g[1]][t] += 1
    res["gold_counts"] = {k: dict(v) for k, v in gold.items()}
    res["gold_pct"] = {k: pct(v) for k, v in gold.items()}
    n = len(kept)
    res["ref_register_shift_rate"] = {f"ann{k}": ref_reg[k] / n for k in (0, 1)}
    res["ref_honorific_shift_rate"] = {f"ann{k}": ref_hon[k] / n for k in (0, 1)}

    print("A. 参考编辑类型分布（%）")
    print(f"{'':20s}" + "".join(f"{k[:10]:>11s}" for k in ("ann0", "ann1", "F", "H", "B", "I")))
    for t in ORDER:
        print(f"{t:20s}" + "".join(f"{res['gold_pct'][k][t]:11.1f}" for k in ("ann0", "ann1", "F", "H", "B", "I")))
    print("   编辑总数:", {k: sum(gold[k].values()) for k in ("ann0", "ann1", "F", "H", "B", "I")})

    # ---- B/C/D. 系统 ----
    res["systems"] = {}
    pooled = defaultdict(lambda: [0, 0, 0.0])          # type -> [n, supported, sum conf]
    pooled_fa = Counter()
    files = sorted(f for d in a.runs for f in glob.glob(f"{d}/*.jsonl")
                   if "smoke" not in f and "baseline" not in f)
    for f in files:
        recs = {}
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            recs[r["sent_id"]] = r
        if len(recs) != len(m2):
            continue
        name = sysname(f)
        by = defaultdict(lambda: [0, 0, 0.0])
        fa = Counter()
        reg = hon = 0
        for i in kept:
            s, r = m2[i], recs[i]
            hyp = r["greedy"]
            a_, b_ = speech_level(s.source), speech_level(hyp)
            reg += int(a_ is not None and b_ is not None and a_ != b_)
            hon += int(honorific_count(s.source) != honorific_count(hyp))
            for e in score_edits(s, hyp, r.get("samples", []), i):
                src_t, cor_t = s.source.split()[e.start:e.end], e.corr.split()
                t = coarse(classify(src_t, cor_t)["type"])
                for d in (by[t], pooled[t]):
                    d[0] += 1; d[1] += int(e.supported); d[2] += e.conf
                if i in noedit:
                    fa[t] += 1; pooled_fa[t] += 1
        tot = sum(v[0] for v in by.values()) or 1
        res["systems"][name] = {
            "share_pct": {t: 100 * by[t][0] / tot for t in ORDER},
            "supported_rate": {t: (by[t][1] / by[t][0] if by[t][0] else None) for t in ORDER},
            "mean_conf": {t: (by[t][2] / by[t][0] if by[t][0] else None) for t in ORDER},
            "n": {t: by[t][0] for t in ORDER},
            "false_alarm_pct": pct(fa), "false_alarm_n": sum(fa.values()),
            "register_shift_rate": reg / n, "honorific_shift_rate": hon / n,
        }
    res["pooled"] = {t: {"n": v[0], "supported_rate": v[1] / v[0], "mean_conf": v[2] / v[0]}
                     for t, v in pooled.items() if v[0]}
    res["pooled_false_alarm_pct"] = pct(pooled_fa)

    print("\nB. 全部系统汇总：类型 | 编辑数 | 被支持率 | ESC 均值 | 误改构成%")
    for t in ORDER:
        v = res["pooled"].get(t)
        if v:
            print(f"{t:20s} {v['n']:6d}  {v['supported_rate']:.3f}  {v['mean_conf']:.3f}  "
                  f"{res['pooled_false_alarm_pct'][t]:5.1f}")
    print("\nD. 句级语体改动率 / 敬语改动率")
    print(f"{'Reference 1':28s} {res['ref_register_shift_rate']['ann0']:.3f} / {res['ref_honorific_shift_rate']['ann0']:.3f}")
    print(f"{'Reference 2':28s} {res['ref_register_shift_rate']['ann1']:.3f} / {res['ref_honorific_shift_rate']['ann1']:.3f}")
    for name, v in res["systems"].items():
        print(f"{name:28s} {v['register_shift_rate']:.3f} / {v['honorific_shift_rate']:.3f}   "
              f"误改数 {v['false_alarm_n']}  词汇替换占比 {v['share_pct']['Lexical choice']:.1f}%  "
              f"词汇替换被支持率 {v['supported_rate']['Lexical choice'] or 0:.2f}  助词被支持率 {v['supported_rate']['Particle'] or 0:.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
