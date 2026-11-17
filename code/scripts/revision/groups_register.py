"""按学习者组的分析（返修 R1.3 / RQ4）：FB / FI / HB / HI（外语或继承语 × 初级或中级）。

对每个配置：各组的官方 M² F0.5（作文聚类 bootstrap CI）、无需修改句误改率；
各组的句长（eojeol）与参考编辑密度；语体改动的方向（해요체 -> 합쇼체 等）按提示条件汇总；
置信度为 1.0 的编辑被支持的比例（"满票"编辑的实际可靠性）。
用法: python groups_register.py [--runs DIR] [--boot 1000]
"""
import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.analysis.edit_types import speech_level  # noqa: E402
from kgc.data.essay_link import link_sentences  # noqa: E402
from kgc.data.groups import assign_groups  # noqa: E402
from kgc.data.kolla import load_essays, load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import score_edits  # noqa: E402
from kgc.eval.m2_official import corpus_multi, corpus_stats_for_runs  # noqa: E402
from kgc.eval.stats import cluster_bootstrap  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("FB", "FI", "HB", "HI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results/runs_v2"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results/revision/groups_register.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = kept_indices(m2)
    ess = load_essays(ROOT / "data/kolla_v2")
    grp = [g for g, _ in assign_groups(ess, m2)]
    links = link_sentences(ess, m2)
    by_g = {g: [i for i in kept if grp[i] == g] for g in GROUPS}
    noedit = {i for i in kept if all(e.is_noop for e in m2[i].edits)}
    res = {"group_info": {g: {"n_sent": len(ix), "mean_eojeol": statistics.mean(len(m2[i].source.split()) for i in ix),
                              "gold_edits_per_sent": statistics.mean(
                                  sum(1 for e in m2[i].edits if not e.is_noop) / 2 for i in ix),
                              "n_noedit": sum(1 for i in ix if i in noedit)} for g, ix in by_g.items()},
           "configs": {}}
    reg_dir = defaultdict(Counter)
    for f in sorted(glob.glob(f"{a.runs}/*.jsonl")):
        if "baseline" in f:
            continue
        R = {}
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            R[r["sent_id"]] = r
        name = sysname(f)
        st = corpus_stats_for_runs(m2, [R[i]["greedy"] for i in range(len(m2))])
        row = {}
        for g, ix in by_g.items():
            cl = [links[i]["cluster"] for i in ix]
            F, lo, hi = cluster_bootstrap(lambda idx, ix=ix: corpus_multi(st, order=[ix[j] for j in idx])[2],
                                          cl, a.boot, seed=3)
            ne = [i for i in ix if i in noedit]
            fa = sum(R[i]["greedy"].split() != m2[i].source.split() for i in ne) / len(ne) if ne else None
            row[g] = {"F": F, "lo": lo, "hi": hi, "false_alarm": fa}
        worst = min(GROUPS, key=lambda g: row[g]["F"])
        # "满票"编辑
        E = [e for i in kept for e in score_edits(m2[i], R[i]["greedy"], R[i]["samples"], i)]
        full = [e for e in E if e.conf >= 0.999]
        row["unanimous_supported"] = sum(e.supported for e in full) / len(full) if full else None
        row["unanimous_share"] = len(full) / len(E) if E else None
        row["worst_group"] = worst
        res["configs"][name] = row
        cond = "4-shot" if "4-shot" in name else "0-shot"
        for i in kept:
            s0, s1 = speech_level(m2[i].source), speech_level(R[i]["greedy"])
            if s0 and s1 and s0 != s1:
                reg_dir[cond][f"{s0}->{s1}"] += 1
        print(f"{name:28s} " + "  ".join(f"{g} {row[g]['F']:.3f}" for g in GROUPS) +
              f" | 最差 {worst} | 满票编辑占 {row['unanimous_share']:.2f}，被支持 {row['unanimous_supported']:.3f}", flush=True)
    res["register_direction"] = {k: dict(v.most_common()) for k, v in reg_dir.items()}
    print("组信息:", json.dumps(res["group_info"], ensure_ascii=False))
    print("语体改动方向:", json.dumps(res["register_direction"], ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
