"""按写作水平分层的可靠性分析（返修 R2.3；原 Table 5b / Fig 7；RQ4）。

分层：作文的人工评分"语法准确性"（두 评分者均值）——低 (<2) / 中 (2–3) / 高 (>3)，与原文一致；
另以作文三分位数切分作敏感性分析。分层是作文属性，故：
- 每个配置的 低−高 差值（ECE、E-AURC）用作文层面的置换检验（在低、高两层的作文之间
  打乱层标签，保持各层作文数）与分层作文 bootstrap 区间；
- 跨模型：每个模型平均两种提示条件后做 Wilcoxon 符号秩检验（精确；Pratt / Wilcox 两种零差值
  处理）与符号检验；
- GEE 逻辑回归（以作文为聚类、可交换相关结构，系统为固定效应）：
    supported ~ conf + low + conf:low + C(system)
  low 的系数 < 0 表示在相同置信度下，弱写作者的编辑更少被参考支持（即置信度对其更"虚高"）。
  另报告以连续的语法评分（中心化）代替 low 的版本。
用法: python stratified.py [--runs DIR ...] [--perm 2000] [--boot 1000]
"""
import argparse
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.essay_link import link_sentences  # noqa: E402
from kgc.data.kolla import load_essays, load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.data.rubric import grammar_score, load_scores  # noqa: E402
from kgc.eval.calibration import ece, excess_aurc, score_edits  # noqa: E402
from selective import wilcoxon_report  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
METRICS = {"ECE": ece, "E_AURC": excess_aurc}


def tier_fixed(score):
    return "low" if score < 2 else ("mid" if score <= 3 else "high")


def diff_stats(edits_by_essay, low_essays, high_essays, metric, n_perm, n_boot, seed):
    fn = METRICS[metric]
    lo = [e for k in low_essays for e in edits_by_essay[k]]
    hi = [e for k in high_essays for e in edits_by_essay[k]]
    obs = fn(lo) - fn(hi)
    rng = random.Random(seed)
    pool = list(low_essays) + list(high_essays)
    nl = len(low_essays)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        a = [e for k in pool[:nl] for e in edits_by_essay[k]]
        b = [e for k in pool[nl:] for e in edits_by_essay[k]]
        if a and b and abs(fn(a) - fn(b)) >= abs(obs) - 1e-12:
            hits += 1
    boots = []
    L, H = list(low_essays), list(high_essays)
    for _ in range(n_boot):
        a = [e for _ in L for e in edits_by_essay[L[rng.randrange(len(L))]]]
        b = [e for _ in H for e in edits_by_essay[H[rng.randrange(len(H))]]]
        if a and b:
            boots.append(fn(a) - fn(b))
    boots.sort()
    return {"low": fn(lo), "high": fn(hi), "diff": obs, "p_perm": (hits + 1) / (n_perm + 1),
            "ci": (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]),
            "n_low": len(lo), "n_high": len(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "results/runs_v2")])
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results/revision/stratified.json"))
    a = ap.parse_args()

    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = kept_indices(m2)
    links = link_sentences(load_essays(ROOT / "data/kolla_v2"), m2)
    gscore = grammar_score(load_scores(ROOT / "data/kolla_v2"))
    essay_of = {i: links[i]["essay"] for i in kept if links[i]["essay"] in gscore}
    essays = sorted(set(essay_of.values()))
    scores = np.array([gscore[k] for k in essays])
    t1, t2 = np.quantile(scores, [1 / 3, 2 / 3])
    schemes = {
        "fixed": {k: tier_fixed(gscore[k]) for k in essays},
        "tertile": {k: ("low" if gscore[k] <= t1 else ("high" if gscore[k] > t2 else "mid")) for k in essays},
    }
    info = {"n_sentences_scored": len(essay_of), "n_essays_scored": len(essays),
            "tertile_cuts": [float(t1), float(t2)],
            "essays_per_tier": {s: {t: sum(1 for v in m.values() if v == t) for t in ("low", "mid", "high")}
                                for s, m in schemes.items()}}
    print(json.dumps(info, ensure_ascii=False))

    files = sorted(f for d in a.runs for f in glob.glob(f"{d}/*.jsonl")
                   if "smoke" not in f and "baseline" not in f)
    per_cfg = {}
    rows = []
    for f in files:
        recs = {}
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            recs[r["sent_id"]] = r
        if len(recs) != len(m2):
            continue
        name = sysname(f)
        by_essay = defaultdict(list)
        for i, k in essay_of.items():
            for e in score_edits(m2[i], recs[i]["greedy"], recs[i].get("samples", []), i):
                by_essay[k].append(e)
                rows.append({"system": name, "essay": f"{k[0]}-{k[1]}", "supported": int(e.supported),
                             "conf": e.conf, "gscore": gscore[k],
                             "low": int(schemes["fixed"][k] == "low"), "high": int(schemes["fixed"][k] == "high")})
        per_cfg[name] = {}
        for sname, tiers in schemes.items():
            low = [k for k in essays if tiers[k] == "low" and by_essay[k]]
            high = [k for k in essays if tiers[k] == "high" and by_essay[k]]
            per_cfg[name][sname] = {m: diff_stats(by_essay, low, high, m, a.perm, a.boot, seed=7)
                                    for m in METRICS}
        d = per_cfg[name]["fixed"]
        print(f"{name:28s} ECE {d['ECE']['low']:.3f}/{d['ECE']['high']:.3f} Δ={d['ECE']['diff']:+.3f} "
              f"p={d['ECE']['p_perm']:.3f} | E-AURC {d['E_AURC']['low']:.3f}/{d['E_AURC']['high']:.3f} "
              f"Δ={d['E_AURC']['diff']:+.3f} p={d['E_AURC']['p_perm']:.3f}", flush=True)

    models = sorted({n.split(" (")[0] for n in per_cfg})
    across = {}
    for sname in schemes:
        for m in METRICS:
            diffs = [statistics.mean(per_cfg[n][sname][m]["diff"] for n in per_cfg if n.startswith(mm + " ("))
                     for mm in models]
            cfg_worse = sum(per_cfg[n][sname][m]["diff"] > 0 for n in per_cfg)
            across[f"{sname}:{m}"] = {**wilcoxon_report(diffs), "configs_low_worse": cfg_worse,
                                      "n_configs": len(per_cfg), "per_model": dict(zip(models, diffs))}
            v = across[f"{sname}:{m}"]
            print(f"[{sname}:{m}] 模型 {v['positive']}/{v['n']} 低层更差，配置 {cfg_worse}/{len(per_cfg)}；"
                  f"median Δ={v['median']:+.4f}  p_pratt={v.get('p_pratt')}  p_wilcox={v.get('p_wilcox')}  p_sign={v['p_sign']}")

    # ---- GEE ----
    df = pd.DataFrame(rows)
    df["conf_c"] = df["conf"] - df["conf"].mean()
    df["g_c"] = df["gscore"] - df["gscore"].mean()
    gee = {}
    sub = df[(df.low == 1) | (df.high == 1)]
    for label, data, formula in (
            ("low_vs_high", sub, "supported ~ conf_c * low + C(system)"),
            ("continuous", df, "supported ~ conf_c * g_c + C(system)")):
        model = smf.gee(formula, groups="essay", data=data, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
        r = model.fit()
        keep = [k for k in r.params.index if not k.startswith("C(system)")]
        gee[label] = {k: {"coef": float(r.params[k]), "se": float(r.bse[k]), "p": float(r.pvalues[k]),
                          "ci": [float(x) for x in r.conf_int().loc[k]]} for k in keep}
        gee[label]["n_edits"] = int(len(data))
        gee[label]["n_essays"] = int(data["essay"].nunique())
        print(f"\nGEE {label}（n={len(data)} 条编辑，{data['essay'].nunique()} 篇作文）")
        for k in keep:
            print(f"   {k:16s} {r.params[k]:+.3f}  SE={r.bse[k]:.3f}  p={r.pvalues[k]:.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"info": info, "per_config": per_cfg, "across_models": across, "gee": gee},
              open(a.out, "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
