"""选择性纠错的操作点分析（返修 R1.8：可部署性主张的检验；方法选择 N 的依据）。

A. 阈值的作文级留出验证：按作文（聚类）随机对半划分，在校准半中选出达到目标覆盖率的
   ESC 阈值，用到留出半上，报告实际覆盖率、unsupported rate 与相对全覆盖的降幅；
   重复 R 次，报告中位数与 2.5–97.5 百分位。
B. 采样数 N 的敏感性：从每句的 10 个样本中随机子抽 N ∈ {2,4,6,8}（重复 20 次），
   重算 ESC 的 E-AURC、ECE 与 Δ50。
用法: python operating_point.py [--runs DIR] [--reps 200]
"""
import argparse
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.essay_link import link_sentences  # noqa: E402
from kgc.data.kolla import load_essays, load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import ece, excess_aurc, score_edits  # noqa: E402
from selective import delta50  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TARGET = 0.5


def threshold_for(edits, target):
    """最小的 τ（取值于 {0, .1, ..., 1}），使保留比例 <= target；与原文 Algorithm 1 一致。"""
    n = len(edits)
    for k in range(11):
        tau = k / 10
        if sum(e.conf >= tau - 1e-9 for e in edits) / n <= target:
            return tau
    return 1.01


def risk(edits):
    return sum(not e.supported for e in edits) / len(edits) if edits else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results/runs_v2"))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results/revision/operating_point.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = kept_indices(m2)
    links = link_sentences(load_essays(ROOT / "data/kolla_v2"), m2)
    res = {}
    for f in sorted(glob.glob(f"{a.runs}/*.jsonl")):
        if "baseline" in f:
            continue
        R = {}
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            R[r["sent_id"]] = r
        name = sysname(f)
        by_cl = defaultdict(list)
        for i in kept:
            by_cl[links[i]["cluster"]] += score_edits(m2[i], R[i]["greedy"], R[i]["samples"], i)
        clusters = sorted(by_cl)
        rng = random.Random(0)
        cov, rk, red, taus = [], [], [], []
        for _ in range(a.reps):
            rng.shuffle(clusters)
            half = len(clusters) // 2
            cal = [e for c in clusters[:half] for e in by_cl[c]]
            test = [e for c in clusters[half:] for e in by_cl[c]]
            tau = threshold_for(cal, TARGET)
            kept_e = [e for e in test if e.conf >= tau - 1e-9]
            cov.append(len(kept_e) / len(test))
            base = risk(test)
            rk.append(risk(kept_e))
            red.append((base - risk(kept_e)) / base * 100 if kept_e and base else float("nan"))
            taus.append(tau)

        def summ(v):
            v = sorted(x for x in v if x == x)
            return {"median": statistics.median(v), "lo": v[int(0.025 * len(v))], "hi": v[int(0.975 * len(v)) - 1]}

        # N 敏感性
        nsens = {}
        for n_s in (2, 4, 6, 8, 10):
            vals = defaultdict(list)
            for rep in range(1 if n_s == 10 else 20):
                rr = random.Random(100 + rep)
                E = []
                for i in kept:
                    smp = R[i]["samples"]
                    sub = smp if n_s >= len(smp) else rr.sample(smp, n_s)
                    E += score_edits(m2[i], R[i]["greedy"], sub, i)
                vals["E_AURC"].append(excess_aurc(E))
                vals["ECE"].append(ece(E))
                vals["delta50"].append(delta50(E))
            nsens[n_s] = {k: statistics.mean(v) for k, v in vals.items()}
        res[name] = {"heldout": {"coverage": summ(cov), "risk": summ(rk), "reduction_pct": summ(red),
                                 "tau_values": dict(sorted(((str(t), taus.count(t)) for t in set(taus))))},
                     "n_sensitivity": nsens}
        h = res[name]["heldout"]
        print(f"{name:28s} 留出：覆盖率 {h['coverage']['median']:.2f} [{h['coverage']['lo']:.2f},{h['coverage']['hi']:.2f}] "
              f"降幅 {h['reduction_pct']['median']:5.1f}% [{h['reduction_pct']['lo']:.1f},{h['reduction_pct']['hi']:.1f}] | "
              f"N=2/4/6/8/10 E-AURC " + " ".join(f"{nsens[k]['E_AURC']:.3f}" for k in (2, 4, 6, 8, 10)) +
              " Δ50 " + " ".join(f"{nsens[k]['delta50']:.1f}" for k in (2, 4, 6, 8, 10)), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
