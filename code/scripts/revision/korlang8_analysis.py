"""Kor-Lang8 外部验证分析（返修 R2.4）。

无作者信息，故按句子重抽样（局限中说明）。对每个配置：
- 官方 M²（多参考，全部 3,000 句）P/R/F、句子 bootstrap 95% CI；GLEU；
- 双参考子集（1,562 句）上的多参考 vs 单参考（参考 1、参考 2 及平均）；
- ESC 选择性纠错：supported 率、ECE、E-AURC、Δ50；五种估计器的 E-AURC、Δ50、句内排序准确率。
另报告与 KoLLA 主结果的系统排名一致性（Spearman，按 M² F0.5 与 Δ50）。
用法: python korlang8_analysis.py [--runs DIR] [--kolla JSON]
"""
import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.kolla import load_m2  # noqa: E402
from kgc.eval.calibration import CONF_SOURCES, ece, excess_aurc, score_edits  # noqa: E402
from kgc.eval.gleu_official import GleuCache, _picks, corpus_np  # noqa: E402
from kgc.eval.m2_official import corpus_multi, corpus_single, corpus_stats_for_runs  # noqa: E402
from selective import delta50, within_sentence  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results/runs_korlang8_v2"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--kolla", default=str(ROOT / "results/revision/stats_main.json"))
    ap.add_argument("--kolla_sel", default=str(ROOT / "results/revision/selective.json"))
    ap.add_argument("--out", default=str(ROOT / "results/revision/korlang8.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kor_lang8/korlang8_eval.m2")
    n = len(m2)
    two = [i for i, s in enumerate(m2) if len(s.annotator_groups()) == 2]
    gc, picks = GleuCache(m2), _picks(n)
    res = {}
    for f in sorted(glob.glob(f"{a.runs}/*.jsonl")):
        R = {}
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            R[r["sent_id"]] = r
        if len(R) != n:
            print("跳过（不完整）", f, len(R)); continue
        name = sysname(f)
        hyps = [R[i]["greedy"] for i in range(n)]
        st = corpus_stats_for_runs(m2, hyps)
        P, Rc, F, _ = corpus_multi(st)
        rng = random.Random(1)
        boots = sorted(corpus_multi(st, order=[rng.randrange(n) for _ in range(n)])[2] for _ in range(a.boot))
        g = corpus_np(np.array(gc.sentence_stats(hyps)), picks)
        Fm2 = corpus_multi(st, order=two)[2]
        f0, f1 = corpus_single(st, 0, order=two)[2], corpus_single(st, 1, order=two)[2]
        row = {"P": P, "R": Rc, "F": F, "F_lo": boots[int(0.025 * a.boot)], "F_hi": boots[int(0.975 * a.boot) - 1],
               "GLEU": g, "two_ref": {"F_multi": Fm2, "F_ann0": f0, "F_ann1": f1,
                                      "gain_vs_mean": (Fm2 / ((f0 + f1) / 2) - 1) * 100,
                                      "gain_vs_best": (Fm2 / max(f0, f1) - 1) * 100},
               "conf": {}}
        for src in CONF_SOURCES:
            E = [e for i in range(n) for e in score_edits(m2[i], R[i]["greedy"], R[i].get("samples", []), i,
                                                           source=src, rec=R[i])]
            ws, wn = within_sentence(E)
            row["conf"][src] = {"n_edits": len(E), "supported": sum(e.supported for e in E) / len(E),
                                "ECE": ece(E), "E_AURC": excess_aurc(E), "delta50": delta50(E),
                                "within_acc": ws / wn if wn else None}
        res[name] = row
        c = row["conf"]["esc"]
        print(f"{name:28s} F={F:.4f} [{row['F_lo']:.4f},{row['F_hi']:.4f}] GLEU={g:.4f} | 双参考 gain "
              f"{row['two_ref']['gain_vs_mean']:.1f}% / vs best {row['two_ref']['gain_vs_best']:.1f}% | "
              f"supp={c['supported']:.3f} ECE={c['ECE']:.3f} E-AURC={c['E_AURC']:.3f} Δ50={c['delta50']:.1f}% "
              f"within esc={c['within_acc']:.3f} mintok={row['conf']['mintok']['within_acc']:.3f}", flush=True)

    summary = {}
    try:
        K = json.load(open(a.kolla))["systems"]
        KS = json.load(open(a.kolla_sel))["configs"]
        common = [s for s in res if s in K and s in KS]
        summary["spearman_F"] = sps.spearmanr([K[s]["F"] for s in common], [res[s]["F"] for s in common])._asdict()
        summary["spearman_delta50"] = sps.spearmanr([KS[s]["esc"]["delta50"] for s in common],
                                                    [res[s]["conf"]["esc"]["delta50"] for s in common])._asdict()
        summary["n_common"] = len(common)
        print("与 KoLLA 的排名一致性：", summary)
    except FileNotFoundError:
        pass
    for src in CONF_SOURCES:
        summary.setdefault("median", {})[src] = {k: statistics.median(res[s]["conf"][src][k] for s in res)
                                                 for k in ("ECE", "E_AURC", "delta50", "within_acc")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_sentences": n, "n_two_ref": len(two), "configs": res, "summary": summary},
              open(a.out, "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
