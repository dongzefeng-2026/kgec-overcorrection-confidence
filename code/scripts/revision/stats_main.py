"""返修主表 + 作文聚类统计（R2.1 指标、R2.3 统计）。

对每个系统输出：
- 官方 M²（语料级）多参考 P/R/F0.5，作文聚类 bootstrap 95% CI；
- 单参考 F0.5（annotator 0、annotator 1 及二者平均），多参考增益（相对平均 / 相对较好者）；
- GLEU（官方实现，500 次迭代）及 CI；
- 句级平均 F0.5（原文口径，次要指标）；
- 无需修改句（两位标注者都未改）上的误改率及 CI；提出的编辑数。
配对比较（作文聚类近似随机化检验 + 配对 bootstrap 差值 CI，Holm 校正）：
- 每个模型 4-shot vs 0-shot；
- 同规模多语言 vs 韩语专精：Qwen3-8B vs EXAONE-3.5-7.8B、vs A.X-4.0-Light；
- 规模：Qwen3-32B vs 14B vs 8B（若有 32B 输出）。

用法: python stats_main.py [--runs DIR ...] [--full] [--boot 1000] [--perm 2000]
--full 使用原始 1,418 句（敏感性分析）；默认按 kolla_audit 排除 25 条。
"""
import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kgc.data.essay_link import link_sentences  # noqa: E402
from kgc.data.kolla import load_essays, load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.gleu_official import GleuCache, _picks, corpus_np  # noqa: E402
from kgc.eval.m2_official import corpus_multi, corpus_single, corpus_stats_for_runs  # noqa: E402
from kgc.eval.m2score import score_sentence  # noqa: E402
from kgc.eval.stats import cluster_bootstrap, holm, paired_bootstrap, paired_randomization  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SHORT = {
    "k-intelligence_midm-2-0-base-instruct": "Midm-2.0",
    "kakaocorp_kanana-1-5-8b-instruct-2505": "Kanana-1.5-8B",
    "skt_a-x-4-0-light": "A.X-4.0-Light",
    "lgai-exaone_exaone-3-5-7-8b-instruct": "EXAONE-3.5-7.8B",
    "naver-hyperclovax_hyperclovax-seed-text-instruct-1-5b": "HyperCLOVAX-1.5B",
    "qwen_qwen3-8b": "Qwen3-8B",
    "qwen_qwen3-14b": "Qwen3-14B",
    "qwen_qwen3-32b": "Qwen3-32B",
    "baseline_identity": "No-op",
}


def sysname(path):
    b = os.path.basename(path)[:-6]
    for k, v in SHORT.items():
        if b.startswith(k):
            ns = b.rsplit("_ns", 1)[1] if "_ns" in b else "-"
            return v if v == "No-op" else f"{v} ({ns}-shot)"
    return b


_CTX = {}


def _run_contrast(c):
    """单组配对比较（在子进程中运行）：近似随机化检验 + 配对 bootstrap，M² F0.5 与 GLEU。"""
    fam, x, y = c
    S, kept, clusters, picks, n = (_CTX[k] for k in ("S", "kept", "clusters", "picks", "n"))
    sx, sy = S[x]["m2"], S[y]["m2"]
    gx, gy = S[x]["gleu"], S[y]["gleu"]

    def pair_m2(swap):
        A = [sy[kept[j]] if swap[j] else sx[kept[j]] for j in range(n)]
        B = [sx[kept[j]] if swap[j] else sy[kept[j]] for j in range(n)]
        return corpus_multi(A)[2], corpus_multi(B)[2]

    def pair_gleu(swap):
        w = np.asarray(swap)[:, None, None]
        A = np.where(w, gy[kept], gx[kept])
        B = np.where(w, gx[kept], gy[kept])
        return corpus_np(A, picks[:, kept]), corpus_np(B, picks[:, kept])

    d_m2, p_m2 = paired_randomization(pair_m2, clusters, _CTX["perm"], seed=11)
    _, lo_m2, hi_m2 = paired_bootstrap(_CTX["m2F"](sx), _CTX["m2F"](sy), clusters, _CTX["boot"], seed=12)
    d_g, p_g = paired_randomization(pair_gleu, clusters, _CTX["perm"], seed=13)
    _, lo_g, hi_g = paired_bootstrap(_CTX["gleu"](gx), _CTX["gleu"](gy), clusters, _CTX["boot"], seed=14)
    return dict(family=fam, a=x, b=y, dF=d_m2, dF_lo=lo_m2, dF_hi=hi_m2, p_F=p_m2,
                dGLEU=d_g, dGLEU_lo=lo_g, dGLEU_hi=hi_g, p_GLEU=p_g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "results/runs_v2")])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results/revision/stats_main.json"))
    a = ap.parse_args()

    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = list(range(len(m2))) if a.full else kept_indices(m2)
    links = link_sentences(load_essays(ROOT / "data/kolla_v2"), m2)
    clusters = [links[i]["cluster"] for i in kept]
    noedit = {i for i, s in enumerate(m2) if all(e.is_noop for e in s.edits)}
    gc = GleuCache(m2)
    picks = _picks(len(m2))
    print(f"句子 {len(kept)} / {len(m2)}；聚类 {len(set(clusters))}；无需修改句 {len(noedit & set(kept))}")

    files = sorted(f for d in a.runs for f in glob.glob(f"{d}/*.jsonl")
                   if "smoke" not in os.path.basename(f))
    S = {}
    for f in files:
        recs = sorted((json.loads(l) for l in open(f, encoding="utf-8")), key=lambda r: r["sent_id"])
        if len(recs) != len(m2):
            print("跳过（不完整）", f, len(recs)); continue
        hyps = [r["greedy"] for r in recs]
        name = sysname(f)
        S[name] = {
            "m2": corpus_stats_for_runs(m2, hyps),
            "gleu": np.array(gc.sentence_stats(hyps)),
            "fa": [1.0 if hyps[i].split() != m2[i].source.split() else 0.0 for i in range(len(m2))],
            "sentF": [score_sentence(m2[i], hyps[i], "best")["F"] for i in range(len(m2))],
        }

    def m2F(st):
        return lambda idx: corpus_multi(st, order=[kept[j] for j in idx])[2]

    def gleu(arr):
        return lambda idx: corpus_np(arr, picks, [kept[j] for j in idx])

    def far(fa):
        def f(idx):
            v = [fa[kept[j]] for j in idx if kept[j] in noedit]
            return sum(v) / len(v) if v else float("nan")
        return f

    rows = {}
    for name, d in S.items():
        st = d["m2"]
        order = kept
        P, R, F, _ = corpus_multi(st, order=order)
        f0 = corpus_single(st, 0, order=order)[2]
        f1 = corpus_single(st, 1, order=order)[2]
        _, flo, fhi = cluster_bootstrap(m2F(st), clusters, a.boot, seed=1)
        g, glo, ghi = cluster_bootstrap(gleu(d["gleu"]), clusters, a.boot, seed=2)
        fa, falo, fahi = cluster_bootstrap(far(d["fa"]), clusters, a.boot, seed=3)
        rows[name] = dict(P=P, R=R, F=F, F_lo=flo, F_hi=fhi, F_ann0=f0, F_ann1=f1,
                          gain_vs_mean=(F / ((f0 + f1) / 2) - 1) * 100 if f0 + f1 else None,
                          gain_vs_best=(F / max(f0, f1) - 1) * 100 if max(f0, f1) else None,
                          GLEU=g, GLEU_lo=glo, GLEU_hi=ghi,
                          sentF=statistics.mean(d["sentF"][i] for i in kept),
                          false_alarm=fa, FA_lo=falo, FA_hi=fahi,
                          proposed=int(sum(max(p for _c, p, _g in st[i].values()) for i in kept)))
        print(f"{name:28s} F={F:.4f} [{flo:.4f},{fhi:.4f}]  GLEU={g:.4f} [{glo:.4f},{ghi:.4f}]  "
              f"FA={fa:.3f} [{falo:.3f},{fahi:.3f}]  F0={f0:.4f} F1={f1:.4f}", flush=True)

    # ---- 配对比较 ----
    contrasts = []
    for base in sorted({n.split(" (")[0] for n in S if " (" in n}):
        a0, a4 = f"{base} (0-shot)", f"{base} (4-shot)"
        if a0 in S and a4 in S:
            contrasts.append(("prompt", a4, a0))
    for ns in ("0", "4"):
        q = f"Qwen3-8B ({ns}-shot)"
        for k in ("EXAONE-3.5-7.8B", "A.X-4.0-Light"):
            if q in S and f"{k} ({ns}-shot)" in S:
                contrasts.append(("multilingual_vs_korean", q, f"{k} ({ns}-shot)"))
        for big, small in (("Qwen3-32B", "Qwen3-14B"), ("Qwen3-14B", "Qwen3-8B"), ("Qwen3-32B", "Qwen3-8B")):
            if f"{big} ({ns}-shot)" in S and f"{small} ({ns}-shot)" in S:
                contrasts.append(("scale", f"{big} ({ns}-shot)", f"{small} ({ns}-shot)"))

    n = len(kept)
    tests = []
    if a.full:            # 全量 1,418 句只作敏感性分析：报告点估计与区间，不重复配对检验
        contrasts = []
    # 各组对比相互独立：fork 出多个进程并行（子进程继承下面这些全局量，无需序列化大对象）
    global _CTX
    _CTX = dict(S=S, kept=kept, clusters=clusters, picks=picks, n=n, perm=a.perm, boot=a.boot,
                m2F=m2F, gleu=gleu)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(len(contrasts), max(1, mp.cpu_count() - 2)) or 1) as pool:
        for t in pool.imap(_run_contrast, contrasts):
            tests.append(t)
            print(f"[{t['family']}] {t['a']} − {t['b']}: ΔF={t['dF']:+.4f} [{t['dF_lo']:+.4f},{t['dF_hi']:+.4f}] "
                  f"p={t['p_F']:.4f} | ΔGLEU={t['dGLEU']:+.4f} [{t['dGLEU_lo']:+.4f},{t['dGLEU_hi']:+.4f}] "
                  f"p={t['p_GLEU']:.4f}", flush=True)
    for fam in {t["family"] for t in tests}:
        idx = [k for k, t in enumerate(tests) if t["family"] == fam]
        for key in ("p_F", "p_GLEU"):
            adj = holm([tests[k][key] for k in idx])
            for k, v in zip(idx, adj):
                tests[k][key + "_holm"] = v

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_sentences": n, "n_clusters": len(set(clusters)), "full": a.full,
               "systems": rows, "tests": tests}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("写入", a.out)


if __name__ == "__main__":
    main()
