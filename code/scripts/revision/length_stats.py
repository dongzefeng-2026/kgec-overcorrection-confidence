"""词数统计（返修 R2.2）：源句、两位标注者的参考、各系统输出的 eojeol 数与语素数。

韩语"词"有两种口径：eojeol（空格分隔的语节，即 .m2 文件的 token 单位，标点已独立成 token
的不计入）与语素（Kiwi 形态分析，不计标点）。对每个系统报告：
- 总数及与源句、与参考均值的相对差；
- 逐句与"最接近的参考"的绝对差均值（系统长度是否落在人工改法的范围内）；
- 长度发生变化的句子比例。
用法: ~/.venvs/kgec-rev1/bin/python length_stats.py [--runs DIR ...] [--full]
"""
import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kiwipiepy import Kiwi  # noqa: E402

from kgc.data.kolla import load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.gleu_official import apply_edits  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PUNCT = set(""".,?!;:…()[]{}<>"'“”‘’""")
kiwi = Kiwi()
_cache = {}


def eojeol(s):
    return sum(1 for t in s.split() if not all(ch in PUNCT for ch in t))


def morph(s):
    if s not in _cache:
        _cache[s] = sum(1 for t in kiwi.tokenize(s) if not t.tag.startswith("S"))
    return _cache[s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(ROOT / "results/runs_v2")])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/revision/length_stats.json"))
    a = ap.parse_args()
    m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
    kept = list(range(len(m2))) if a.full else kept_indices(m2)
    src = [m2[i].source for i in kept]
    refs = [[apply_edits(m2[i], ann) for ann in (0, 1)] for i in kept]

    def summary(texts, unit):
        f = eojeol if unit == "eojeol" else morph
        lens = [f(t) for t in texts]
        s_l = [f(t) for t in src]
        r_l = [[f(r) for r in rr] for rr in refs]
        tot, tot_src = sum(lens), sum(s_l)
        tot_ref = statistics.mean(sum(r[k] for r in r_l) for k in (0, 1))
        return {"total": tot, "per_sent": tot / len(lens),
                "vs_source_pct": (tot / tot_src - 1) * 100,
                "vs_ref_mean_pct": (tot / tot_ref - 1) * 100,
                "abs_diff_nearest_ref": statistics.mean(min(abs(x - y) for y in rr) for x, rr in zip(lens, r_l)),
                "changed_len_rate": sum(1 for x, y in zip(lens, s_l) if x != y) / len(lens)}

    out = {"n_sentences": len(kept), "rows": {}}
    out["rows"]["Source"] = {u: summary(src, u) for u in ("eojeol", "morpheme")}
    for k in (0, 1):
        out["rows"][f"Reference {k + 1}"] = {u: summary([r[k] for r in refs], u) for u in ("eojeol", "morpheme")}
    for f in sorted(f for d in a.runs for f in glob.glob(f"{d}/*.jsonl") if "smoke" not in f):
        recs = {json.loads(l)["sent_id"]: json.loads(l) for l in open(f, encoding="utf-8")}
        if len(recs) != len(m2):
            continue
        hyps = [recs[i]["greedy"] for i in kept]
        out["rows"][sysname(f)] = {u: summary(hyps, u) for u in ("eojeol", "morpheme")}
    print(f"{'':28s} {'eojeol':>8s} {'Δsrc%':>7s} {'Δref%':>7s} {'|Δ|ref':>7s} {'chg%':>6s} | {'morph':>7s} {'Δsrc%':>7s} {'Δref%':>7s} {'|Δ|ref':>7s}")
    for name, r in out["rows"].items():
        e, m = r["eojeol"], r["morpheme"]
        print(f"{name:28s} {e['total']:8d} {e['vs_source_pct']:+7.1f} {e['vs_ref_mean_pct']:+7.1f} "
              f"{e['abs_diff_nearest_ref']:7.2f} {100*e['changed_len_rate']:6.1f} | {m['total']:7d} "
              f"{m['vs_source_pct']:+7.1f} {m['vs_ref_mean_pct']:+7.1f} {m['abs_diff_nearest_ref']:7.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
