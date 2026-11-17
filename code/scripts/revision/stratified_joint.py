"""分层效应的联合作文置换检验（补充 stratified.py）。

所有配置共用同一批作文，跨模型 Wilcoxon 把模型当作独立重复，忽略了作文抽样的共同波动。
这里在作文层面置换"低/高"标签——同一个置换同时作用于所有配置——检验跨配置平均的
低−高差值（ECE、E-AURC、支持率、平均置信度）。另报告 ECE 的等样本量对照：从高层抽取与
低层同样多的编辑重复计算，用来估计 ECE 的小样本偏差。
"""
import json, random, statistics, sys, glob
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.essay_link import link_sentences
from kgc.data.kolla import load_essays, load_m2
from kgc.data.kolla_audit import kept_indices
from kgc.data.rubric import grammar_score, load_scores
from kgc.eval.calibration import ece, excess_aurc, score_edits
from stratified import tier_fixed
from stats_main import sysname

ROOT = Path(__file__).resolve().parents[3]
m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2"); kept = kept_indices(m2)
links = link_sentences(load_essays(ROOT / "data/kolla_v2"), m2)
gs = grammar_score(load_scores(ROOT / "data/kolla_v2"))
essay_of = {i: links[i]["essay"] for i in kept if links[i]["essay"] in gs}
tier = {k: tier_fixed(v) for k, v in gs.items()}
LOW = sorted(k for k in set(essay_of.values()) if tier[k] == "low")
HIGH = sorted(k for k in set(essay_of.values()) if tier[k] == "high")
cfgs = {}
for f in sorted(glob.glob(str(ROOT / "results/runs_v2/*.jsonl"))):
    if "baseline" in f: continue
    R = {json.loads(l)["sent_id"]: json.loads(l) for l in open(f, encoding="utf-8")}
    by = defaultdict(list)
    for i, k in essay_of.items():
        if tier[k] in ("low", "high"):
            by[k] += score_edits(m2[i], R[i]["greedy"], R[i]["samples"], i)
    cfgs[sysname(f)] = by
STATS = {"ECE": ece, "E_AURC": excess_aurc,
         "support": lambda es: sum(e.supported for e in es) / len(es),
         "mean_conf": lambda es: sum(e.conf for e in es) / len(es)}

def mean_diff(low, high, fn):
    d = []
    for by in cfgs.values():
        a = [e for k in low for e in by[k]]; b = [e for k in high for e in by[k]]
        d.append(fn(a) - fn(b))
    return statistics.mean(d)

rng = random.Random(0); N = 2000
pool = LOW + HIGH
OUTJ = {"n_low_essays": len(LOW), "n_high_essays": len(HIGH), "n_configs": len(cfgs), "n_perm": N, "tests": {}}
for name, fn in STATS.items():
    obs = mean_diff(LOW, HIGH, fn); hits = 0
    for _ in range(N):
        rng.shuffle(pool)
        if abs(mean_diff(pool[:len(LOW)], pool[len(LOW):], fn)) >= abs(obs) - 1e-12: hits += 1
    OUTJ["tests"][name] = {"mean_diff": obs, "p": (hits + 1) / (N + 1)}
    print(f"{name:10s} 跨配置平均 低−高 = {obs:+.4f}   联合作文置换 p = {(hits+1)/(N+1):.4f}", flush=True)
# 样本量与 ECE 小样本偏差
nl = [sum(len(by[k]) for k in LOW) for by in cfgs.values()]; nh = [sum(len(by[k]) for k in HIGH) for by in cfgs.values()]
print("每个配置的编辑数 低层 中位", statistics.median(nl), "高层 中位", statistics.median(nh))
bias = []
for by in cfgs.values():
    hi = [e for k in HIGH for e in by[k]]; nlow = sum(len(by[k]) for k in LOW)
    full = ece(hi); sub = statistics.mean(ece(random.Random(s).sample(hi, min(nlow, len(hi)))) for s in range(50))
    bias.append(sub - full)
print(f"把高层编辑下采样到低层样本量后，ECE 平均上升 {statistics.mean(bias):+.4f}（小样本偏差的量级）")
OUTJ["ece_downsample_bias"] = statistics.mean(bias)
OUTJ["median_edits_low"], OUTJ["median_edits_high"] = statistics.median(nl), statistics.median(nh)
json.dump(OUTJ, open(ROOT / "results/revision/stratified_joint.json", "w"), indent=1)
