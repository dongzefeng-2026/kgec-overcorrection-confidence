"""分层抽取不被参考支持的系统编辑，供人工/LLM 核查（返修 R2.2）。

每个配置抽 PER_CFG 条（seed 固定），只取审计保留句；条目中隐去系统名与置信度（另存映射表）。
输出：results/revision/unsupported_sample.jsonl（判定用）与 unsupported_sample_key.jsonl（映射）。
"""
import glob, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.analysis.edit_types import classify, coarse
from kgc.data.kolla import load_m2
from kgc.data.kolla_audit import kept_indices
from kgc.data.tokenize import detok
from kgc.eval.calibration import score_edits
from kgc.eval.gleu_official import apply_edits
from stats_main import sysname

PER_CFG, SEED = 19, 2026
ROOT = Path(__file__).resolve().parents[3]
m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2"); kept = kept_indices(m2)
items, keys = [], []
rng = random.Random(SEED)
# 标注样本抽自格式规范化之前的 v1 输出（标注中发现的格式故障促成了 v2 规范化），
# 为保证样本可复现，这里固定使用 v1 输出。
files = sorted(glob.glob(str(ROOT / "results/runs/*.jsonl")) + glob.glob(str(ROOT / "results/runs_rev/*.jsonl")))
for f in files:
    if "baseline" in f: continue
    R = {json.loads(l)["sent_id"]: json.loads(l) for l in open(f, encoding="utf-8")}
    pool = []
    for i in kept:
        for e in score_edits(m2[i], R[i]["greedy"], R[i]["samples"], i):
            if not e.supported:
                pool.append((i, e))
    for i, e in rng.sample(pool, PER_CFG):
        s = m2[i]; toks = s.source.split()
        span = " ".join(toks[e.start:e.end])
        items.append({"source": detok(s.source), "ref1": detok(apply_edits(s, 0)), "ref2": detok(apply_edits(s, 1)),
                      "system": detok(R[i]["greedy"]), "edit_from": span or "∅ (insertion)",
                      "edit_to": e.corr or "∅ (deletion)",
                      "edit_context": " ".join(toks[max(0, e.start - 3):e.start] + ["[[", span, "→", e.corr, "]]"] + toks[e.end:e.end + 3])})
        keys.append({"config": sysname(f), "sent_id": i, "start": e.start, "end": e.end, "corr": e.corr,
                     "conf": e.conf, "type": coarse(classify(toks[e.start:e.end], e.corr.split())["type"])})
order = list(range(len(items))); rng.shuffle(order)
out = ROOT / "results/revision"; out.mkdir(parents=True, exist_ok=True)
with open(out / "unsupported_sample.jsonl", "w", encoding="utf-8") as fi, open(out / "unsupported_sample_key.jsonl", "w", encoding="utf-8") as fk:
    for new_id, k in enumerate(order):
        fi.write(json.dumps({"id": new_id, **items[k]}, ensure_ascii=False) + "\n")
        fk.write(json.dumps({"id": new_id, **keys[k]}, ensure_ascii=False) + "\n")
print(len(items), "items")
