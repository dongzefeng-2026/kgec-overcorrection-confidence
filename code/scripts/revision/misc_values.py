"""方法部分用到的几项数值：句子-作文链接率、无需修改句数、修复前后的 supported 比例（v2 输出）。"""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kgc.data.essay_link import link_sentences, summarize
from kgc.data.kolla import load_essays, load_m2
from kgc.data.kolla_audit import kept_indices
from kgc.eval.calibration import _canon, _gold_edits, _norm_corr
from kgc.eval.m2score import merge_edits

ROOT = Path(__file__).resolve().parents[3]
m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2"); kept = kept_indices(m2)
links = link_sentences(load_essays(ROOT / "data/kolla_v2"), m2)
s = summarize(links)
noedit = sum(1 for i in kept if all(e.is_noop for e in m2[i].edits))
json.dump({**s, "n_noedit_kept": noedit}, open(ROOT / "results/revision/link_summary.json", "w"), indent=1)

def old_gold(sent):
    """原（有缺陷的）实现：跨标注者合并同 span，再拼接相邻编辑。"""
    by = {}
    for e in sent.edits:
        if not e.is_noop:
            by.setdefault((e.start, e.end), []).extend(e.corrs)
    return merge_edits([(a, b, list(dict.fromkeys(cs))) for (a, b), cs in by.items()])

tot = old = new = 0
for f in glob.glob(str(ROOT / "results/runs_v2/*.jsonl")):
    if "baseline" in f: continue
    R = {json.loads(l)["sent_id"]: json.loads(l) for l in open(f, encoding="utf-8")}
    for i in kept:
        G0, G1 = old_gold(m2[i]), _gold_edits(m2[i])
        for st, en, c in _canon(m2[i].source, R[i]["greedy"]):
            tot += 1
            old += any(st == a and en == b and c in cs for a, b, cs in G0)
            new += any(st == a and en == b and _norm_corr(c) in cs for a, b, cs in G1)
json.dump({"old": old / tot, "new": new / tot, "n_edits": tot}, open(ROOT / "results/revision/supported_old_new.json", "w"), indent=1)
print(json.dumps({"linked_rate": s["linked_rate"], "n_noedit_kept": noedit, "supported_old": old / tot, "supported_new": new / tot}))
