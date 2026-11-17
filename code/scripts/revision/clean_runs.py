"""对已有运行输出施加 kgc.infer.postprocess.normalize_output，写出 v2 版本并记录触发次数。

用法: python clean_runs.py <in_dir> <out_dir>
原贪心输出保存在 greedy_v1，原采样保存在 samples_v1，便于做"修复前后"的敏感性分析。
"""
import json, sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kgc.infer.postprocess import normalize_output

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
logp = Path(out_dir) / "_parse_fix_counts.json"
log = json.load(open(logp)) if logp.exists() else {}
for f in sorted(glob.glob(f"{src_dir}/*.jsonl")):
    if "smoke" in f:
        continue
    cnt = {"n": 0, "greedy_label_echo": 0, "greedy_arrow": 0, "greedy_markdown": 0, "greedy_identity_mapped": 0,
           "samples_n": 0, "samples_changed": 0}
    out = []
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        cnt["n"] += 1
        g, fl = normalize_output(r["greedy"], r["source"])
        for k, v in fl.items():
            cnt[f"greedy_{k}"] += int(v)
        new_s = []
        for s in r.get("samples", []):
            s2, _ = normalize_output(s, r["source"])
            cnt["samples_n"] += 1
            cnt["samples_changed"] += int(s2 != s)
            new_s.append(s2)
        r["greedy_v1"], r["samples_v1"] = r["greedy"], r.get("samples", [])
        r["greedy"], r["samples"] = g, new_s
        out.append(r)
    with open(Path(out_dir) / os.path.basename(f), "w", encoding="utf-8") as fo:
        for r in out:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    log[os.path.basename(f)] = cnt
json.dump(log, open(logp, "w"), indent=1)
for k, v in log.items():
    print(f"{k:62s} echo={v['greedy_label_echo']:5d} arrow={v['greedy_arrow']:3d} md={v['greedy_markdown']:3d} samples_changed={v['samples_changed']}")
