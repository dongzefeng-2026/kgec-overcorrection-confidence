"""构建 Kor-Lang8 外部验证集（返修 R2.4）：平行句对 -> .m2 文件 + 清单。

数据：Kor-Lang8（Yoon et al., 2023）——韩语学习者在 Lang-8 上的写作与其他用户的修改，
由 KAGAS 作者的公开脚本从 NAIST Lang-8 语料筛得 13,080 对（与其 README 日志一致）。
本文使用的副本见 data/kor_lang8/raw/kor_lang8.txt（md5 22cf8396a331bc10e648fa133d57b287）。

预先固定的规则（在看任何模型结果之前写定）：
1. 源句与修改句统一空白，并用与 KoLLA 相同的 retok 还原到 token 网格；
2. 剔除长度比（修改句/源句的 token 数）不在 [0.5, 2.0] 内的修改句——Lang-8 中存在
   截断（"언니는 ."）和整段改写，它们不是语法纠错参考；
3. 剔除 token 数 > 60 的源句（多句拼接，且可能超出生成长度上限）；
4. 同一源句的不同修改句视为不同"标注者"，每句至多保留 2 条（随机抽取），与 KoLLA 的
   双参考设计对应；
5. 抽样：全部有 2 条参考的源句 + 随机单参考源句，凑足 N_TOTAL 句（seed 固定）。

参考编辑由与系统输出相同的 eojeol 级 LCS 对齐抽取，相邻编辑合并。M² scorer 在打分时
会在编辑格上搜索与参考最匹配的切分，因而对参考编辑的粒度不敏感。
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kgc.data.tokenize import retok  # noqa: E402
from kgc.eval.m2score import merge_edits  # noqa: E402

N_TOTAL = 3000
MAX_REFS = 2
LEN_RATIO = (0.5, 2.0)
MAX_TOKENS = 60
SEED = 2026


def gold_edits(src: str, tgt: str):
    s, t = src.split(), tgt.split()
    items = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, t, autojunk=False).get_opcodes():
        if tag != "equal":
            items.append((i1, i2, [" ".join(t[j1:j2]) or "-NONE-"]))
    return [(a, b, c[0]) for a, b, c in merge_edits(items)]


def main():
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(root / "data/kor_lang8/raw/kor_lang8.txt"))
    ap.add_argument("--out", default=str(root / "data/kor_lang8"))
    a = ap.parse_args()

    stats = OrderedDict()
    pairs = []
    for ln, line in enumerate(open(a.raw, encoding="utf-8")):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue
        pairs.append((ln, retok(parts[0]), retok(parts[1])))
    stats["pairs_raw"] = len(pairs)

    by_src: "OrderedDict[str, list]" = OrderedDict()
    for ln, s, t in pairs:
        by_src.setdefault(s, [])
        if t != s and all(t != x[1] for x in by_src[s]):
            by_src[s].append((ln, t))
    stats["unique_sources"] = len(by_src)

    kept = OrderedDict()
    drop_len = drop_ratio = 0
    for s, refs in by_src.items():
        n = len(s.split())
        if n > MAX_TOKENS:
            drop_len += 1
            continue
        ok = [(ln, t) for ln, t in refs if LEN_RATIO[0] <= len(t.split()) / max(1, n) <= LEN_RATIO[1]]
        drop_ratio += len(refs) - len(ok)
        if ok:
            kept[s] = ok
    stats["sources_dropped_too_long"] = drop_len
    stats["targets_dropped_length_ratio"] = drop_ratio
    stats["sources_after_filter"] = len(kept)

    rng = random.Random(SEED)
    multi = [s for s, r in kept.items() if len(r) >= 2]
    single = [s for s, r in kept.items() if len(r) == 1]
    stats["sources_with_2plus_refs"] = len(multi)
    chosen = multi + rng.sample(single, max(0, N_TOTAL - len(multi)))
    rng.shuffle(chosen)
    stats["sampled"] = len(chosen)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    with open(out / "korlang8_eval.m2", "w", encoding="utf-8") as f:
        for i, s in enumerate(chosen):
            refs = kept[s]
            if len(refs) > MAX_REFS:
                refs = rng.sample(refs, MAX_REFS)
            f.write(f"S {s}\n")
            for ann, (_ln, t) in enumerate(refs):
                es = gold_edits(s, t)
                if not es:
                    f.write(f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{ann}\n")
                for a_, b_, c in es:
                    f.write(f"A {a_} {b_}|||UNK|||{c}|||REQUIRED|||-NONE-|||{ann}\n")
            f.write("\n")
            manifest.append({"sent_id": i, "raw_lines": [ln for ln, _ in refs], "n_refs": len(refs)})
    stats["sampled_2_refs"] = sum(1 for m in manifest if m["n_refs"] == 2)
    json.dump({"rules": {"N_TOTAL": N_TOTAL, "MAX_REFS": MAX_REFS, "LEN_RATIO": LEN_RATIO,
                         "MAX_TOKENS": MAX_TOKENS, "SEED": SEED},
               "stats": stats, "sentences": manifest},
              open(out / "korlang8_eval.manifest.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
