"""评判者 B：用 Qwen3-32B 按同一份指南独立判定未匹配编辑（返修 R2.2）。

与评判者 A 相同：只给源句、两条参考、系统完整输出和该编辑，不给系统名与置信度。
贪心解码，关闭思考模式；要求只输出一个 JSON 对象。解析失败时记为 None 并在汇总中报告。
用法（服务器）: python judge_qwen.py --model /dev/shm/models/Qwen3-32B \
    --items results/revision/unsupported_sample.jsonl --guide scripts/revision/annotation_guideline.md \
    --out results/revision/judge_B_qwen32b.jsonl
"""
import argparse
import json
import re

LABELS = ("VALID", "UNNECESSARY", "ERROR", "GRANULARITY")


def build(guide: str, it: dict) -> list:
    user = (
        f"{guide}\n\n---\nItem {it['id']}\n"
        f"SOURCE: {it['source']}\nREF-1: {it['ref1']}\nREF-2: {it['ref2']}\n"
        f"SYSTEM: {it['system']}\nEDIT: {it['edit_from']} → {it['edit_to']}\n"
        f"EDIT IN CONTEXT: {it['edit_context']}\n\n"
        "Answer with one JSON object only, e.g. "
        '{"id": 0, "label": "UNNECESSARY", "register": false, "rationale": "..."}'
    )
    return [{"role": "system", "content": "You are an experienced teacher of Korean as a second language "
             "and an annotator for grammatical error correction research."},
            {"role": "user", "content": user}]


def parse(text: str, item_id: int):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            lab = str(d.get("label", "")).upper()
            if lab in LABELS:
                return {"id": item_id, "label": lab, "register": bool(d.get("register", False)),
                        "rationale": str(d.get("rationale", ""))}
        except json.JSONDecodeError:
            pass
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", text.upper()):
            return {"id": item_id, "label": lab, "register": None, "rationale": "(label recovered from text)"}
    return {"id": item_id, "label": None, "register": None, "rationale": text[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--guide", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    guide = open(a.guide, encoding="utf-8").read()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92)
    prompts = [tok.apply_chat_template(build(guide, it), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for it in items]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
    n_fail = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for it, o in zip(items, outs):
            r = parse(o.outputs[0].text, it["id"])
            r["raw"] = o.outputs[0].text
            n_fail += r["label"] is None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[judge] {len(items)} 条，解析失败 {n_fail}")


if __name__ == "__main__":
    main()
