"""在租用的多卡机上批量跑韩语 GEC 推理（vLLM）。

产出 JSONL，每行一句：
  {sent_id, source, group, greedy, samples: [...], n_shot, model}

置信度设计（论文方法贡献）：
- greedy 输出作为系统修正；
- 另采样 N 条（temperature>0）用于**编辑级自洽置信度**：
  某条编辑在 N 个样本中出现的频率 = 该编辑的置信度。
  相比序列 logprob，它与 GEC 的编辑粒度天然对齐，且可直接用于选择性纠错阈值。

用法（租卡机上）：
  python -m kgc.infer.run_vllm --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
      --m2 data/kolla_v2/KoLLA_multi-refs.m2 --data-dir data/kolla_v2 \
      --out runs/exaone_zs.jsonl --n-shot 0 --n-samples 10 --tp 1
断点续跑：已存在的 out 文件中的 sent_id 会被跳过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kgc.data.groups import assign_groups
from kgc.data.kolla import load_essays, load_m2
from kgc.data.tokenize import detok, retok
from kgc.infer.prompts import build_messages, clean_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None,
                    help="写入输出的模型 ID；--model 为本地路径时用它记录规范名")
    ap.add_argument("--m2", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="KoLLA CSV 目录，用于学习者分组；外部数据集（无分组）可省略")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-shot", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=10, help="自洽采样条数；0 关闭")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="上下文上限。本任务提示<800token、输出<=256，2048 足够；"
                         "不限制会让大模型因 KV cache 不足而启动失败")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 句（冒烟用）")
    ap.add_argument("--verbalized", action="store_true",
                    help="额外跑一遍口头置信度（置信度基线之一）")
    args = ap.parse_args()

    m2 = load_m2(args.m2)
    if args.data_dir:
        groups = assign_groups(load_essays(args.data_dir), m2)
    else:
        groups = [(None, "none")] * len(m2)
    if args.limit:
        m2, groups = m2[: args.limit], groups[: args.limit]

    done = set()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        with open(outp, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sent_id"])
                except Exception:
                    pass
        print(f"[resume] 已完成 {len(done)} 句，跳过", flush=True)

    todo = [(i, s, g) for i, (s, (g, _src)) in enumerate(zip(m2, groups))
            if i not in done]
    if not todo:
        print("[done] 无待处理句子")
        return
    print(f"[info] 待推理 {len(todo)} 句 | model={args.model} n_shot={args.n_shot} "
          f"n_samples={args.n_samples}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_frac, trust_remote_code=True,
              dtype="bfloat16", max_model_len=args.max_model_len)

    def chat(msgs):
        """套用对话模板。Qwen3 等混合推理模型默认开启 thinking，会先生成
        <think>...</think> 长链再作答：既让本任务慢 20 倍以上，又会被
        max_tokens 截断成无法解析的残缺输出。故在支持该参数的模型上显式关闭。"""
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    prompts = [chat(build_messages(detok(s.source), args.n_shot))
               for _i, s, _g in todo]

    greedy = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=1))

    sampled = None
    if args.n_samples > 0:
        sampled = llm.generate(prompts, SamplingParams(
            temperature=args.temperature, top_p=0.95, n=args.n_samples,
            max_tokens=args.max_tokens, seed=1234))

    # 口头置信度基线：直接问模型对自己修正的把握（0-100）。
    verbal = None
    if args.verbalized:
        vprompts = []
        for k, (_i, s, _g) in enumerate(todo):
            hyp, _st = clean_output(greedy[k].outputs[0].text, detok(s.source))
            msgs = build_messages(s.source, args.n_shot)
            msgs.append({"role": "assistant", "content": hyp})
            msgs.append({"role": "user", "content":
                         "방금 교정에 대한 확신도를 0에서 100 사이의 숫자로만 답하세요."})
            vprompts.append(chat(msgs))
        verbal = llm.generate(vprompts, SamplingParams(
            temperature=0.0, max_tokens=8))

    with open(outp, "a", encoding="utf-8") as f:
        for k, (i, s, g) in enumerate(todo):
            g_out = greedy[k].outputs[0]
            _g_raw, _g_st = clean_output(g_out.text, detok(s.source))
            _g_txt = retok(_g_raw)
            rec = {
                "sent_id": i,
                "source": s.source,
                "group": g,
                "model": args.name or args.model,
                "n_shot": args.n_shot,
                "greedy": _g_txt,
                "parse_status": ("no_change" if _g_txt.strip() == s.source.strip()
                                 else ("fallback" if _g_st == "fallback" else "ok")),
                "greedy_raw": g_out.text,
                "greedy_cumlogprob": g_out.cumulative_logprob,
                "greedy_ntok": len(g_out.token_ids),
                # 逐 token (文本, logprob)：供 logprob 类置信度基线在离线阶段
                # 按字符偏移定位到具体编辑，无需重跑推理。
                "greedy_tokens": _token_logprobs(g_out, tok),
                "samples": ([retok(clean_output(o.text, detok(s.source))[0])
                             for o in sampled[k].outputs] if sampled else []),
            }
            if verbal is not None:
                rec["verbalized_raw"] = verbal[k].outputs[0].text.strip()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] 写入 {outp}", flush=True)


def _token_logprobs(out, tok):
    """提取 [(token_text, logprob), ...]；vLLM 版本差异时安全降级为空表。"""
    try:
        pairs = []
        for tid, lp in zip(out.token_ids, out.logprobs or []):
            entry = lp.get(tid) if isinstance(lp, dict) else None
            val = getattr(entry, "logprob", None) if entry is not None else None
            pairs.append([tok.decode([tid]), val])
        return pairs
    except Exception:
        return []


if __name__ == "__main__":
    main()
