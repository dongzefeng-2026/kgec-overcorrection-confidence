#!/usr/bin/env bash
# 单卡工作循环：逐模型 下载(ModelScope) -> Kor-Lang8 0/4-shot -> 删除权重。
# 用法: GPU=0 bash korlang8_worker.sh <model_id> [<model_id> ...]
# 目录内有 .pdl_complete（下载器在 sha256 全部通过后写入）才视为可用，否则（续传）下载；
# KEEP_MODELS 中的模型跑完不删除。设置与 KoLLA 完全一致。
set -uo pipefail
# FlashInfer 采样内核在本机无法 JIT 编译（CUDA 12.8 工具链不支持 sm_120 的编译目标），
# 改用 vLLM 原生 PyTorch top-k/top-p 采样：同一截断分布上的精确采样，贪心解码不受影响。
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$(dirname "$0")/../.."
source /root/autodl-tmp/kgec/venv/bin/activate
M2=../data/kor_lang8/korlang8_eval.m2
N=$(grep -c '^S ' $M2)
SHM=/dev/shm/models
mkdir -p runs_korlang8 logs_rev
slug() { echo "$1" | tr '/' '_' | tr '.' '-' | tr '[:upper:]' '[:lower:]'; }
for m in "$@"; do
  d=$SHM/$(basename "$m")
  if [ ! -f "$d/.pdl_complete" ]; then
    echo "[$(date +%T)] 下载 $m"
    # HF 与 ModelScope 上这些模型的权重 sha256 逐文件一致（已核对）。HF 经学术加速时段性
    # 限速到 <1 MB/s，故统一用 ModelScope（也与原 14 个配置的权重来源一致）。
    src=${SRC:-ms}
    python scripts/revision/pdl.py "$m" "$d" --src $src > "logs_rev/pdl_$(slug "$m").log" 2>&1 \
      || { echo "[skip] 下载/校验失败 $m"; rm -rf "$d"; continue; }
  fi
  for ns in 0 4; do
    out=runs_korlang8/$(slug "$m")_ns${ns}.jsonl
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$N" ]; then echo "[skip] $out"; continue; fi
    echo "[$(date +%T)] 推理 $m ns=$ns gpu=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python -m kgc.infer.run_vllm --model "$d" --name "$m" \
      --m2 $M2 --out "$out" --n-shot $ns --n-samples 10 --tp 1 --gpu-frac 0.92 \
      --verbalized > "logs_rev/korlang8_$(slug "$m")_ns${ns}.log" 2>&1
    echo "[$(date +%T)] 完成 $out ($(wc -l < "$out" 2>/dev/null || echo 0) 行)"
  done
  case " ${KEEP_MODELS:-Qwen/Qwen3-32B} " in *" $m "*) ;; *) rm -rf "$d" ;; esac
done
