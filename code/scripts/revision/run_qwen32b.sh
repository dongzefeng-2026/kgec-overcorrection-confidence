#!/usr/bin/env bash
# 返修：Qwen3-32B（bf16）在 KoLLA v2 上的 0-shot / 4-shot，设置与原 14 个配置完全一致。
# 权重在 /dev/shm（数据盘不够放 65GB）。两个 n_shot 各占一张卡并行。
set -uo pipefail
# 见 korlang8_worker.sh：本机改用 vLLM 原生采样器
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$(dirname "$0")/../.."
source /root/autodl-tmp/kgec/venv/bin/activate
M=/dev/shm/models/Qwen3-32B
DATA=../data/kolla_v2
mkdir -p runs_rev logs_rev
GPUS=(${GPUS:-1 2})
k=0
for ns in 0 4; do
  out=runs_rev/qwen_qwen3-32b_ns${ns}.jsonl
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} python -m kgc.infer.run_vllm \
    --model $M --name Qwen/Qwen3-32B --m2 $DATA/KoLLA_multi-refs.m2 --data-dir $DATA \
    --out $out --n-shot $ns --n-samples 10 --tp 1 --gpu-frac 0.92 \
    --verbalized > logs_rev/qwen_qwen3-32b_ns${ns}.log 2>&1 &
  k=$((k+1))
done
wait
wc -l runs_rev/*.jsonl
