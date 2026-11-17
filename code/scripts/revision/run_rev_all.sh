#!/usr/bin/env bash
# 返修推理总调度（3 卡）：
#   GPU0：Kor-Lang8 上的 Midm / Kanana / A.X / EXAONE
#   GPU1、GPU2：先跑 Qwen3-32B 在 KoLLA 上的 0/4-shot，
#               之后 GPU1 跑 Kor-Lang8 上的 HyperCLOVAX / Qwen3-8B / Qwen3-14B，GPU2 跑 Kor-Lang8 上的 Qwen3-32B
set -uo pipefail
# 见 korlang8_worker.sh：本机改用 vLLM 原生采样器
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$(dirname "$0")/../.."
source /root/autodl-tmp/kgec/venv/bin/activate
D=/dev/shm/models/Qwen3-32B
echo "[$(date +%T)] 校验 Qwen3-32B 全部分片 sha256"
python scripts/revision/pdl.py Qwen/Qwen3-32B $D --src hf --seg 24 || { echo "Qwen3-32B 校验失败"; exit 1; }

GPU=0 bash scripts/revision/korlang8_worker.sh K-intelligence/Midm-2.0-Base-Instruct \
  kakaocorp/kanana-1.5-8b-instruct-2505 skt/A.X-4.0-Light LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  > logs_rev/worker_gpu0.log 2>&1 &

mkdir -p runs_rev logs_rev
CUDA_VISIBLE_DEVICES=1 python -m kgc.infer.run_vllm --model $D --name Qwen/Qwen3-32B \
  --m2 ../data/kolla_v2/KoLLA_multi-refs.m2 --data-dir ../data/kolla_v2 --out runs_rev/smoke_q32.jsonl \
  --n-shot 0 --n-samples 2 --tp 1 --gpu-frac 0.92 --limit 30 --verbalized > logs_rev/smoke_q32.log 2>&1
echo "[$(date +%T)] smoke rc=$? 行数=$(wc -l < runs_rev/smoke_q32.jsonl 2>/dev/null)"
GPUS="1 2" bash scripts/revision/run_qwen32b.sh
echo "[$(date +%T)] Qwen3-32B KoLLA 完成"

GPU=1 bash scripts/revision/korlang8_worker.sh naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B \
  Qwen/Qwen3-8B Qwen/Qwen3-14B > logs_rev/worker_gpu1.log 2>&1 &
GPU=2 bash scripts/revision/korlang8_worker.sh Qwen/Qwen3-32B > logs_rev/worker_gpu2.log 2>&1 &
wait
echo "[$(date +%T)] 全部完成"
ls -la runs_rev runs_korlang8
