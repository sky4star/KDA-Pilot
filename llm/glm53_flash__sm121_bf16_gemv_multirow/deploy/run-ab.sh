#!/usr/bin/env bash
# run-ab.sh <0|1>  — parameterized launcher for the GEMV A/B.
#   OVERLAY=0|1   mount the patched unquant + sm120_bf16_gemv, or not
#   SPEC=0|1      enable DSPARK speculative decoding, or not
set -euo pipefail
RANK="${1:?usage: run-ab.sh <0|1>}"
OVERLAY="${OVERLAY:-0}"
SPEC="${SPEC:-1}"
IMAGE="${IMAGE:-sglang-ppmtp:glm53-ppmtp-dspark-v1}"
MODEL="${MODEL:-/models/GLM-5.3-Flash-NVFP4}"
DRAFT="${DRAFT:-/models/GLM-5.3-Flash-DSpark-sglang}"
MF="${MF:-0.90}"
HTTP_PORT="${HTTP_PORT:-8940}"
DIST_PORT="${DIST_PORT:-50110}"
MASTER_ADDR="${MASTER_ADDR:-10.100.8.1}"
NAME="dspark-tp2-rank${RANK}"
APIKEY="${SGLANG_API_KEY:-REPLACE_WITH_API_KEY}"
OV="${OV:-$HOME/glm53-gemv-overlay}"

MOUNTS=()
if [ "$OVERLAY" = "1" ]; then
  MOUNTS+=(-v "$OV/unquant.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/unquant.py:ro")
  MOUNTS+=(-v "$OV/sm120_bf16_gemv.py:/sgl-workspace/sglang/python/sglang/kernels/ops/gemm/sm120_bf16_gemv.py:ro")
  MOUNTS+=(-v "$OV/sm120_bf16_gemv.cuh:/sgl-workspace/sglang/python/sglang/kernels/ops/gemm/sm120_bf16_gemv.cuh:ro")
fi

SPECARGS=()
if [ "$SPEC" = "1" ]; then
  SPECARGS=(--speculative-algorithm DSPARK
            --speculative-draft-model-path "$DRAFT"
            --speculative-dspark-block-size "${DSPARK_GAMMA:-7}")
fi

echo ">> ab rank${RANK}: overlay=${OVERLAY} spec=${SPEC}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  --memory 115g --memory-swap 115g \
  -v "$HOME/models:/models" \
  -v "$HOME/glm53-dspark-tp2/dbg:/dspark_dbg" \
  -v "$HOME/profiles:$HOME/profiles" \
  "${MOUNTS[@]}" \
  -e SGL_DS_TEST1="${SGL_DS_TEST1:-}" \
  -e SGL_DS_DIR=/dspark_dbg \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e TP_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name dspark \
    --api-key "$APIKEY" \
    --trust-remote-code \
    --tp-size 2 --pp-size 1 --nnodes 2 --node-rank "$RANK" \
    --dist-init-addr "${MASTER_ADDR}:${DIST_PORT}" \
    --attention-backend dsa \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 \
    --disable-shared-experts-fusion \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    "${SPECARGS[@]}" \
    --mem-fraction-static "$MF" \
    --mamba-full-memory-ratio "${MAMBA_RATIO:-0.9}" \
    --context-length 160000 \
    --chunked-prefill-size "${CHUNKED:-512}" \
    --cuda-graph-max-bs-decode 4 \
    --max-running-requests 4 \
    --disable-flashinfer-autotune \
    --stream-interval 1 --sleep-on-idle \
    --host 0.0.0.0 --port "$HTTP_PORT"
