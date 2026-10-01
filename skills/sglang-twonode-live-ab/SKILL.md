---
name: sglang-twonode-live-ab
description: Use to A/B a patched SGLang change on the live 2-node DGX Spark service: stop/restart both nodes, bind-mount an overlay of patched source over the image, run standalone work in a dev container, poll health, capture traces, and roll back. Triggers include "restart the server", "overlay mount", "rank0/rank1", "dspark-tp2", "live A/B", "线上服务", "重新起服务".
---

# Live A/B on the 2-node DGX Spark SGLang service

The production service is a `docker run` pair (no compose), one container per
node, `tp=2 / pp=1 / nnodes=2`. The SGLang source is **baked into the image**, so
change it by bind-mounting files over the in-image path — never edit inside the
container and expect it to survive a restart.

## Topology (verify each time)

```
spark1 (this host)  rank0  MASTER_ADDR=10.100.8.1:50110  HTTP :8940
spark2 (10.100.8.2) rank1
image:  sglang-ppmtp:glm53-ppmtp-dspark-v1
container names: dspark-tp2-rank0 / dspark-tp2-rank1
source in image: /sgl-workspace/sglang/python/sglang
api key: pass via SGLANG_API_KEY (do NOT commit it)
```

`~/glm53-dspark-tp2/run-dspark-tp2.sh <0|1>` is the canonical raw launcher on
each node; `run-dspark-tp2-overlay.sh` is the same with mounts;
`~/glm53-gemv-overlay/run-ab.sh <0|1>` is the parameterized `OVERLAY=0|1 SPEC=0|1`
variant used for controlled A/B. Copy the scripts + overlay dir to spark2
(`scp -r`) so both nodes mount identical files.

## Start / restart (both nodes, in this order)

```bash
docker stop dspark-tp2-rank0; ssh spark2 'docker stop dspark-tp2-rank1'
sleep 4
ssh spark2 'OVERLAY=1 SPEC=1 nohup bash ~/glm53-gemv-overlay/run-ab.sh 1 > ~/ab_r1.log 2>&1 &'
OVERLAY=1 SPEC=1 nohup bash ~/glm53-gemv-overlay/run-ab.sh 0 > ~/ab_r0.log 2>&1 &
```

Wait for readiness (model ~100GB load ⇒ **8–10 min**; the first HTTP 200 trails
several 503s; `/health` not `/get_server_info`):

```bash
KEY=$SGLANG_API_KEY
for i in $(seq 1 50); do
  c=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
      http://10.100.8.1:8940/health 2>/dev/null || echo 000)
  [ "$c" = 200 ] && break; sleep 15
done
```

`--sleep-on-idle` means the first request after idle pays a wakeup; warm up
before timing.

## Overlay mount pattern

```bash
docker run ... \
  -v "$OV/unquant.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/unquant.py:ro" \
  -v "$OV/foo.py:/sgl-workspace/sglang/python/sglang/kernels/ops/gemm/foo.py:ro" \
  -v "$OV/foo.cuh:/sgl-workspace/sglang/python/sglang/kernels/ops/gemm/foo.cuh:ro" \
  "$IMAGE" python3 -m sglang.launch_server ...
```

- Mount **files**, `:ro`. New files need no image rebuild; rollback is just the
  original script.
- If a new JIT kernel's `.cuh` lives beside its `.py`, `load_jit` with
  `cuda_files=[<abs path>]` works; the in-tree convention is `cuda_files=["gemm/x.cuh"]`
  resolved against `csrc/`.
- Verify the JIT built the expected variants:
  `docker exec dspark-tp2-rank0 bash -lc 'ls -d /root/.cache/sglang/jit/*/sgl_*foo*'`.

## Standalone kernel work (needs the GPU memory)

The live service holds ~103/121 GB unified memory. To benchmark a kernel in
isolation, start a dev container from the same image after stopping the service
(or accept that benchmarks will OOM):

```bash
docker run -d --name gemv-dev --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v $HOME/kda/KDA-Pilot:/kda "$IMAGE" sleep infinity
docker exec gemv-dev bash -lc 'cd /kda/<task> && python3 bench/benchmark.py'
docker stop gemv-dev   # free memory again before restarting the service
```

## Traces / profiling

See `spec-decode-ab`. Capture with `POST /start_profile`, `docker cp` the
`/tmp/*-TP-0.trace.json.gz` out, then parse kernel families and `step[...]`
annotations. `/home/spark/profiles/` is bind-mounted into the server container.

## Don't forget

- Restore the service when done; confirm `/health` 200 on both nodes.
- Never commit the API key. Keep it in `SGLANG_API_KEY`.
- GB10 thermal/clock drift makes cross-start absolute numbers unreliable; prefer
  same-run A/B and matched launch counts.
