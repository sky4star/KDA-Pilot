# Incident: long-prefill profile capture crashed the server

## Symptom
`prefill_capture.py --len 32768` hung on the host (>300 s), and the scheduler
container `dspark-tp2-rank0` (and rank1) later showed `Exited (255)`.

## What actually happened (from the server log)
1. The cold 32K prefill **ran normally**: 64 chunk-forwards at ~1070–1100 tok/s,
   last `Prefill batch` + `POST /generate 200 OK` at 16:38:19.
2. The host then sent `POST /stop_profile`; the server logged `Stop profiling...`
   at 16:38:20 and **the process died (ExitCode=255, OOMKilled=false)** while
   serialising the trace.
3. Because the server died mid-request, the host's `urllib` call to
   `/stop_profile` never got a response → **the host script blocked** (its 3600 s
   timeout) → looked like a hang.

## Root cause
**Profiling a full long prefill produces a trace too large to export.**
A 32K prefill = 64 prefill forwards, each a full model forward with hundreds of
kernels (~10⁵–10⁶ events), plus `detailed_annotations`. The torch-profiler
chrome-trace export blew up and the scheduler exited 255.

The *request length itself is not the problem* — the prefill completed. The
problem is capturing **every** prefill chunk.

## Not
- Not an OOM-kill (`OOMKilled=false`).
- Not the prefill compute (it finished).

## Safe prefill profiling rules (for next time)
- **Limit the number of captured forwards**: `ProfileReq.num_steps` (e.g. 4–8),
  or `start_step` to sample only the high-context chunks.
- **Disable `detailed_annotations`** for large captures.
- Keep the host HTTP timeouts short for `/stop_profile` and treat a non-response
  as "server likely died".
- For full-context prefill cost, prefer **timing only** (no trace), then profile
  a small sampled window.

## Status
Service restarted on both nodes (overlay=1, spec=1); waiting for health.

---

## Incident #2 (after restart): profiling any prefill OOM-kills the scheduler
- Log: `[01:34:44] Subprocess scheduler_0 (pid=132) crashed with exit code -9.
  Triggering SIGQUIT for cleanup...` (exit -9 = SIGKILL = OOM).
- Trigger: `start_profile(num_steps=6)` + an 8K cold prefill; the generate
  returned HTTP 500 and the scheduler was OOM-killed.
- Cause: the box has only **~5 GB free** after the 100 GB model loads
  (`free -g`: used 116/121). The profiler buffer + prefill activations exceed it.

## Incident #3: spark2 (rank1) becomes unreachable
- After the crash the rank1 container was removed/restarted; spark2 now answers
  **ping** and **TCP :22 is open**, but the **SSH banner never arrives**
  (`Connection timed out during banner exchange`; raw `head <&3` gets nothing) —
  i.e. the node is CPU/memory-starved (likely swap death). rank0 then exited on
  distributed-init timeout. **spark2 needs a manual check/reboot.**

## Conclusion for this box
Do **not** run the torch profiler on prefills here. Use timing-only for prefill,
and if a trace is required, capture a tiny window on an otherwise-idle box.
