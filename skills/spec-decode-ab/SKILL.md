---
name: spec-decode-ab
description: Use when measuring the end-to-end effect of a kernel or numerical change on a speculative-decoding server (DSpark, EAGLE, MTP, ngram). Explains the acceptance-rate confound (a rounding difference flips sampled tokens -> the two arms generate different text -> tokens/s is not a valid A/B) and gives the acceptance-independent protocol: step[...] trace annotations, decode-only profiling, and no-spec control runs. Triggers include "spec decode", "DSpark", "accept rate", "投机", "e2e tok/s 不可信", "verify step".
---

# A/B a change under speculative decoding without lying to yourself

## The trap

Speculative decoding's throughput = `accepted_tokens / target_forward_time`. Any
change that perturbs numerics by even ~1e-2 (a different reduction order, a
different bf16 tile) can flip an argmax, so the two arms **generate different
text**. Different text has a different accept length, so `tokens/s` differs for
reasons unrelated to kernel speed. Observed in one session: a GEMV was **1.14×
faster standalone and cut kernel time 17.6%**, yet single-prompt DSpark tok/s
went 25.9 → 23.0; across 3 prompts it was 23.9/21.65, 16.9/16.4, 29.6/42.1
(mean went *up*, median *down*). The spread between prompts (14–42 tok/s) dwarfed
the kernel effect. **A single-prompt tokens/s A/B is not a valid measurement.**

## Acceptance-independent metrics (use these)

1. **Designed control: disable speculation.** Run both arms with the spec args
   removed. One token per step, deterministic length, so `tokens/s` is purely
   `1/step_time`. This is the cleanest number.
2. **Profiler step annotations.** Torch-profiler traces of a spec server carry
   per-stage spans. Compare their per-call averages across arms:
   - `step[VERIFY bs=N]` — target forward (the thing your kernel speeds up)
   - `step[DRAFT bs=N]` — draft forward
   - `scheduler.run_batch` — whole step
   These are wall-clock and independent of acceptance. If the two traces cover
   the same number of steps (check that launch counts match), the comparison is
   fair.

Report **both**: the no-spec `tokens/s` delta and the `step[...]` latency delta.
State explicitly that spec `tokens/s` is content-confounded.

## Decode-only trace capture

SGLang exposes `POST /start_profile` / `POST /stop_profile`:

```bash
curl -s -X POST $BASE/start_profile -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"num_steps":8,"activities":["CPU","GPU"],"with_stack":false,"record_shapes":true}'
# send a request; then /stop_profile
```

- Without `output_dir` the trace lands in the **container** (`/tmp/<ts>-TP-0.trace.json.gz`);
  `docker cp` it out. With `output_dir` it goes to that path on the rank-0 node.
- `num_steps` starts counting from the first forward step, so **prefill counts as
  a step**. To profile decode only, first send the same prompt once (populate the
  prefix cache), then `start_profile`, then repeat the prompt (cache hit ⇒ decode
  only). Otherwise the baseline trace contains a prefill (`nccl`/allreduce time)
  and is not comparable.
- `stop_profile` may return HTTP 500 after auto-stop; the trace is still written.

Then aggregate by kernel family and by annotation:
```python
# per-kernel GPU time
agg = collections.defaultdict(lambda: [0,0.0])
for e in events:
    if e.get("cat")=="kernel" and e.get("ph")=="X":
        agg[e["name"]][0]+=1; agg[e["name"]][1]+=e["dur"]   # dur is µs
# per-stage wall time
for e in events:
    if e.get("ph")=="X" and e.get("cat") in ("user_annotation","gpu_user_annotation"):
        ...
```

## Rules of thumb

- Match arms by **kernel launch counts / step counts**, not by elapsed prompt
  time or by generated text.
- Values from different server starts are not comparable in absolute terms
  (GB10 thermal/clock drift is real). The robust signal is a same-launch-count
  family delta, or the per-step annotation latencies.
- If you must use tokens/s, average over **many distinct prompts** and report the
  distribution, and still prefer the step-latency evidence.
