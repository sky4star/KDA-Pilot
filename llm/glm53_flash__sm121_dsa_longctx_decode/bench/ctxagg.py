#!/usr/bin/env python3
"""Aggregate a torch-profiler trace by kernel family, split by step type
(EXTEND vs VERIFY/DRAFT decode) using the gpu_user_annotation windows."""
import gzip, json, re, sys
from collections import defaultdict

FAM = [
    ("moe_grouped_gemm", r"MainloopSm120|BlockScaled|GroupProblemShape"),
    ("moe_aux", r"tensorrt_llm::kernels"),
    ("proj_bf16_gemv", r"sm120_bf16_gemv"),
    ("proj_cublas", r"cutlass_80_wmma_tensorop_bf16|nvjet|tensorop_bf16_s16816"),
    ("dsa_indexer_logits_topk", r"paged_mqa_logits|mqa_logits|fast_topk|topk_v2|kpool|_router_triton|indexer"),
    ("dsa_sparse_attn", r"sparse|main_kernel|fmha|BatchPrefill|BatchDecode|flash_attn|flashattn"),
    ("kda_conv_deltarule", r"fused_sigmoid_gating_delta_rule|_causal_conv1d|chunk_kda|chunk_gla|mamba_state"),
    ("gated_norm", r"layer_norm_gated|rms_norm|norm_kernel|fused_add_rmsnorm|flashinfernorm"),
    ("mhc_hyperconn", r"mhc_|deep_gemm::sm120_tf32_hc_prenorm"),
    ("nccl", r"ncclDevKernel"),
    ("other", r".*"),
]

def fam(name):
    for f, pat in FAM:
        if re.search(pat, name, re.I):
            return f
    return "other"

def load(p):
    return json.load(gzip.open(p) if p.endswith(".gz") else open(p))

def main(path):
    d = load(path)
    ev = d["traceEvents"]
    ann = [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
           if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"]
    def stepkind(ts):
        best = None
        for s, e, n in ann:
            if s <= ts <= e and (best is None or (e - s) < (best[1] - best[0])):
                best = (s, e, n)
        if not best:
            return "none"
        n = best[2]
        if n.startswith("step[VERIFY") or n.startswith("step[DRAFT"):
            return "decode"
        if n.startswith("step[EXTEND"):
            return "extend"
        return "other"
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        k = stepkind(e["ts"]); f = fam(e["name"])
        agg[k][f][0] += e.get("dur", 0)
        agg[k][f][1] += 1
    print(f"# {path}")
    for k in ("decode", "extend", "other", "none"):
        if k not in agg: continue
        tot = sum(v[0] for v in agg[k].values())
        print(f"## {k}: total {tot/1000:.2f} ms")
        for f, (us, n) in sorted(agg[k].items(), key=lambda kv: -kv[1][0]):
            print(f"   {us/1000:9.3f} ms {100*us/tot:6.2f}% n={n:5d}  {f}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
