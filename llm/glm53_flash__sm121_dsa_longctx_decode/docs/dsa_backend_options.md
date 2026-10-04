# DSA attention/indexer backend options on GB10 (sm_121) — assessment

Question: can DeepSelect / FlashMLA / FlashAttention replace or accelerate the
DSA attention+indexer (currently `--dsa-prefill-backend tilelang
--dsa-decode-backend tilelang`)?

## What each library maps to
| piece | kernel today | candidate library |
|---|---|---|
| indexer **logits** (Q·K over context, O(N²) prefill) | `deep_gemm::sm120_fp8_mqa_logits` / `paged_mqa_logits` (`dsa_paged_mqa_logits_backend=auto`) | none of the three |
| indexer **top-k** | `sgl_kernel.fast_topk_v2` (`dsa_topk_backend=sgl-kernel`) | **DeepSelect** (a TopK kernel, 2–20× vs `torch.topk`) |
| **sparse attention** (top-2048) | `tilelang` | **FlashMLA / FlashAttention** |

## Findings (measured/inspected in the image)
- **FlashMLA**: `flash_mla` is **not installed**; `flashmla_sparse_q8` is
  **SM90-only**; `flash_mla_sm120.py` says the flash_mla CUDA kernel is not
  available on SM120. → not usable on GB10.
- **FlashAttention**: `flash-attn-4` is installed, but the DSA `fa3` backend is
  SM90-only and FA is dense attention, not the selected/sparse MLA. → not
  applicable to DSA.
- **`flashinfer_sparse_mla`**: allowed only for arch `GlmMoeDsaForCausalLM` on
  sm12+fp8; our arch is `Glm5NextForConditionalGeneration` → rejected by
  `_validate_flashinfer_sparse_mla_backend`.
- **`triton` DSA backend**: rejected — *"only supported on ROCm/HIP"* (the CLI
  error even says: for CUDA use `flashmla_kv` on Hopper, `trtllm` on Blackwell).
- **`trtllm` DSA backend** (tried): **crashes on GB10** during CUDA-graph capture
  with `tvm.error.InternalError: ... 'TllmGenFmhaRunner' ... Unsupported
  architecture` — the trtllm FMHA runner needs sm100/sm103, not sm121.
- **DeepSelect**: not installed; and SGLang already uses a specialized TopK
  (`fast_topk_v2`) for the indexer, so it is a like-for-like swap at best.
  The dominant indexer cost is the O(N²) **logits**, not the top-k.

## Conclusion
On **GB10 (sm_121) + GLM-5.3-Flash**, **no alternative DSA attention backend is
usable**: `flashmla_*` (Hopper / absent), `fa3` (Hopper), `flashinfer_sparse_mla`
(different arch), `triton` (ROCm), `trtllm` (sm100). **`tilelang` is the only
working backend**, which is what production already uses.

So the prefill "C ≈ 45 s" cannot be attacked by swapping in FlashMLA/FA/trtllm on
this hardware. The only remaining levers for the indexer/attention are
kernel-level changes **inside** the tilelang prefill attention and the deepgemm
indexer-logits kernel (a native-kernel task), not a backend swap. DeepSelect could
only help the top-k slice (small), and would need porting + integration into
`dsa_topk_backend`.
