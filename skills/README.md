# Skills — DGX Spark / GB10 (sm_121) + DSpark

Agent skills distilled from optimizing a GLM-5.3-Flash-NVFP4 bf16 GEMV on a
2-node DGX Spark (GB10, sm_121) served with SGLang + DSpark speculative
decoding. Each folder holds a `SKILL.md` with frontmatter (`name`,
`description` with trigger keywords) — the format Claude Code / Codex / opencode
load.

| Skill | Use when |
| --- | --- |
| [`dgx-spark-gb10`](dgx-spark-gb10/SKILL.md) | Working on GB10/sm_121 kernels or SGLang serving: hardware facts (48 SMs, 25 MB L2, ~228 GB/s DRAM, ~121 GB unified memory), sm_121 kernel gotchas, which cuBLAS/SGLang JIT paths exist or are missing, shared-memory and live-service constraints. |
| [`spec-decode-ab`](spec-decode-ab/SKILL.md) | Measuring a change end-to-end under speculative decoding (DSpark / EAGLE / MTP / ngram): the acceptance-rate confound and the acceptance-independent protocol (step annotations, decode-only profiling, no-spec control runs). |
| [`sglang-twonode-live-ab`](sglang-twonode-live-ab/SKILL.md) | A/B a patched SGLang change on the live 2-node DGX Spark service: stop/restart both nodes, bind-mount overlay of patched source, dev container, health polling, trace capture, rollback. |
| [`spark-github-pr`](spark-github-pr/SKILL.md) | Pushing a branch / opening a PR from this box: SSH vs the stored PAT, clean worktree from the fork's `main`, `main` not `master`, secret redaction. |

These complement the platform sections (`ascend/` for the NPU port, `external/`
for the KernelWiki / ncu-report-skill submodules) and the task trees under
`llm/`, `diffusion/`, `nvidia_agent_tasks/`, and `mega_kernel/`.

The concrete kernel task behind them lives in
[`llm/glm53_flash__sm121_bf16_gemv_multirow/`](../llm/glm53_flash__sm121_bf16_gemv_multirow).
