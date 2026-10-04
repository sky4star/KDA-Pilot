# GLM-5.3-Flash 本地服务算子优化 — 核心分析交接

> 单文件交接。新会话先读本文件 + 对应 task 的 `docs/`，即可继续。
> 生成时间：2026-10-02。工作根：`~/kda/KDA-Pilot`。

## 0. 环境 / 服务（重要）

- 模型：`nvidia/GLM-5.3-Flash-NVFP4`（`/home/spark/models/GLM-5.3-Flash-NVFP4`），
  DSpark 草稿模型 `/models/GLM-5.3-Flash-DSpark-sglang`。
- 拓扑：`tp=2` 跨 **2× NVIDIA GB10（sm_121）**，`pp=1`, `nnodes=2`，服务端口 **8940**。
  - spark1 = rank0 = `10.100.8.1`；spark2 = rank1 = `10.100.8.2`。
  - 容器名 `dspark-tp2-rank0` / `dspark-tp2-rank1`。
  - 镜像 `sglang-ppmtp:glm53-ppmtp-dspark-v1`（SGLang `0.0.0.dev1+gf609d677b`，flashinfer 0.6.17，tilelang 0.1.12，torch 2.13+cu130）。
  - API key：`REDACTED_SET_SGLANG_API_KEY`（`~/glm53-gemv-overlay/run-ab.sh` 里）。
- 启动脚本：`~/glm53-gemv-overlay/run-ab.sh <0|1>`，环境变量：
  `OVERLAY=1 SPEC=1 CHUNKED=4096 DSAPRE=tilelang DSADEC=tilelang`。
  **当前两个容器都已停止（GPU 空闲）**；恢复命令：
  ```bash
  docker rm -f dspark-tp2-rank0; ssh spark2 'docker rm -f dspark-tp2-rank1'; sleep 3
  ssh spark2 'CHUNKED=4096 OVERLAY=1 SPEC=1 nohup bash ~/glm53-gemv-overlay/run-ab.sh 1 > ~/ab_r1.log 2>&1 &'
  CHUNKED=4096 OVERLAY=1 SPEC=1 nohup bash ~/glm53-gemv-overlay/run-ab.sh 0 > ~/ab_r0.log 2>&1 &
  # 就绪约 10–15 分钟；curl -H "Authorization: Bearer <KEY>" :8940/health == 200
  ```
- **硬件事实**：GB10 sm_121，48 SM，L2 25MB，121GB 统一内存，实测 copy/read 上限
  **~230–240 GB/s**（64–256MB）。模型加载后**只剩 ~5GB 余量**（used ~118/121）。
- **安全红线**：**不要在这台机器上用 torch profiler 采长 prefill**（已两次把服务搞崩）。
  见 `llm/glm53_flash__sm121_dsa_longctx_decode/docs/incident_long_prefill_profile.md`。
  采 trace 一律 `num_steps<=2~4`、`detailed_annotations=False`；能只测时间就只测时间。

## 1. 模型结构（决定一切）

45 层：3 dense + 42 MoE；注意力混合：**34 层 KDA（线性注意力）+ 11 层 DSA（DeepSeek 稀疏 MLA）**。
- hidden 4096；MoE `n_routed_experts=288`（TP2 → 144/rank）、`topk=8`、`shared_experts=1`、
  `moe_intermediate=2048`、`swiglu_limit=10`、`routed_scaling_factor=2.5`。
- DSA：`index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`、`index_kpool=4`（压缩）、
  `index_share_for_mtp_iteration=True`。
- NVFP4 专家（E2M1 + E4M3 block-16 + global scale）；bf16 ignore-list：
  注意力/KDA/dense/共享专家/lm_head。

## 2. Decode（单请求）结论

**算子占比（decode trace，整段）**：
MoE 分组 GEMM **46.5%**、bf16 GEMV **27%**、bf16 cuBLAS 回退 10%、NCCL 4.3%、
KDA/mamba 2.4%、MHC 1.7%、router 1%、DSA indexer 0.3%。

- **bf16 GEMV 已优化**（前序任务 `llm/glm53_flash__sm121_bf16_gemv_multirow`，M∈[1,8]）：
  no-spec decode +15%，DSpark verify-step 1.15×、draft-step 1.35×。**已达/超带宽屋顶**。
- **MoE grouped GEMM 是 decode 唯一有空间的大头**：
  - 基线（cold-L2+CUDA-graph）M=1: **533.9us / ~188 GB/s**（ceil ~240）→ **~1.25–1.35×**；M≥2 已达顶。
  - NCU：grid `(1,48,1)`=48 CTA/1 波、22% 占用、168 寄存器、spill、warp-spec NANOSLEEP → **延迟/占用受限**。
  - 任务 `llm/glm53_flash__sm121_nvfp4_moe_decode`：基线+NCU+roofline 齐全；
    native CUTLASS SM120 block-scaled GEMM 可编译运行，但**未达 fused-MoE 数值一致**
    （卡点=fused 的 activation SFA 布局 / SwiGLU epilogue / alpha 约定）。
- 小 Triton 核（如 KDA gated norm）在 trace 里的耗时是 **profiler 假象**（CUDA graph 下仅 1.6–4.6us）。

## 3. 长上下文 decode（4K/32K/100K）—— 无新瓶颈

- 每步延迟**基本不变（100K 甚至略低）**：VERIFY 1.71/1.63/**1.21**ms，DRAFT 3.23/2.09/**1.96**ms。
- decode kernel 总量 32K vs 100K：**778.6 vs 774.8ms**（几乎相同）。
- 唯一随上下文增长的是 **DSA indexer**（`sm120_fp8_mqa_logits` 0.84→1.60ms + index-k gather 0.23→0.69ms），
  合计仍 **<1%**。
- 原因：DSA 只对 **top-2048** 做 attention（O(2048)），KDA 递推 O(1)。
- 任务：`llm/glm53_flash__sm121_dsa_longctx_decode`（`docs/results.md`）。

## 4. Prefill / TTFT —— **真正的长上下文瓶颈，且有 ~2× 现成收益**

**冷 prefill TTFT（单请求，unique prompt）**：

| context | chunk=512 | chunk=4096 | 加速 |
|---:|---:|---:|---:|
| 8K | 7.73 s | 4.25 s | 1.82× |
| 32K | 34.61 s | 16.66 s | 2.08× |
| 100K | 102.34 s | 51.25 s | 2.00× |

- 原因：chunked prefill **每个 chunk 都要把全部权重流一遍**，chunk 512→100K 要 ~196 遍，
  4096 只要 ~25 遍。**已实锤建议：`--chunked-prefill-size 4096`**（再大收益递减、内存风险大）。
- 记 `docs/results_prefill.md`。

### 4.1 Prefill 算子占比（实测，8K/16K 稳定）

| 算子族 | 8K | 16K |
|---|---:|---:|
| **bf16 投影 GEMM（cuBLAS wmma/nvjet）** | **24.3%** | **23.9%** |
| **MoE routed 分组 GEMM（NVFP4）** | **21.2%** | **20.8%** |
| MHC / hyper-connection | 12.9% | 12.6% |
| other（elementwise/quant/rope/copy） | 12.2% | 12.0% |
| NCCL | 7.6% | 8.2% |
| MoE aux（routing/sort/act/finalize） | 7.2% | 7.0% |
| **DSA sparse attention（tilelang）** | **6.9%** | **6.8%** |
| KDA conv + gating/delta-rule | 6.4% | 6.2% |
| gated norm / indexer / gemv | 其余 | 其余 |

- **prefill ≈ GEMM 主导**（投影 24% + MoE 21% + aux 7% ≈ 52%）。
- **sparse attention 只有 ~7%**，indexer <1%（每 chunk 时间恒定 → 非 O(N²) 主导）。
- 投影在 M=4096 时 ~1.5ms/[4096×4096,4096] bf16 ≈ **~90 TFLOPS → 计算受限**（与 decode 带宽受限不同）；
  MoE fp4 M=4096 每 GEMM ~5ms，MFU 偏低。
- 记 `docs/results_prefill_composition.md`。

## 5. 后端 / 库可行性（都已验证）

- **DSA attention 后端在 GB10 上无替代**：需要 `flash_mla`（未装/SM90）、`fa3`（SM90）、
  `flashinfer_sparse_mla`（只允许 arch=`GlmMoeDsaForCausalLM`，我们是 `Glm5NextForConditionalGeneration`）；
  `triton` **ROCm-only**；`trtllm` **实测崩溃**（`TllmGenFmhaRunner ... Unsupported architecture`，需 sm100/103）。
  → **只有 `tilelang` 能跑**（就是线上配置）。记 `docs/dsa_backend_options.md`。
- **DeepGEMM**：MoE 是 FP8/MXFP8(UE8M0)，与 NVFP4(E4M3 block-16) 不兼容；新版才有 `cublaslt_nvfp4`。
- **DeepSelect**：只是 indexer 的 **TopK** 算子；SGLang 已用 `fast_topk_v2`；且 topk 不是大头（logits 才是）。
- **FlashAttention**：dense attention，非 DSA 选择性 MLA，不适用。
- **EP 替代 TP**：2 节点下专家分片度相同（144/rank），单请求无增益；只有加节点（ep_size>tp_size）才有意义。
- **异步重叠**：单请求只有通信可藏（~4–8%）；KDA-Pilot `mega_kernel/LEARNINGS.md`（GLM-5.2 bs=1）
  已判死 finalize+AR 融合、重写 AR 等；megakernel 只有 0.2ms/iter 级小增量。

## 6. 已建任务 / 产物

| 任务 | 结论 | 关键文件 |
|---|---|---|
| `llm/glm53_flash__sm121_bf16_gemv_multirow` | ✅ 已上线（前序） | solution/ |
| `llm/glm53_flash__sm121_nvfp4_moe_decode` | 有 ~1.25× 空间；native 候选卡在 parity | `docs/results.md`、`docs/draft.md`、`solution/cutlass_fp4*` |
| `llm/glm53_flash__sm121_bf16_gemm_multi_m` | ❌ no-go（M=42 extend，cuBLAS 已达顶） | `docs/results.md` |
| `llm/glm53_flash__sm121_dsa_longctx_decode` | 长上下 decode 无瓶颈；prefill 有 ~2× chunk 收益；含组成分析 | `docs/{results,results_prefill,results_prefill_composition,dsa_backend_options,incident_long_prefill_profile}.md` |

剖面/脚本：`~/profiles/longctx/`（L*_overlay, P*）、`bench/{capture,prefill_probe,prefill_capture_safe,ctxagg,prefill_chunk_timing}.py`。

## 7. 下一步（新会话从这里继续）

**目标：优化 prefill 的两大 GEMM（占比 ~45%，且被证明是计算受限/MFU 低）**

按 KDA 方法（standalone、cold-L2、CUDA-graph、对称 ABI、native CUDA 候选）：
1. 起 dev 容器（服务已停、GPU 空闲）：
   ```bash
   docker run -d --name gemm-dev --gpus all --privileged --network host --ipc host \
     --ulimit memlock=-1:-1 -v $HOME/kda/KDA-Pilot:/kda \
     sglang-ppmtp:glm53-ppmtp-dspark-v1 sleep infinity
   ```
2. **先做 roofline 再动手**（避免重复 task A 的 no-go）：
   - 测 GB10 bf16 峰值 TFLOPS（大方阵 cuBLAS）；
   - 测投影 GEMM（M=4096，各 K/N）cuBLAS 的 TFLOPS 与 MFU；
   - 测 MoE grouped NVFP4（M=4096, E=144, topk=8, K=4096, N=2048）时间与有效 MFU
     （复用 `glm53_flash__sm121_nvfp4_moe_decode/bench/probe_baseline.py`，把 M 改成 4096）。
3. 谁离屋顶远就优化谁：
   - **MoE fp4 M=4096**（MFU 低）→ 最可能；注意 flashinfer CUTLASS config/tile。
   - **bf16 投影 M=4096**（~90 TFLOPS）→ 若峰值远高于此则有空间；候选可用 CUTLASS
     SM80 bf16（**注意：CUTLASS 4.x 的 SM120 非 blockscaled builder 只支持 F8F6F4，bf16 需手写或走 SM80 collective**，见 task A 的教训）。
4. 候选通过后再考虑并回线上（`~5GB` 余量，谨慎）。

**其他待办**：MoE decode M=1 的 native parity（fused SFA/SFB + SwiGLU epilogue + alpha）。

## 7b. Prefill GEMM roofline 结论（2026-10-02 新会话，任务 `glm53_flash__sm121_prefill_gemm`）

- **bf16 投影 GEMM（24%）= NO-GO**：GB10 cuBLAS bf16 顶峰实测 **~100–102 TFLOPS**
  （4096³=93.5 / 16384³=102.2）；生产 M=4096 各 (K,N) 跑在 **91–99 TFLOPS（85–99% MFU）**。
  与 task A 的 no-go 一致。
- **MoE routed NVFP4 grouped（21–36%）= 有 ~2.5–3× 空间**：M=4096 时
  **93 TFLOPS / 102 GB/s**，既非计算也非带宽屋顶；NCU：**grid=48、1 波、occupancy 20%**、
  168 regs、spill、warp-spec NANOSLEEP → 延迟受限。**等价 dense FP4 GEMM 只要 5.4ms**，
  grouped 要 17.4ms。
- 库替代都试过：`grouped_mm_fp4`（cuDNN 需 ≥9.21，本机 9.20）不可用；
  cute-dsl 不支持 cap121；**`b12x_fused_moe`（SM12x CuTe-DSL）1.04× 持平**，occupancy 仅 10%；
  `tune_max_num_tokens` sweep 无变化。
- M 扫描：93→117→127 TFLOPS @ M=4096→8192→16384 → 单个 8192 chunk 比两个 4096 chunk
  的 MoE 吞吐高 **1.26×**（config 级小抓手）。
- 产物：`llm/glm53_flash__sm121_prefill_gemm/{bench,docs}`（`docs/roofline.md`、
  `profile_evidence.md`、`backend_survey.md`、`results.md`）。
- **下一步（RLCR）**：写一个高 occupancy / 非 persistent 的 grouped block-scaled SM120 GEMM
  （CUTLASS array-TMA），或改 CUTLASS 调度；先过 parity（dequant oracle + baseline），
  再 cold-L2/CUDA-graph 对称 benchmark，最后走 `--moe-runner-backend` e2e。

## 7c. TileLang 可行性（2026-10-02）

- **装机的 tilelang 0.1.12：写不了**。fp4 `T.gemm` 直接 `Unsupported dtype:
  float4_e2m1fnx2`；block-scaled 只有 tcgen05（`TargetIsSm100` + E8M0 尺度），
  GB10 无 tcgen05/wgmma，回退的 `mma.h` 无任何 fp4。
- **tilelang 0.1.15：可以（稠密已实测）**。新增
  `mma.sync...m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3`
  （正是 NVFP4 E4M3 block-16），API 是 `T.mma_gemm_blockscaled(sf_layout=
  "blockscaled_chunk_kmajor", sf_*_granularity_k=16)`；官方示例
  `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` 是**非 persistent**
  tiled GEMM（grid=N/128×M/128，高 occupancy）。
- **需要一个 1 行补丁**：该 header 的 guard 只认 `CUTLASS_ARCH_MMA_SM120A`，
  而 GB10 是 SM121A → device assert。改成同时接受 `SM121A` 即可（CUTLASS 自己
  对 SM121A 是使能该宏的）。补丁在
  `glm53_flash__sm121_prefill_gemm/solution/tilelang/mma_block_scale_sm121a.patch`。
- **稠密实测（GB10，128x128x256，2 stage）**：4096³=261、16384×4096×4096=305、
  **32768×4096×4096=313**、**32768×4096×2048=259 TFLOPS**；w13+w2 合计 5.64ms
  vs grouped MoE 17.7ms。→ 张量核路径能到屋顶，缺的只是 grouped 调度。
- **MoE 还缺**：① grouped dispatch（无现成 NVFP4 fused-MoE 示例，上游 fusedmoe
  是 fp16，需自己写 grid over experts + 动态 per-expert token range）；
  ② SF 转到 `blockscaled_chunk_kmajor` uint32；③ activation/SwiGLU/scatter；
  ④ 部署：镜像需从 0.1.12 升到 0.1.15（SGLang/DSA 都 pin 0.1.12，要重验）。
- **grouped NVFP4 GEMM 已写出来并跑通**（`solution/tilelang_moe/grouped_nvfp4_gemm.py`）：
  非 persistent grid、warp-specialized、TMA+cp.async、`blockscaled_chunk_kmajor`。
  关键坑：grid 必须 **N 方向变化最快**（否则 A 被重读 32 次，L2 53%）→ w13 18.3ms→8.0ms。
  M=4096 w13=8.03ms(137 TFLOPS)、w2=4.80ms(115)，合计 12.83ms vs baseline grouped 13.88ms
  （**1.08×**），正确性 rel<0.4%（独立 dequant oracle）。M=16384 时 w13 到 206 TFLOPS。
  仍受限：occupancy 15%（regs197+smem73KB 都卡 1 block/SM），且 TileLang SM120 路
  block_M=block_N=128 固定，无法让一个 CTA 吃下一个 expert 的 227 行 → 到不了 dense 屋顶。
  下一步：SwiGLU+fp4 requant 的 epilogue/w2/scatter 融合 + parity。
  详见 `docs/tilelang_moe_milestone.md`。
- **完整 MoE 已跑通**（`solution/tilelang_moe/full_moe.py`）：gather→w13→SwiGLU(limit)→
  fp4 requant→w2→scatter+权重，端到端 vs fp32 dequant 参考 max_rel 0.26%。
  M=4096 两个 grouped GEMM 合计 **13.4ms**（123 TFLOPS），≈ baseline GEMM 部分（13.88ms）。
  但原型 torch glue 不可用（87ms，因为 144 专家 ×2 次单独 `fp4_quantize`）→
  必须把 activation 量化/requant 融成单个 kernel。真正超过 baseline 17.73ms 的路径是：
  w13 epilogue 融合 SwiGLU+requant、gather/quant 融合、scatter 融合，再对 baseline 做
  activation-quant parity；预期上限 ~14ms（~1.25×）。
  实测 glue 分解（M=4096，改成整张量一次 `fp4_quantize` 后仍 ~69ms 总、GEMM 13.3ms）：
  gather 2ms、`x.float().abs().max()` 10.7ms（改 bf16 amax 后 ~1）、fp4_quantize 1.6×2、
  pack+swizzle 4.6×2、`o2.float()*scale`+`index_add` 20ms、`o1.float()*scale` 3.6ms。
  全是 baseline 里用 3.8ms 的 C++ fused kernel（expand/activation/finalize）干的活。
  → 必须写 3 个 TileLang kernel：①w13 epilogue 内 gate+up 双 MMA + SwiGLU + fp4 量化
  （难点：SM120 blockscaled 固定 block_M=block_N=128，双累加器 ~128 extra regs）；
  ②gather+quant（X+topk_ids→A1 fp4+块尺度）；③fused scatter（`y[tok]+=w*out2`）。
  **已做**：③ fused scatter（`solution/tilelang_moe/scatter_kernel.py`，TileLang atomicAdd）
  = **5.62ms vs torch index_add 14.94ms（2.66×, bit-exact）**；另加 fused SwiGLU
  （`swiglu_kernel.py`，**1.85ms vs 20.5ms = 11×**）、`pack_scale_words` 改 int32、
  `repeat_interleave`。全 forward 87→**35.1ms**（GEMM 13.35ms 不变），校验仍 0.27%。
- **重要负面结论（M=4096）**：即使 glue 全融合，地板 ≈ GEMM 13.35 + gather 1.4 +
  swiglu/quant 1.4 + scatter 3.4 ≈ **19.5ms > baseline 17.73ms** → **在 chunk=4096
  打不过 baseline**。`grouped_nvfp4_gemm_reuse`（一 CTA 吃一个 expert 的全部 m-block
  以复用权重）**更慢**（10.6 vs 8.5ms，串行 lm 掉并行度）。
- **完整 MoE 随 M 对比**（full forward）：M=4096 35.1ms vs baseline 17.7；
  M=8192 61.6 vs 28.2；M=16384 116.9 vs 51.8。**TileLang 的 GEMM 本身在
  M=8192/16384 反而比 baseline 整个 fused MoE 还快**（19.4<28.2、33.4<51.8），
  但 **torch glue 随 M 线性增长**（42/83ms）把优势吃光；全融合后的地板也只到
  break-even。→ **结论：M=4096/8192 下不推荐用 TileLang MoE 替换 baseline**；
  唯一可行路径是单 kernel 全融合 + parity，且上限有限。prefill 的收益更可能来自
  **chunked-prefill 配置**（更大 chunk 同时提升两边 MoE 利用率）和非 GEMM 算子。
  详见 `docs/tilelang_moe_milestone.md`。
- 详见 `llm/glm53_flash__sm121_prefill_gemm/docs/tilelang_feasibility.md`、
  `solution/tl_sm120_nvfp4.py`、`bench/tl_probe.py`。

## 7d. TileLang 全融合 MoE（目标：对齐 baseline，2026-10-02）

目标改为"用 TileLang 写出与 `cutlass_fused_moe` 一致的性能"，评估 TileLang 是否够生产。
草案 `docs/fused_moe_tilelang_draft.md`，实现 `solution/tilelang_moe/fused_moe.py`（4 kernel）：

| kernel | ms | 说明 |
|---|---:|---|
| K0 gather+quant | 0.76 | gather + NVFP4 block-16 E4M3，tile [128,128]（`act_quant_kernel.py`） |
| K1 w13+swiglu | 7.49 | gate/up 双 block-scaled MMA + 融合 SiLU（`w13_swiglu_kernel.py`） |
| K2 requant | 0.90 | 同 K0，identity gather |
| K3 w2 GEMM | 4.70 | grouped w2，连续写 out2（无 atomic）（`w2_gemm_kernel.py`） |
| C combine | 1.46 | gather-reduction 合并 topk（`moe_combine_kernel.py`） |
| **合计** | **15.31** | baseline 17.73 → **1.16× 更快**，dequant oracle **max_rel=0** |

- **关键**：一开始把 scatter 融合进 w2 epilogue 用 atomicAdd（8.50ms），比 baseline 的
  finalize（1.37ms）慢 ~2ms。改成 **baseline 方式（方案 A）**——K3 只做 w2 GEMM 连续写
  `out2`，另写一个 **gather-reduction combine**（每个 token 按 topk 行 gather 求和，
  无原子）——combine 贴到 **1.46ms 内存地板**，且去掉 C_sh staging 后 w2 GEMM 也更快
  （4.70 vs 5.15 standalone）。
- M-scaling（都 **~1.16–1.17× 快于 baseline**）：M=4096 15.31 vs 17.73；8192 24.04 vs
  28.20；16384 44.18 vs 51.76。不像 atomic 版那样随 M 恶化。
- 其他关键修复：**双累加器 kernel 必须关 warp specialization**（否则 spill 122M 次，
  K1 18.8→7.5ms）；**K0/K2 量化 tile [128,16]→[128,128]**（3D (blk_m,G,16) fragment +
  `T.reduce_absmax(dim=2)`，2.32→0.75ms）。
- 参考：TileLang `examples/fusedmoe`、`examples/deepseek_v4/act_quant.py`、我们的 SM120
  `T.mma_gemm_blockscaled`。DeepGEMM 是 CUDA/SM90-100，不能直接用。
- **判定：TileLang 不仅够生产，还比 CUDA baseline 快 ~1.16×**。上线前仍需：
  ①tilelang 0.1.15 + SM121A 补丁并重验 SGLang/DSA；②对 baseline activation-quant 做
  parity；③`--moe-runner-backend` 集成 + cold-L2/CUDA-graph A/B。

## 7e. 当前状态 / 新会话继续看这里（2026-10-03）

**完整交接见 `llm/glm53_flash__sm121_prefill_gemm/docs/session_summary.md`。**

- **成果**：TileLang 全融合 SM120 NVFP4 MoE，M=4096 **15.31ms vs baseline 17.73ms（1.16×）**，
  M=8192/16384 均 1.17×；数值精确（dequant oracle max_rel=0）。
  PR：**https://github.com/sky4star/sglang/pull/7**（分支 `tilelang-nvfp4-sm120-moe`）。
- **环境**：`gemm-dev` 容器，tilelang 0.1.15 影子装在 `/tmp/tl15site`（需
  `PYTHONPATH=/tmp/tl15site` + SM121A 补丁）；脚本入口
  `solution/tilelang_moe/fused_moe.py --M 4096 --E 144 --K 4096 --N 2048`。
- **三个负结果（别重做）**：packed 无空块=中性；auto-WS+setmaxnreg=编译挂起；
  手工 `T.ws`=正确但慢 1.5×。根因：sm_121 寄存器累加器 + **同步 mma.sync** + 100KB smem
  → 占用率(16.8%)/延迟天花板，TileLang 调度手段解决不了。
- **下一步（按潜力）**：①自写更小 tile 的 SM120 atom 降寄存器提占用（最大 lever）；
  ②`T.Persistent`+`num_stages=3`（便宜未试）；③bf16 真峰值实验（docs/open_questions.md）；
  ④chunked-prefill 配置 A/B；⑤parity + `--moe-runner-backend` 集成 + 0.1.15 升级重验。

## 8. 关键命令行速查

```bash
KEY=REDACTED_SET_SGLANG_API_KEY
# 健康
curl -H "Authorization: Bearer $KEY" http://10.100.8.1:8940/health
# 单请求冷 prefill 计时
python3 ~/kda/KDA-Pilot/llm/glm53_flash__sm121_dsa_longctx_decode/bench/prefill_probe.py 3
# 安全小窗口 profile（2 个 chunk）
python3 ~/kda/KDA-Pilot/llm/glm53_flash__sm121_dsa_longctx_decode/bench/prefill_capture_safe.py --len 8192 --num-steps 2
# 解析
python3 ~/kda/KDA-Pilot/llm/glm53_flash__sm121_dsa_longctx_decode/bench/ctxagg.py <trace.json.gz>
```
