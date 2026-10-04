
## 4. K1/K3 移植 + 全管线 A/B（2026-10-03 续）

`solution/tilelang_moe/k1k3_emaj.py`: K1（w13+SwiGLU 双 MMA）与 K3（w2）的 emaj 栅格版，
SF 打包不变（BM=128）。oracle: 小形状 rel≈0.3%，全管线 dequant oracle max_rel=0。

| M | baseline flashinfer | PR#7 bnmaj | **PR#7+emaj** | emaj vs baseline |
|---|---:|---:|---:|---:|
| 4096 | 17.73 | 15.56 | **14.88** (K1 7.40 + K3 4.42) | **1.19×** |
| 8192 | 28.34 | 23.90 | **21.70** (K1 8.94 + K3 5.90) | **1.31×** |

K1 emaj 单核: M4096 7.77→7.40（flashinfer w13+act 9.45 → 1.28×）;
M8192 10.75→8.94（1.20× vs bnmaj）。K3 emaj: M8192 6.32→5.90; M4096 持平。
M=8192（当前生产 chunk）收益最大。运行入口: `fused_moe.py --M 8192 --emaj`（补丁版已存
`solution/tilelang_moe/fused_moe_emaj_runner.py`）。

## 5. 正确性与精度验证（2026-10-03，`bench/verify_emaj.py`）

| # | 验证项 | 结果 |
|---|---|---|
| A | K1/K3-emaj vs dequant oracle，**全部 144 专家**、生产形状 | PASS：max_rel 0.25%/0.375%，均值 0.0075%/0.14%，全 finite |
| B1 | 全管线 emaj vs bnmaj（同输入，M=4096） | PASS：**bit-exact**（纯调度改动，无数值影响） |
| B2 | 全管线 vs 自身 dequant oracle（M=4096/8192，全专家） | PASS：max_rel=0（精确） |
| C1 | 全管线 vs fp32 未量化权重 oracle | PASS：err/absmax p50=2.3%、p99=9.0%、max 26%；cosine 0.9714 |
| C2 | vs 理想 NVFP4 模拟管线（纯 torch + flashinfer 量化） | PASS：cosine **0.9950**；与理想实现的差异=量化实现差异（SIG p50 5.7%） |
| C3 | vs 理想 NVFP4 的噪声基准 | 理想 NVFP4 自身 vs fp32 oracle: cos 0.9731, p50 2.2% → **管线噪声=NVFP4 固有** |
| D1 | 空专家（24 个 size=0） | PASS：bit-exact，finite |
| D2 | 极端激活 ×60（swiglu limit=10 截断路径） | PASS：bit-exact，finite |
| D3 | 单热专家（全 token 命中 e=77，cap 压力） | PASS：finite |

**C. flashinfer 对照说明**：`probe_baseline` 的 quant_scales 配方（1/gs）是纯计时用、从未数值
验证；实测其输出 scale 错 ~2000×（median ratio 0.0005；1/gs、gs、1.0 三种 alpha 均不匹配）。
这是 decode 任务遗留的 fused-path alpha 约定问题（见 nvfp4_moe_decode/docs/results.md §3），
不是本管线的问题。真正对线上的 parity 待 `--moe-runner-backend` 集成时在服务端做。

**结论**：emaj 管线 = 调度改动（对 bnmaj bit-exact）+ 数学精确（dequant oracle max_rel=0）
+ 端到端噪声与理想 NVFP4 实现不可区分（cos 0.995）。

## 6. 横向实现调查：grouped GEMM / bf16 投影还有没有空间（2026-10-03）

### bf16 投影（prefill 24%）— NO-GO 加固
| 实现 | 4096³ / 8192³ / 4096×12288×4096 |
|---|---|
| torch.mm (cuBLAS) | 97.5 / 102.3 / 98.5 TFLOPS |
| flashinfer mm_bf16[cudnn] | 90.0 / 97.2 / —（更慢） |
| flashinfer mm_bf16[cutlass/tgv] | 不支持 capability 121 |
| flashinfer mm_bf16[cublaslt] | 布局要求未满足（同 cuBLAS 家族） |
| TileLang bf16（本次新测） | ~40 TFLOPS 顶（99KB smem 限死 stage 数） |
| DeepGEMM | 无 SM120/SM121 路径（csrc grep 空） |
三个独立实现族都在 100–102 饱和 → cuBLAS≈硅上限成立，"NCU 85%→120" 假设无任何实现支撑。关闭。

### Grouped GEMM（NVFP4 MoE）— 已 90% 关账，最后杠杆=cluster multicast
- flashinfer fused MoE（生产）：CUTLASS persistent cooperative，无 B-reuse 序 → 被 emaj 胜 1.37×/1.24×
- flashinfer `group_gemm_nvfp4_nt_groupwise`（SM120 专用）：暴露但**无人接线**（sglang/flashinfer 内零调用者）；
  ClusterShape 硬编码 1×1×1；tile_n∈{32,64,128}=tile 缩小方向（已证伪）；swap_ab 未测但同 collective 家族
- DeepGEMM：无 SM120；vLLM：本地只有 ascend 分支；TileLang 官方 fusedmoe：fp16 非 blockscaled
- **剩余唯一实质杠杆**：CUTLASS ClusterShape<2,1,1> + TMA **multicast B**（相邻 m-tile 的 2 个 CTA 硬件多播同一
  B tile，替代 L2 兜底）→ 把 GEMM 从 90% 屋顶推到 ~100%。任何本地实现都没有；需 native CUDA 任务
  （TileLang sm120 路径无 cluster 支持）。预期收益：GEMM 再 ~1.1× ≈ TTFT 2-3%，边际收益递减。
- 管线剩余非 GEMM 空间：K2 requant 2.05ms vs 地板 ~1.4ms（1.5×）≈ 0.6ms；其余贴地板。
