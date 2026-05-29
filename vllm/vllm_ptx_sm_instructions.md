# vLLM PTX SM 系列指令集深度挖掘报告

> 源码路径: `C:\Users\Administrator\Desktop\nccl\vllm`
> 生成时间: 2026-05-29
> 涵盖版本: 最新版 (commit 84b2a8a7e)

---

## 目录

1. [SM架构标志与计算能力](#1-sm架构标志与计算能力)
2. [PTX汇编与内联PTX指令](#2-ptx汇编与内联ptx指令)
3. [SM架构守卫与条件编译](#3-sm架构守卫与条件编译)
4. [Triton内核SM配置](#4-triton内核sm配置)
5. [GPU架构代号与产品名](#5-gpu架构代号与产品名)
6. [Warp级与Tensor Core指令](#6-warp级与tensor-core指令)
7. [NCCL与通信原语](#7-nccl与通信原语)
8. [量化与精度SM特性](#8-量化与精度sm特性)

详细分卷:
- [part1_sm_arch_flags.md](part1_sm_arch_flags.md) - SM架构标志
- [part2_ptx_assembly.md](part2_ptx_assembly.md) - PTX汇编指令
- [part3_sm_guards.md](part3_sm_guards.md) - SM架构守卫
- [part4_triton_sm.md](part4_triton_sm.md) - Triton内核
- [part5_gpu_arch_names.md](part5_gpu_arch_names.md) - GPU架构代号
- [part6_warp_tensor_core.md](part6_warp_tensor_core.md) - Warp/Tensor Core
- [part7_nccl_comm.md](part7_nccl_comm.md) - NCCL通信
- [part8_quant_sm.md](part8_quant_sm.md) - 量化SM特性

---

## 1. SM架构标志与计算能力

### 1.1 全部SM版本总览

| SM版本 | 架构代号 | 代表产品 | 线程/SM | 首次出现位置 |
|--------|---------|---------|---------|------------|
| sm_70 | Volta | V100 | 2048 | `csrc/launch_bounds_utils.h:28` |
| sm_72 | Volta | GV100/GL | 2048 | `csrc/launch_bounds_utils.h:28` |
| sm_75 | Turing | T4, RTX 2080 | 1024 | `csrc/launch_bounds_utils.h:16` |
| sm_80 | Ampere GA100 | A100 | 2048 | `csrc/launch_bounds_utils.h:28` |
| sm_86 | Ampere GA10x | A10, A30, A40 | 1536 | `csrc/launch_bounds_utils.h:20` |
| sm_87 | Ampere GA10x | A16 | 1536 | `csrc/launch_bounds_utils.h:20` |
| sm_89 | Ada Lovelace | L40, L4, RTX 4090 | 1536 | `csrc/launch_bounds_utils.h:20` |
| sm_90 | Hopper | H100, H200 | 2048 | `csrc/launch_bounds_utils.h:28` |
| sm_90a | Hopper (PTX) | H100, H200 | 2048 | `cmake/utils.cmake:237` |
| sm_100 | Blackwell | B200, GB200 | 2048 | `csrc/launch_bounds_utils.h:29` |
| sm_100a | Blackwell (PTX) | B200 | 2048 | `CMakeLists.txt` |
| sm_100f | Blackwell (variant) | GB202 | — | `cmake/external_projects/qutlass.cmake:35` |
| sm_101 | Thor | — | — | `csrc/launch_bounds_utils.h:21` |
| sm_103 | Blackwell (variant) | — | 2048 | `csrc/launch_bounds_utils.h:29` |
| sm_103a | Blackwell (PTX variant) | — | — | `CMakeLists.txt` |
| sm_110 | Thor | — | — | `csrc/launch_bounds_utils.h:21` |
| sm_120 | Consumer Blackwell | RTX 5090 | 1536 | `csrc/launch_bounds_utils.h:21` |
| sm_120a | Consumer Blackwell (PTX) | — | — | `CMakeLists.txt:742` |
| sm_120f | Consumer Blackwell (PTX) | — | — | `cmake/external_projects/qutlass.cmake:35` |
| sm_121 | Consumer Blackwell | DGX Spark GB10 | 1536 | `csrc/launch_bounds_utils.h:21` |
| sm_121a | Consumer Blackwell (PTX) | DGX Spark GB10 | — | `CMakeLists.txt:742` |

### 1.2 CMake gencode构建配置

**核心构建参数 (CMakeLists.txt):**

| 内核 | 目标架构 | 行号 |
|------|---------|------|
| Marlin (通用) | `8.0+PTX; 12.0f` 或 `8.0+PTX; 12.0a; 12.1a` | 367-369 |
| Marlin SM75 | `7.5` | 372 |
| Marlin BF16 | `8.0+PTX; 9.0+PTX; 12.0f/a` | 375-377 |
| Marlin FP8 | `8.9; 12.0f/a` | 384-386 |
| Marlin Other | `7.5; 8.0+PTX` | 389 |
| ES MXFP8 Grouped MM | `10.0f; 11.0f` 或 `10.0a; 10.1a; 10.3a` | 500-502 |
| Machete | `9.0a` | 530 |
| DSV3 Fused A GEMM | `9.0a; 10.0f; 11.0f` 或 `9.0a; 10.0a; 10.1a; 10.3a` | 670-672 |
| AllSpark GPTQ | `8.0; 8.6; 8.7; 8.9` | 687 |
| Scaled MM (FP8) | `9.0a` / `12.0f` | 708, 740 |

**FlashInfer构建 (tools/flashinfer-build.sh):**

| CUDA版本 | TORCH_CUDA_ARCH_LIST |
|----------|---------------------|
| 基础 | `7.5 8.0 8.9` |
| CUDA 12 | `7.5 8.0 8.9 9.0a` |
| 扩展Blackwell | `7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0` |
| 扩展Thor | `7.5 8.0 8.9 9.0a 10.0f 11.0 12.0f` |

---

## 2. PTX汇编与内联PTX指令

### 2.1 指令类别总览

| 类别 | 指令 | 最低SM | 主要文件 |
|------|------|--------|---------|
| 异步拷贝 | `cp.async.cg.shared.global` | sm_80 | `csrc/quantization/marlin/marlin.cuh` |
| 异步提交 | `cp.async.commit_group` | sm_80 | 同上 |
| 异步等待 | `cp.async.wait_group` / `wait_all` | sm_80 | 同上 |
| 矩阵加载 | `ldmatrix.sync.aligned.m8n8` | sm_75 | `csrc/quantization/marlin/marlin_template.h` |
| 全局加载(获取) | `ld.global.acquire.gpu.b32` | sm_70 | `csrc/persistent_topk.cuh` |
| 全局加载(CG) | `ld.global.cg.v4.u32` | sm_70 | 同上 |
| 易失性加载 | `ld.volatile.global.v4.f32` | sm_70 | `csrc/minimax_reduce_rms_kernel.cu` |
| 释放存储 | `st.release.gpu.global.b32` | sm_70 | `csrc/persistent_topk.cuh` |
| 内存屏障 | `fence.acq_rel.gpu` | sm_70 | 多个文件 |
| 原子归约 | `red.relaxed.gpu.global.add.s32` | sm_70 | 多个文件 |
| 位操作 | `lop3.b32` | sm_70 | `csrc/quantization/marlin/dequant.h` |
| 字节置换 | `prmt.b32` | sm_70 | 同上 |
| 近似倒数 | `rcp.approx.ftz.f32` | sm_70 | `csrc/moe/mxfp8_moe/mxfp8_experts_quant.cuh` |
| 融合乘加 | `fma.rn.f32x2` | sm_70 | `csrc/moe/dsv3_router_gemm_float_out.cu` |
| 网格依赖控制 | `griddepcontrol.wait/launch_dependents` | sm_90 | 多个MOE内核 |
| LDSM | `ldsm_4` (ldmatrix封装) | sm_75 | `csrc/libtorch_stable/quantization/gptq_allspark/allspark_utils.cuh` |

### 2.2 关键PTX代码片段

**cp.async (sm_80+):**
```cuda
asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(gmem), "n"(bytes));
asm volatile("cp.async.commit_group;\n" ::);
asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
```

**ldmatrix (sm_75+):**
```cuda
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(r0) : "r"(addr));
```

**fence + atomic reduce (sm_70+):**
```cuda
asm volatile("fence.acq_rel.gpu;\n");
asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));
asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
```

**griddepcontrol (sm_90+):**
```cuda
asm volatile("griddepcontrol.wait;");
asm volatile("griddepcontrol.launch_dependents;");
```

---

## 3. SM架构守卫与条件编译

### 3.1 `__CUDA_ARCH__` 守卫模式

| 守卫模式 | SM阈值 | 用途 | 文件数 |
|---------|--------|------|--------|
| `__CUDA_ARCH__ < 750` | sm_75 | Marlin内核最低要求 | 4 |
| `__CUDA_ARCH__ == 750` | sm_75精确 | Turing专用MMA路径 | 8 |
| `__CUDA_ARCH__ < 800` | sm_80 | BF16/cp.async最低要求 | 8 |
| `__CUDA_ARCH__ >= 800` | sm_80 | BF16支持, cp.async | 6 |
| `__CUDA_ARCH__ < 890` | sm_89 | FP8 MMA可用性 | 2 |
| `__CUDA_ARCH__ >= 900` | sm_90 | Hopper特性(griddepcontrol, wgmma, TMA) | 6 |
| `__CUDA_ARCH__ >= 1000` | sm_100 | Blackwell特性 | 1 |
| `__CUDA_ARCH__ >= 610` | sm_61 | GGUF DP4A指令 | 1(22处) |
| `__CUDA_ARCH__ >= 700` | sm_70 | Volta+内存操作 | 1 |

### 3.2 Hopper (sm_90) 特有守卫

DSV3 Fused A GEMM 使用 **16处** `__CUDA_ARCH__ >= 900` 守卫，涵盖:
- TMA加载/存储/预取
- wgmma (Warp Group MMA)
- 线程组屏障
- 集群同步

---

## 4. Triton内核SM配置

### 4.1 Triton内核统计

vLLM中包含 **~505个 `@triton.jit` 装饰的内核**。

### 4.2 SM敏感的Triton内核

| 内核 | 文件 | SM相关逻辑 |
|------|------|-----------|
| TurboQuant Decode | `triton_turboquant_decode.py` | FP8格式: e4b15(sm<89) vs e4nv(sm>=90) |
| TurboQuant Store | `triton_turboquant_store.py` | 同上 |
| Reshape & Cache Flash | `triton_reshape_and_cache_flash.py` | FP8/INT8 KV cache量化 |
| Decode Attention | `triton_decode_attention.py` | FP8 KV cache读取 |
| Merge Attention States | `triton_merge_attn_states.py` | 平台FP8 dtype |
| MLA Sparse | `xpu_mla_sparse.py` | Block配置 |

### 4.3 FP8格式选择逻辑

```
SM < 89 (Ampere/Ada)  → fp8e4b15 (软件模拟E4格式)
SM >= 89 (Ada原生)    → 原生FP8 MMA
SM >= 90 (Hopper+)    → fp8e4nv (原生NV格式)
```

---

## 5. GPU架构代号与产品名

详见 [part5_gpu_arch_names.md](part5_gpu_arch_names.md)

### 5.1 架构代号映射

| 代号 | SM | 产品 | vLLM中的关键引用 |
|------|-----|------|-----------------|
| Pascal | 6.0 | — | BF16不支持 |
| Volta | 7.0/7.2 | V100 | 2048线程/SM |
| Turing | 7.5 | T4, RTX 2080 | 1024线程/SM, IS_TURING标志 |
| Ampere | 8.0 | A100 | 2048线程/SM, BF16支持 |
| Ampere GA10x | 8.6/8.7 | A10, A40 | 1536线程/SM |
| Ada Lovelace | 8.9 | L40, RTX 4090 | 1536线程/SM, FP8 MMA |
| Hopper | 9.0 | H100, H200 | 2048线程/SM, TMA/wgmma/griddepcontrol |
| Blackwell | 10.0 | B200, GB200 | 2048线程/SM, MXFP8/MXFP4/NVFP4 |
| Thor | 10.1/11.0 | — | 1536线程/SM |
| Consumer Blackwell | 12.0/12.1 | RTX 5090, DGX Spark GB10 | 1536线程/SM, NVFP4/MXFP4原生, FlashInfer b12x |

### 1.3 SM12x (Consumer Blackwell) 详细实现

vLLM对SM12x有大量专门实现:

**CUTLASS SM120内核 (6个专用文件):**
- `scaled_mm_sm120_fp8.cu` / `_dispatch.cuh` — FP8 GEMM分发 (4种配置: default/M64/M32/M16)
- `scaled_mm_blockwise_sm120_fp8.cu` / `_dispatch.cuh` — Blockwise FP8 (3种配置: default/pingpong/swapab)
- `scaled_mm_c3x_sm120.cu` — SM120 FP8入口 (INT8不支持SM120)

**NVFP4 SM120内核 (4个专用文件):**
- `nvfp4_scaled_mm_sm120_kernels.cu` — SM120 NVFP4 GEMM (`CUTLASS_ARCH_MMA_SM120_SUPPORTED`)
- `nvfp4_blockwise_moe_kernel.cu` — SM120 NVFP4 MoE (仅支持BF16输出)
- `nvfp4_scaled_mm_entry.cu` — `ENABLE_NVFP4_SM120` 宏控制
- `nvfp4_quant_entry.cu` — NVFP4量化入口

**FlashInfer b12x后端:**
- `flashinfer_b12x_fused_moe` — CuteDSL融合MoE (SM120/SM121)
- `FlashInferB12xNvFp4LinearKernel` — NVFP4 GEMM (CuTe DSL warp-level MMA)
- 需要 `has_device_capability(120)` 检测

**CMake构建标志:**
- `-DENABLE_SCALED_MM_SM120=1` (CMakeLists.txt:754)
- `-DENABLE_NVFP4_SM120=1`
- `-DENABLE_CUTLASS_MOE_SM120=1`
- 架构: `12.0f` (RTX 5090), `12.0a`, `12.1a` (DGX Spark)

---

## 6. Warp级与Tensor Core指令

### 6.1 Marlin MMA指令矩阵

| SM | 指令 | A类型 | B类型 | 形状 |
|----|------|-------|-------|------|
| sm_75 | HMMA.884 | FP16 | FP16 | 8x8x4 |
| sm_80+ | HMMA.1688 | FP16 | FP16 | 16x8x8 |
| sm_80+ | HMMA.1688 | BF16 | BF16 | 16x8x8 |
| sm_80+ | IMMA | INT8 | INT8/UINT8 | 16x8x32 |
| sm_80+ | HMMA.16816 | FP16/BF16 | FP16/BF16 | 16x8x16 |
| sm_89+ | FP8 MMA | E4M3 | E4M3/BF16 | 16x8x32 |
| sm_89+ | FP8 MMA | E5M2 | E5M2/BF16 | 16x8x32 |

### 6.2 Hopper (sm_90) 特有指令

| 指令 | 用途 | 文件 |
|------|------|------|
| TMA load/store | 张量内存加速器异步批量拷贝 | `csrc/libtorch_stable/dsv3_fused_a_gemm.cu` |
| wgmma | Warp Group矩阵乘累加 (4个Warp) | 同上 |
| mbarrier | 线程组屏障同步 | 同上 |
| griddepcontrol | 持久内核网格依赖控制 | 多个MOE内核 |

### 6.3 ROCm WMMA (AMD RDNA3)

| 内核 | 形状 | Wave数 | 文件 |
|------|------|--------|------|
| `gemm_q4_wmma_16x16_1w` | 16x16 | 1 | `csrc/rocm/q_gemm_rdna3_wmma.cu:372` |
| `gemm_q4_wmma_32x16_2w` | 32x16 | 2 | 同上:656 |
| `gemm_q4_wmma_64x16_4w` | 64x16 | 4 | 同上:916 |
| `gemm_q4_wmma_64x32_4w` | 64x32 | 4 | 同上:1157 |
| `gemm_q4_wmma_64x64_4w` | 64x64 | 4 | 同上:1411 |
| `gemm_q4_wmma_128x64_k16` | 128x64 | 4 | 同上:1610 |
| `gemm_q4_wmma_128x64_k32` | 128x64 | 4 | 同上:1809 |

---

## 7. NCCL与通信原语

### 7.1 vLLM通信架构

```
vLLM通信层
├── NCCL (默认后端)
│   ├── PyNcclCommunicator (Python封装)
│   ├── P2P NCCL Engine (KV传输)
│   ├── Weight Transfer (权重广播)
│   └── torch.distributed (集合操作)
├── Custom All-Reduce
│   ├── IPC-based (CUDA进程间通信)
│   └── 可通过 --disable-custom-all-reduce 禁用
└── Ray DAG Channel
    ├── nccl (需要cupy)
    └── shm (共享内存)
```

### 7.2 关键NCCL API使用

| API | 文件 | 用途 |
|-----|------|------|
| `ncclGetUniqueId` | `p2p_nccl_engine.py:217` | P2P通信握手 |
| `ncclCommInitRank` | `p2p_nccl_engine.py:224` | 初始化通信器 |
| `ncclSend` | `p2p_nccl_engine.py:598` | 发送张量 |
| `ncclRecv` | `p2p_nccl_engine.py:617` | 接收张量 |
| `packed_nccl_broadcast_producer` | `packed_tensor.py:136` | 权重广播(生产者) |
| `packed_nccl_broadcast_consumer` | `packed_tensor.py:182` | 权重广播(消费者) |

### 7.3 环境变量

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `VLLM_NCCL_SO_PATH` | None | NCCL库路径(绕过2.19+bug) |
| `VLLM_NCCL_INCLUDE_PATH` | None | NCCL头文件路径 |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `"auto"` | Ray DAG通道类型 |

---

## 8. 量化与精度SM特性

### 8.1 量化方法SM需求矩阵

| 量化方法 | 最低SM | 数据类型 | 内核 | 构建架构 |
|---------|--------|---------|------|---------|
| GGUF (DP4A) | sm_61 | INT4/INT8多种 | GGUF CUDA | `__CUDA_ARCH__ >= 610` |
| Marlin INT4/INT8 | sm_75 | INT4/INT8 | Marlin CUDA | `7.5; 8.0+PTX` |
| Marlin BF16 | sm_80 | BF16 | Marlin CUDA | `8.0+PTX; 9.0+PTX` |
| Marlin FP8 | sm_89 | E4M3/E5M2 | Marlin CUDA | `8.9` |
| FP8 KV Cache | sm_80 | float8_e4m3fn | Triton | 运行时检测 |
| FP8 TurboQuant | sm_80 | fp8e4b15/e4nv | Triton | 运行时SM检测 |
| FP8 Scaled MM | sm_90a | FP8 | Scaled MM CUDA | `9.0a` |
| Machete W4A16 | sm_90a | W4A16 | Machete CUDA | `9.0a` |
| DeepGEMM FP8 | sm_90 | FP8 | DeepGEMM | `9.0+` |
| MXFP8 | sm_100 | MXFP8 | ES Grouped MM | `10.0f; 10.0a` |
| NVFP4 KV | sm_100 | NVFP4 | Custom+FlashInfer | `10.0+` |
| MXFP4 | sm_100 | MXFP4 | FlashInfer | `10.0+` |
| AllSpark W8A16 | sm_80 | W8A16 | AllSpark CUDA | `8.0; 8.6; 8.7; 8.9` |
| torchao | sm_89 | 各种 | torchao | 运行时检测 |
| INT8 Per-Token-Head | sm_80 | INT8 | Custom | 运行时检测 |

### 8.2 FP8格式选择 (SM依赖)

```
sm_80 ~ sm_88 (Ampere):  fp8e4b15 (软件模拟)
sm_89 (Ada Lovelace):    原生FP8 E4M3 MMA
sm_90+ (Hopper):         fp8e4nv (原生NV格式, TMA加速)
sm_100+ (Blackwell):     MXFP8/MXFP4/NVFP4 原生支持
```

### 8.3 量化位宽效率

| 方法 | 位/权重 | 相对FP16压缩比 |
|------|---------|---------------|
| FP8 / FBGEMM_FP8 / PTPC_FP8 | 1.0 | 2x |
| ModelOpt MXFP8 | 1.0 | 2x |
| MXFP4 / AWQ / GPTQ / Petit NVFP4 | 0.5 | 4x |
| Experts INT8 | 1.0 | 2x |

---

## 附录: 文件索引

| 分卷 | 文件名 | 内容 |
|------|--------|------|
| Part 1 | `part1_sm_arch_flags.md` | SM架构标志、gencode、构建配置 |
| Part 2 | `part2_ptx_assembly.md` | PTX汇编指令、内联PTX、cp.async、ldmatrix |
| Part 3 | `part3_sm_guards.md` | `__CUDA_ARCH__`条件编译守卫 |
| Part 4 | `part4_triton_sm.md` | Triton内核SM配置 |
| Part 5 | `part5_gpu_arch_names.md` | GPU架构代号与产品名映射 |
| Part 6 | `part6_warp_tensor_core.md` | Warp级MMA、Tensor Core、TMA、wgmma |
| Part 7 | `part7_nccl_comm.md` | NCCL通信、自定义All-Reduce、P2P |
| Part 8 | `part8_quant_sm.md` | 量化方法SM需求、FP8/NVFP4/MXFP8 |
