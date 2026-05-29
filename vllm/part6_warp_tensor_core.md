# Part 6: Warp-Level & Tensor Core Instructions in vLLM

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. Summary of Instruction Categories

| Category | Instructions | SM Min | Primary Files |
|----------|-------------|--------|---------------|
| WMMA (Warp Matrix Multiply-Accumulate) | `wmma_mma`, `wmma::mma_sync` | sm_70 | `csrc/quantization/marlin/marlin_mma.h` |
| HMMA (PTX MMA for FP16) | `HMMA.884`, `HMMA.1688` | sm_70/75 | `csrc/quantization/marlin/marlin_mma.h` |
| IMMA (PTX MMA for INT8) | `IMMA` | sm_75 | `csrc/quantization/marlin/marlin_mma.h` |
| ldmatrix | `ldmatrix.sync.aligned.m8n8` | sm_75 | `csrc/quantization/marlin/marlin_template.h`, `csrc/moe/marlin_moe_wna16/marlin_template.h` |
| cp.async | `cp.async.cg.shared.global` | sm_80 | `csrc/quantization/marlin/marlin.cuh`, `csrc/quantization/activation_kernels.cu` |
| Grid Dependency Control | `griddepcontrol.wait/launch_dependents` | sm_90 | `csrc/moe/topk_softplus_sqrt_kernels.cu`, etc. |
| TMA (Tensor Memory Accelerator) | TMA load/store | sm_90 | `csrc/libtorch_stable/dsv3_fused_a_gemm.cu` |
| wgmma (Warp Group MMA) | Hopper wgmma | sm_90 | `csrc/libtorch_stable/dsv3_fused_a_gemm.cu` |
| LDSM (Load Shared Matrix) | `ldsm_4` wrapper | sm_75 | `csrc/libtorch_stable/quantization/gptq_allspark/allspark_utils.cuh` |
| ROCm WMMA | `__builtin_amdgcn_wmma_*` | RDNA3 | `csrc/rocm/q_gemm_rdna3_wmma.cu` |

---

## 2. Marlin MMA (Matrix Multiply-Accumulate) - Detailed

**File:** `csrc/quantization/marlin/marlin_mma.h`

### 2.1 Turing (sm_75) HMMA.884 - FP16 8x8x4

```cpp
// Line 22: #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
// HMMA.884 for FP16 on Turing
asm volatile(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15};\n"
    ...
);
```

### 2.2 Ampere+ (sm_80) HMMA.1688 - FP16 16x8x8

```cpp
// Line 30: sm_80+ path
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    ...
);
```

### 2.3 HMMA.1688 - BF16 16x8x8

```cpp
// Line 38: BF16 variant
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
    ...
);
```

### 2.4 IMMA - INT8 variants

```cpp
// Line 49: INT8 MMA
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
    ...
);

// Line 54: Another INT8 variant
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32 "
    ...
);
```

### 2.5 FP8 E4M3 MMA (sm_89+)

```cpp
// Line 86: FP8 E4M3
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.bf16.f32 "
    ...
);

// Line 96: FP8 E4M3 variant
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    ...
);
```

### 2.6 FP8 E5M2 MMA (sm_89+)

```cpp
// Line 105: FP8 E5M2
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.bf16.f32 "
    ...
);

// Line 110: FP8 E5M2 variant
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
    ...
);
```

### 2.7 Additional MMA Variants

| Line | Instruction | Dtype A | Dtype B | Shape |
|------|------------|---------|---------|-------|
| 61 | `mma.sync.m16n8k16` | bf16 | bf16 | 16x8x16 |
| 70 | `mma.sync.m16n8k16` | f16 | f16 | 16x8x16 |
| 78 | `mma.sync.m16n8k16` | bf16 | bf16 | 16x8x16 |
| 115 | `mma.sync.m16n8k32` | e4m3 | e4m3 | 16x8x32 |
| 120 | `mma.sync.m16n8k32` | e5m2 | e5m2 | 16x8x32 |
| 126 | `mma.sync.m16n8k32` | e4m3 | bf16 | 16x8x32 |
| 156-258 | Multiple 2-element variants | Various | Various | Various |

---

## 3. ldmatrix (Load Matrix from Shared Memory)

### 3.1 NVIDIA ldmatrix

**File:** `csrc/quantization/marlin/marlin_template.h`

```cpp
// Line 87: Load 2 8x8 matrices (x2)
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(r0), "=r"(r1) : "r"(smem_addr));

// Line 92: Load 1 8x8 matrix (x1)
asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
    : "=r"(r0) : "r"(smem_addr));
```

**File:** `csrc/moe/marlin_moe_wna16/marlin_template.h`

```cpp
// Line 95-104: Same ldmatrix patterns for MOE
```

### 3.2 AllSpark LDSM Wrapper

**File:** `csrc/libtorch_stable/quantization/gptq_allspark/allspark_utils.cuh`

```cpp
// Line 244-246: ldsm_4 - loads 4 matrix elements
__device__ __forceinline__ void ldsm_4(T& r0, T& r1, T& r2, T& r3, ...) {
  static_assert(sizeof(T) == 4, "ldsm_4: invalid T");
}
```

**Used in:** `csrc/libtorch_stable/quantization/gptq_allspark/allspark_qgemm_w8a16.cu:289`

---

## 4. Hopper (sm_90) Specific: TMA & wgmma

### 4.1 DeepSeek V3 Fused A GEMM

**File:** `csrc/libtorch_stable/dsv3_fused_a_gemm.cu`

| Line | Feature | Description |
|------|---------|-------------|
| 63-579 | `#if __CUDA_ARCH__ >= 900` | All Hopper-specific code guarded |
| 87 | TMA load | Tensor Memory Accelerator for async bulk copy |
| 98 | TMA descriptor | TMA descriptor setup for multidimensional access |
| 121 | TMA store | Async bulk store to global memory |
| 133 | wgmma | Warp Group Matrix Multiply Accumulate (4 warps) |
| 149 | wgmma pipeline | Overlapped computation with TMA loads |
| 169 | Barrier | Thread group barrier for TMA completion |
| 180 | Cluster sync | Cluster-level synchronization |
| 220-234 | Barrier operations | mbarrier_wait, mbarrier_arrive |
| 315-330 | TMA prefetch | Prefetch TMA descriptors for next tile |
| 418-443 | Main loop | Overlapped wgmma + TMA load pipeline |
| 486-579 | Kernel launch | Multiple kernel variants for different GEMM shapes |

---

## 5. Grid Dependency Control (sm_90+ Persistent Kernels)

| File | Line | Instruction | Context |
|------|------|------------|---------|
| `csrc/moe/topk_softplus_sqrt_kernels.cu` | 171 | `griddepcontrol.wait` | Wait for prologue completion |
| `csrc/moe/topk_softplus_sqrt_kernels.cu` | 298, 423 | `griddepcontrol.launch_dependents` | Launch dependent kernels |
| `csrc/moe/grouped_topk_kernels.cu` | 562 | `griddepcontrol.wait` | Persistent TopK prologue |
| `csrc/moe/grouped_topk_kernels.cu` | 607, 667, 681, 882 | `griddepcontrol.launch_dependents` | Launch dependent kernels |
| `csrc/minimax_reduce_rms_kernel.cu` | 249, 384 | `griddepcontrol.wait` | RMS norm prologue |
| `csrc/minimax_reduce_rms_kernel.cu` | 313, 596 | `griddepcontrol.launch_dependents` | Launch dependent |
| `csrc/moe/dsv3_router_gemm_float_out.cu` | 84 | `griddepcontrol.wait` | DSV3 router prologue |
| `csrc/moe/dsv3_router_gemm_float_out.cu` | 168 | `griddepcontrol.launch_dependents` | Launch dependent |
| `csrc/moe/dsv3_router_gemm_bf16_out.cu` | 83, 168 | Same pattern | Same pattern |

---

## 6. ROCm WMMA (AMD RDNA3)

**File:** `csrc/rocm/q_gemm_rdna3_wmma.cu`

| Line | Feature | Description |
|------|---------|-------------|
| 16 | `v_wmma_f32_16x16x16_{f16,bf16}_w32` | 16x16x16 GEMM instruction |
| 52 | `namespace gptq_rdna3_wmma` | RDNA3 WMMA namespace |
| 58 | `__builtin_amdgcn_wmma_*` | AMD WMMA intrinsics |
| 239 | `compute_wmma_k_split` | K-dimension split heuristic |
| 266 | `compute_wmma_k_split_mn` | M/N-aware split |
| 290 | `wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c)` | FP16 WMMA |
| 293 | `wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c)` | BF16 WMMA |
| 372 | `gemm_q4_wmma_kernel_16x16_1w` | 16x16 1-wave kernel |
| 656 | `gemm_q4_wmma_kernel_32x16_2w` | 32x16 2-wave kernel |
| 916 | `gemm_q4_wmma_kernel_64x16_4w` | 64x16 4-wave kernel |
| 1157 | `gemm_q4_wmma_kernel_64x32_4w` | 64x32 4-wave kernel |
| 1411 | `gemm_q4_wmma_kernel_64x64_4w` | 64x64 4-wave kernel |
| 1610 | `gemm_q4_wmma_kernel_128x64_k16` | 128x64 K=16 kernel |
| 1809 | `gemm_q4_wmma_kernel_128x64_k32` | 128x64 K=32 kernel |

### WMMA Pipeline Optimization Notes (from source comments):

| Line | Note |
|------|------|
| 628 | `one v_wmma every ~30-40 cycles (16-cycle wmma latency + dequant + LDS)` |
| 637 | `scheduler can keep the WMMA pipeline full by interleaving wmmas` |
| 889 | `~28% of WMMA peak. Bottleneck is wmma issue rate per resident wave` |
| 894 | `Doubling resident wave count (2→4) targets ~2× wmma throughput` |
| 902 | `Waves 1-3 idle through dequant but their wmmas can run concurrently` |
| 1044 | `4 wmmas in flight per block per K-iter` |
| 1137 | `4 waves × 2 wmmas = 8 wmmas in flight per K-iter` |
| 1392 | `4 waves × 4 wmmas = 16 wmmas in flight, pipeline fully saturated` |

---

## 7. Marlin Kernel Architecture Summary

The Marlin quantization kernel in vLLM is the most instruction-dense CUDA kernel, using:

| Instruction | Purpose | SM Min |
|------------|---------|--------|
| `ldmatrix` | Load weight fragments from shared memory | sm_75 |
| `cp.async` | Async copy weights from global→shared | sm_80 |
| `mma.sync` | Warp-level matrix multiply | sm_70 |
| `lop3.b32` | Bit manipulation for dequantization | sm_70 |
| `prmt.b32` | Byte permute for weight rearrangement | sm_70 |
| `fence.acq_rel.gpu` | Memory ordering for lock-based sync | sm_70 |
| `red.relaxed.gpu.global.add` | Atomic reduction for output accumulation | sm_70 |
| `ld.global.acquire.gpu` | Lock acquisition with acquire semantics | sm_70 |
| `st.release.gpu` | Lock release with release semantics | sm_70 |

### Marlin SM Dispatch:

| SM | MMA Type | Data Types |
|----|---------|-----------|
| sm_75 | HMMA.884 (m8n8k4) | FP16 only |
| sm_80+ | HMMA.1688 (m16n8k8) | FP16, BF16 |
| sm_80+ | IMMA (m16n8k32) | INT8 |
| sm_89+ | FP8 MMA (m16n8k32) | E4M3, E5M2 |
