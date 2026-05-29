# Part 3: SM Guards & `__CUDA_ARCH__` Conditionals in vLLM

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. Summary of `__CUDA_ARCH__` Guard Patterns

| Guard Pattern | SM Threshold | Purpose | Primary Files |
|--------------|-------------|---------|---------------|
| `__CUDA_ARCH__ < 750` | sm_75 | Minimum for Marlin kernels | `csrc/quantization/marlin/*` |
| `__CUDA_ARCH__ == 750` | sm_75 exactly | Turing-specific MMA paths | `csrc/quantization/marlin/marlin_mma.h` |
| `__CUDA_ARCH__ < 800` | sm_80 | Minimum for BF16, cp.async | `csrc/type_convert.cuh`, `csrc/quantization/w8a8/fp8/*` |
| `__CUDA_ARCH__ >= 800` | sm_80 | BF16 support, cp.async enabled | `csrc/quantization/activation_kernels.cu`, `csrc/quantization/gguf/*` |
| `__CUDA_ARCH__ < 890` | sm_89 | FP8 MMA availability | `csrc/quantization/marlin/marlin_template.h` |
| `__CUDA_ARCH__ >= 900` | sm_90 | Hopper features (griddepcontrol, wgmma) | `csrc/moe/*`, `csrc/minimax_reduce_rms_kernel.cu` |
| `__CUDA_ARCH__ >= 1000` | sm_100 | Blackwell features | `csrc/moe/mxfp8_moe/mxfp8_experts_quant.cuh` |
| `__CUDA_ARCH__ >= 610` | sm_61 | GGUF DP4A instruction | `csrc/quantization/gguf/vecdotq.cuh` |
| `__CUDA_ARCH__ >= 700` | sm_70 | Volta+ features (fence, ld.global) | `csrc/persistent_topk.cuh` |

---

## 2. Detailed `__CUDA_ARCH__` Guards by File

### 2.1 Marlin Quantization Kernels

**File:** `csrc/quantization/marlin/marlin.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 40 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750` | Early exit for pre-Turing GPUs |

**File:** `csrc/quantization/marlin/marlin.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 60 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` | Skip BF16 paths on pre-Ampere |

**File:** `csrc/quantization/marlin/marlin_template.h`

| Line | Guard | Effect |
|------|-------|--------|
| 39 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750` | Minimum arch check |
| 283 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890` | Use non-FP8 MMA on pre-Ada |
| 288 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing-specific MMA path |
| 294 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Another Turing-specific path |

**File:** `csrc/quantization/marlin/marlin_mma.h`

| Line | Guard | Effect |
|------|-------|--------|
| 22 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing HMMA.884 (FP16) |
| 47 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing HMMA variant |
| 104 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing MMA path |
| 154 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Another Turing MMA |
| 179 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing-specific |
| 236 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing-specific |

**File:** `csrc/quantization/marlin/marlin_dtypes.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 73 | `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800` | BF16 types only on sm_80+ |

**File:** `csrc/quantization/marlin/dequant.h`

| Line | Guard | Effect |
|------|-------|--------|
| 70 | `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750` | lop3/prmt only on sm_75+ |

### 2.2 MOE WNA16 Marlin Kernels

**File:** `csrc/moe/marlin_moe_wna16/marlin_template.h`

| Line | Guard | Effect |
|------|-------|--------|
| 39 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750` | Minimum arch check |
| 296 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890` | FP8 MMA guard |
| 301 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing MMA path |
| 310 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Another Turing path |
| 493 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` | Turing-specific dequant |

**File:** `csrc/moe/moe_wna16_utils.h`

| Line | Guard | Effect |
|------|-------|--------|
| 49 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800` | BF16 dequant on sm_80+ |
| 146 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800` | BF16 store on sm_80+ |

**File:** `csrc/moe/moe_wna16.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 28 | `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800` | Early exit for pre-Ampere |
| 220 | `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800` | Another pre-Ampere guard |

### 2.3 FP8 Quantization Kernels

**File:** `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 27 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` | BF16 conversion not available on pre-Ampere |
| 194 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` | Same guard, different function |
| 485 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` | Same guard, different function |

### 2.4 Activation Kernels

**File:** `csrc/quantization/activation_kernels.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 153 | `#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800` | cp.async requires CUDA 11+ and sm_80+ |
| 169 | Same | cp.async.commit_group |
| 177 | Same | cp.async.wait_group |
| 185 | Same | cp.async.wait_all |

### 2.5 Type Conversion

**File:** `csrc/type_convert.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 72 | `#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || defined(USE_ROCM)` | BF16 conversion on sm_80+ or ROCm |

### 2.6 GGUF Quantization

**File:** `csrc/quantization/gguf/vecdotq.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 48-470 | `#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM` | DP4A instruction for dot product (22 occurrences) |

**File:** `csrc/quantization/gguf/ggml-common.h`

| Line | Guard | Effect |
|------|-------|--------|
| 1086 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800` | BF16 support for GGUF |

### 2.7 Hopper (sm_90) Grid Dependency Control

**File:** `csrc/moe/topk_softplus_sqrt_kernels.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 170 | `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))` | griddepcontrol.wait |
| 297 | Same | griddepcontrol.launch_dependents |
| 422 | Same | griddepcontrol.launch_dependents |

**File:** `csrc/moe/grouped_topk_kernels.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 561 | `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))` | griddepcontrol.wait |
| 606 | Same | griddepcontrol.launch_dependents |
| 667 | Same | griddepcontrol.launch_dependents |
| 681 | Same | griddepcontrol.launch_dependents |
| 882 | Same | griddepcontrol.launch_dependents |

**File:** `csrc/minimax_reduce_rms_kernel.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 196 | `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))` | griddepcontrol.wait |
| 248 | Same | griddepcontrol.wait |
| 312 | Same | griddepcontrol.launch_dependents |
| 355 | Same | griddepcontrol.launch_dependents |
| 383 | Same | griddepcontrol.wait |
| 595 | Same | griddepcontrol.launch_dependents |

**File:** `csrc/moe/dsv3_router_gemm_float_out.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 83 | `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))` | griddepcontrol.wait |
| 168 | Same | griddepcontrol.launch_dependents |

**File:** `csrc/moe/dsv3_router_gemm_bf16_out.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 83 | `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))` | griddepcontrol.wait |
| 168 | Same | griddepcontrol.launch_dependents |

### 2.8 DeepSeek V3 Fused A GEMM (sm_90)

**File:** `csrc/libtorch_stable/dsv3_fused_a_gemm.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 63 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` | wgmma / TMA path |
| 87 | Same | Hopper-specific GEMM |
| 98 | Same | Hopper TMA load |
| 121 | Same | Hopper store |
| 133 | Same | Hopper MMA |
| 149 | Same | Hopper epilogue |
| 169 | Same | Hopper pipeline |
| 180 | Same | Hopper barrier |
| 220 | Same | Hopper barrier sync |
| 234 | Same | Hopper commit |
| 315 | Same | Hopper wait |
| 330 | Same | Hopper TMA prefetch |
| 418 | Same | Hopper main loop |
| 443 | Same | Hopper epilogue store |
| 486 | Same | Hopper kernel body |
| 579 | Same | Hopper kernel launch |

### 2.9 Persistent TopK (sm_70)

**File:** `csrc/persistent_topk.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 599 | `#if (__CUDA_ARCH__ >= 700)` | ld.global.acquire on Volta+ |
| 610 | Same | ld.cg.global on Volta+ |
| 622 | Same | fence + red + st.release on Volta+ |

### 2.10 Blackwell (sm_100)

**File:** `csrc/moe/mxfp8_moe/mxfp8_experts_quant.cuh`

| Line | Guard | Effect |
|------|-------|--------|
| 268 | `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000` | Blackwell-specific quantization |

### 2.11 Fused QK Norm Rope Kernel (sm_80)

**File:** `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 117 | `#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)` | Requires sm_80+ |
| 303 | Same | BF16 kernel body |
| 319 | Same | FP16 kernel body |
| 534 | Same | Kernel dispatch |

### 2.12 Fused DeepSeek V4 (sm_80)

**File:** `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`

| Line | Guard | Effect |
|------|-------|--------|
| 384 | `// no-op stub for sm_70/sm_75` | Stub for pre-Ampere |
| 564 | `// bf16 on pre-Ampere (sm_70/sm_75)` | BF16 unavailable |
| 569 | `"requires sm_80+"` | Runtime error for sm_70/sm_75 |
