# Part 1: SM Architecture Flags & Compute Capability References in vLLM

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. Summary: All SM Versions Found

| SM Version | Architecture | Key Products | Threads/SM | First Seen |
|------------|-------------|-------------|-----------|------------|
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

### SM12x 产品映射 (源码注释)

**File:** `csrc/cutlass_extensions/common.hpp`

```cpp
// Line 128: SM12x family includes SM120 (RTX 5090) and SM121 (DGX Spark GB10)
```

**File:** `csrc/quantization/marlin/generate_kernels.py`

```cpp
// Line 18: SM89 and the SM12x family (SM120 RTX 5090, SM121 DGX Spark GB10)
```

---

## 2. CMake Build System - gencode Flags

### 2.1 CMakeLists.txt Main Build Configuration

**File:** `CMakeLists.txt`

| Line | gencode / Arch Flag | Purpose |
|------|---------------------|---------|
| 180-181 | `clear_cuda_arches` / `extract_unique_cuda_archs_ascending` | Extract CUDA arch flags from CMAKE_CUDA_FLAGS |
| 185 | `cuda_archs_loose_intersection(CUDA_ARCHS ...)` | Filter supported arch list |
| 194 | `override_gpu_arches(VLLM_GPU_ARCHES ...)` | Override GPU arches for build |
| 268 | `set_gencode_flags_for_srcs` | Set per-file gencode flags |
| 355 | `set_gencode_flags_for_srcs` | Additional gencode flags |
| 367 | `MARLIN_ARCHS "8.0+PTX;12.0f"` | Marlin kernel: sm_80+ with PTX, sm_120f |
| 369 | `MARLIN_ARCHS "8.0+PTX;12.0a;12.1a"` | Marlin kernel: alternative arch variants |
| 372 | `MARLIN_SM75_ARCHS "7.5"` | Marlin sm_75 specific kernels |
| 375 | `MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX;12.0f"` | Marlin BF16 kernels for sm_80/sm_90/sm_120f |
| 377 | `MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX;12.0a;12.1a"` | Marlin BF16 alternative variants |
| 384 | `MARLIN_FP8_ARCHS "8.9;12.0f"` | Marlin FP8 kernels for sm_89/sm_120f |
| 386 | `MARLIN_FP8_ARCHS "8.9;12.0a;12.1a"` | Marlin FP8 alternative variants |
| 389 | `MARLIN_OTHER_ARCHS "7.5;8.0+PTX"` | Marlin other precision: sm_75/sm_80 |
| 500 | `ES_MXFP8_GROUPED_MM_ARCHS "10.0f;11.0f"` | Expert-sparse MXFP8: sm_100f/sm_110f |
| 502 | `ES_MXFP8_GROUPED_MM_ARCHS "10.0a;10.1a;10.3a"` | Expert-sparse MXFP8 alternative |
| 530 | `MACHETE_ARCHS "9.0a"` | Machete quantization kernel: sm_90a |
| 670-672 | `DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0f;11.0f"` / `"9.0a;10.0a;10.1a;10.3a"` | DeepSeek V3 fused GEMM |
| 687 | `ALLSPARK_ARCHS "8.0;8.6;8.7;8.9"` | AllSpark GPTQ kernels |
| 708 | `SCALED_MM_ARCHS "9.0a"` | Scaled MM (FP8 matmul): sm_90a |
| 740 | `SCALED_MM_ARCHS "12.0f"` | Scaled MM: sm_120f |

### 2.2 cmake/utils.cmake - gencode Utilities

**File:** `cmake/utils.cmake`

| Line | Function/Macro | Description |
|------|---------------|-------------|
| 202 | `gencode_to_cmake_version` | Convert gencode version to cmake version |
| 210-226 | `clear_cuda_arches(CUDA_ARCH_FLAGS)` | Extract and clear `-gencode` flags from CMAKE_CUDA_FLAGS |
| 237 | `extract_unique_cuda_archs_ascending` | Extract unique CUDA archs sorted ascending |
| 258-283 | `set_gencode_flag_for_srcs` | Set `-gencode arch=compute_XX,code=sm_XX` for specific source files |
| 296-331 | `set_gencode_flags_for_srcs` | Set gencode flags for a list of source files |
| 340-368 | `cuda_archs_loose_intersection` | Compute loose intersection of CUDA arch lists |
| 487 | `override_gpu_arches` | Override GPU architectures for build |

### 2.3 External Project gencode Flags

**File:** `cmake/external_projects/qutlass.cmake`

| Line | Arch Flag | Purpose |
|------|-----------|---------|
| 35 | `"10.0f;12.0f"` | QUTLASS for sm_100f/sm_120f |
| 37 | `"12.0a;12.1a;10.0a;10.3a"` | QUTLASS alternative variants |

**File:** `cmake/external_projects/flashmla.cmake`

| Line | Arch Flag | Purpose |
|------|-----------|---------|
| 61 | `compute_10x family` | FlashMLA supports all sm_10x (Blackwell) |
| 68 | `cuda_archs_loose_intersection(FLASH_MLA_ARCHS ...)` | Filter supported archs |

**File:** `cmake/external_projects/deepgemm.cmake`

| Line | Arch Flag | Purpose |
|------|-----------|---------|
| 50 | `cuda_archs_loose_intersection(DEEPGEMM_ARCHS ...)` | DeepGEMM arch filtering |

### 2.4 FlashInfer Build Script

**File:** `tools/flashinfer-build.sh`

| Line | TORCH_CUDA_ARCH_LIST | CUDA Version |
|------|---------------------|-------------|
| 30 | `"7.5 8.0 8.9"` | Base (CUDA 11) |
| 32 | `"7.5 8.0 8.9 9.0a"` | CUDA 12 |
| 35 | `"7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0"` | Extended Blackwell |
| 38 | `"7.5 8.0 8.9 9.0a 10.0f 11.0 12.0f"` | Extended Thor/Blackwell |

---

## 3. Launch Bounds Per SM (Threads/SM)

**File:** `csrc/launch_bounds_utils.h`

```cpp
// Line 16-17: 1024 thr/SM: Turing (sm_75)
// Line 20-21: 1536 thr/SM: Ampere GA10x (sm_86/87), Ada (sm_89),
//             GB20x consumer (sm_120/121), Thor (sm_101 or sm_110)
// Line 28-29: 2048 thr/SM: Volta (sm_70/72), Ampere GA100 (sm_80),
//             Hopper (sm_90), Blackwell (sm_100/103)
```

| Threads/SM | SM Versions |
|-----------|-------------|
| 1024 | sm_75 (Turing) |
| 1536 | sm_86, sm_87, sm_89, sm_101, sm_110, sm_120, sm_121 |
| 2048 | sm_70, sm_72, sm_80, sm_90, sm_100, sm_103 |

---

## 4. Python-Level SM Detection & Capability Checks

### 4.1 Platform Detection

**File:** `vllm/platforms/cuda.py`

| Line | Code | Context |
|------|------|---------|
| 168 | `dist_backend: str = "nccl"` | NCCL backend for CUDA |
| 177 | `Ampere and Hopper or later NVIDIA GPUs.` | BF16 support check |
| 180 | `Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported` | BF16 unsupported on sm_70/75 |

### 4.2 Device Capability Checks

**File:** `vllm/v1/attention/ops/prefix_prefill.py`

| Line | Code | Context |
|------|------|---------|
| 19 | `IS_TURING = current_platform.get_device_capability() == (7, 5)` | Turing detection |
| 674 | `Turing does have tensor cores for float32 multiplication` | Turing tensor core note |

### 4.3 SM Version Requirements

| Requirement | SM Min | File | Line |
|------------|--------|------|------|
| FlashAttention | sm_80 | `tests/kernels/attention/test_mha_attn.py` | 87 |
| MXFP8 | sm_75 | `tests/quantization/test_compressed_tensors.py` | 660 |
| Machete W4A16 | sm_90 | `tests/quantization/test_cutlass_w4a16.py` | 19 |
| mxfp8 quantization | sm_100 | `tests/models/quantization/test_mxfp8.py` | 37 |
| FlashInfer + NVFP4 | sm_100 | `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py` | 40 |
| FlashInfer + MXFP8 | sm_100 | `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py` | 27 |
| FlashInfer + MXFP4 | sm_100 | `vllm/model_executor/kernels/linear/mxfp4/flashinfer.py` | 27 |
| torchao quantization | sm_89 | `vllm/model_executor/layers/quantization/torchao.py` | 106,127 |
| DeepGEMM | sm_90 | `vllm/utils/deep_gemm.py` | 89 |

### 4.4 Kernel SM Dispatch

**File:** `vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py`

| Line | Code | Context |
|------|------|---------|
| 215-217 | `cuda_arch = device_capability.to_int(); if cuda_arch >= 80:` | DeepGEMM MoE requires sm_80+ |

---

## 5. Buildkite CI

**File:** `.buildkite/image_build/image_build_arm64.sh`

| Line | Content |
|------|---------|
| 24 | `# build (Grace/GH200 is the arm64 GPU target; sm_90)` |

---

## 6. SM12x (Consumer Blackwell) 构建配置

### 6.1 CMakeLists.txt SM12x 构建

| 行号 | Arch Flag | 目的 |
|------|-----------|------|
| 367 | `MARLIN_ARCHS "8.0+PTX;12.0f"` | Marlin: sm_120f |
| 369 | `MARLIN_ARCHS "8.0+PTX;12.0a;12.1a"` | Marlin: sm_120a/sm_121a |
| 375 | `MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX;12.0f"` | Marlin BF16: sm_120f |
| 377 | `MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX;12.0a;12.1a"` | Marlin BF16: sm_120a/sm_121a |
| 384 | `MARLIN_FP8_ARCHS "8.9;12.0f"` | Marlin FP8: sm_120f |
| 386 | `MARLIN_FP8_ARCHS "8.9;12.0a;12.1a"` | Marlin FP8: sm_120a/sm_121a |
| 740 | `SCALED_MM_ARCHS "12.0f"` | Scaled MM (FP8): sm_120f |
| 742 | `SCALED_MM_ARCHS "12.0a;12.1a"` | Scaled MM: sm_120a/sm_121a |
| 754 | `-DENABLE_SCALED_MM_SM120=1` | 启用SM120 Scaled MM |
| 1137 | `MARLIN_MOE_ARCHS "8.0+PTX;12.0f"` | Marlin MoE: sm_120f |
| 1139 | `MARLIN_MOE_ARCHS "8.0+PTX;12.0a;12.1a"` | Marlin MoE: sm_120a/sm_121a |
| 1147 | `MARLIN_MOE_FP8_ARCHS "8.9;12.0;12.1"` | Marlin MoE FP8: sm_120/sm_121 |

### 6.2 Docker 默认构建架构

**File:** `docker/versions.json`

```json
"default": "7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX"
```

### 6.3 QUTLASS (NVFP4) SM12x 构建

**File:** `cmake/external_projects/qutlass.cmake`

| 行号 | Arch | 目的 |
|------|------|------|
| 35 | `"10.0f;12.0f"` | QUTLASS NVFP4: sm_100f/sm_120f |
| 37 | `"12.0a;12.1a;10.0a;10.3a"` | QUTLASS: sm_120a/sm_121a/sm_100a/sm_103a |
| 99 | `"[QUTLASS] Skipping build: no supported arch (12.0f / 10.0f)"` | 构建检查 |

### 6.4 cmake/utils.cmake SM12x 家族匹配

**File:** `cmake/utils.cmake`

| 行号 | 内容 |
|------|------|
| 204 | `__CUDA_ARCH_FAMILY_SPECIFIC__ definition. E.g. "121a" -> "12.1a"` |
| 388 | `SRC="12.0f" matches TGT="12.1a" since SM121 is in the SM12x family` |

### 6.5 SM12x CUTLASS Arch Guard

**File:** `csrc/cutlass_extensions/common.hpp`

```cpp
// Line 114: enable_sm120_only - 仅SM120
struct enable_sm120_only : Kernel { ... };

// Line 128-130: SM12x family includes SM120 (RTX 5090) and SM121 (DGX Spark GB10)
struct enable_sm120_family : Kernel { ... };
```

### 6.6 SM12x Scaled MM 内核文件

| 文件 | 目的 |
|------|------|
| `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu` | SM120 FP8 Scaled MM入口 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu` | SM120 FP8 GEMM实现 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh` | SM120 FP8 分发逻辑 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu` | SM120 Blockwise FP8 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh` | SM120 Blockwise分发 |

### 6.7 SM12x NVFP4 内核文件

| 文件 | 目的 |
|------|------|
| `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu` | SM120 NVFP4 GEMM |
| `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu` | NVFP4入口 (`ENABLE_NVFP4_SM120`) |
| `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu` | NVFP4量化入口 |
| `csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu` | SM120 NVFP4 MoE |
| `csrc/libtorch_stable/cache_kernels.cu` | KV Cache (`ENABLE_NVFP4_SM120`) |

### 6.8 SM12x CUTLASS MoE 内核

| 文件 | 宏 | 目的 |
|------|-----|------|
| `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu` | `ENABLE_CUTLASS_MOE_SM120` | SM120 MoE GEMM |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu` | (SM100同文件,含sm100配置) | Grouped MM配置 |

---

## 7. SM10x (Data Center Blackwell) 构建配置

### 7.1 CMakeLists.txt SM10x

| 行号 | Arch Flag | 目的 |
|------|-----------|------|
| 500 | `ES_MXFP8_GROUPED_MM_ARCHS "10.0f;11.0f"` | Expert-Sparse MXFP8: sm_100f/sm_110f |
| 502 | `ES_MXFP8_GROUPED_MM_ARCHS "10.0a;10.1a;10.3a"` | ES MXFP8: sm_100a/sm_101a/sm_103a |

### 7.2 SM100 Scaled MM 内核文件

| 文件 | 目的 |
|------|------|
| `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu` | SM100 FP8入口 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu` | SM100 FP8 GEMM |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8_dispatch.cuh` | SM100 FP8分发 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu` | SM100 Blockwise FP8 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8_dispatch.cuh` | SM100 Blockwise分发 (enable_sm100_to_sm120) |

### 7.3 SM100 MLA Decode

| 文件 | 目的 |
|------|------|
| `csrc/libtorch_stable/torch_bindings.cpp:225` | `sm100_cutlass_mla_decode` Op注册 |
| `csrc/libtorch_stable/torch_bindings.cpp:232` | `sm100_cutlass_mla_get_workspace_size` Op注册 |

---

## 8. FlashInfer b12x (SM12x 专用后端)

### 8.1 Python接口

**File:** `vllm/utils/flashinfer.py`

| 行号 | 函数 | 目的 |
|------|------|------|
| 138-139 | `flashinfer_b12x_fused_moe` | FlashInfer SM12x MoE (懒加载) |
| 274-275 | `has_flashinfer_b12x_gemm()` | 检测b12x FP4 GEMM可用性 |
| 289 | `has_flashinfer_b12x_moe()` | 检测b12x MoE可用性 |

### 8.2 NVFP4 b12x GEMM

**File:** `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`

| 行号 | 代码 | 目的 |
|------|------|------|
| 222 | `NVFP4 GEMM via FlashInfer's b12x CuTe DSL warp-level MMA kernel (SM120+)` | SM120专用NVFP4 GEMM |
| 228 | `current_platform.has_device_capability(120)` | SM120检测 |
| 232 | `"FlashInfer b12x requires SM120+"` | 错误提示 |

### 8.3 b12x MoE

**File:** `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`

| 行号 | 代码 | 目的 |
|------|------|------|
| 30 | `FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121)` | SM12x专用MoE |
| 122 | `has_flashinfer_b12x_moe()` | 检测可用性 |
| 207 | `flashinfer_b12x_fused_moe(...)` | 调用b12x MoE |

### 8.4 配置注册

**File:** `vllm/envs.py:1683` - `"flashinfer-b12x"` NVFP4 GEMM后端选项
**File:** `vllm/config/kernel.py:131` - `"flashinfer_b12x"` MoE后端选项
**File:** `vllm/model_executor/kernels/linear/__init__.py:826` - `"flashinfer-b12x"` 内核注册

---

## 9. SM 版本需求补充

| 需求 | SM Min | 文件 | 行号 |
|------|--------|------|------|
| b12x FP4 GEMM | sm_120 | `tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py` | 92 |
| b12x MoE | sm_120 | `tests/kernels/moe/test_flashinfer_b12x_moe.py` | 11 |
| NVFP4 QUTLASS | sm_100或sm_120 | `tests/kernels/quantization/test_nvfp4_qutlass.py` | 38 |
| MXFP4 QUTLASS | sm_100或sm_120 | `tests/kernels/quantization/test_mxfp4_qutlass.py` | 37 |
| SM120 NVFP4 MoE (仅BF16输出) | sm_120 | `csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu` | 704 |
| CUTLASS_ARCH_MMA_SM120_SUPPORTED | sm_120 | `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu` | 241 |
| sm100 MLA Decode | sm_100 | `vllm/_custom_ops.py` | 3062 |
| FP8 ViT Attention (GB200) | sm_100 | `docs/features/quantization/fp8_vit_attn.md` | 17 |
| Blackwell CUDA 12.8+ | sm_100 | `docs/getting_started/installation/gpu.cuda.inc.md` | 37 |
