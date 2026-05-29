# Part 8: Quantization & Precision SM Features in vLLM

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. Summary: Quantization Methods & SM Requirements

| Method | SM Min | Dtype | Kernel | Key Files |
|--------|--------|-------|--------|-----------|
| FP8 E4M3/E5M2 KV Cache | sm_80 | torch.float8_e4m3fn | Triton reshape_and_cache | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` |
| FP8 E4M3/E5M2 KV Cache | sm_80 | torch.float8_e5m2 | Triton decode attention | `vllm/v1/attention/ops/triton_decode_attention.py` |
| FP8 TurboQuant | sm_80 | fp8e4b15/fp8e4nv | Triton TQ decode/store | `vllm/v1/attention/ops/triton_turboquant_*.py` |
| FP8 DS MLA | sm_80 | fp8_ds_mla | Custom kernel | `vllm/v1/kv_cache_interface.py:355` |
| FP8 Per-Tensor Scale | sm_80 | fp8 | Custom kernel | `vllm/v1/kv_cache_interface.py:40` |
| FP8 Per-Token-Head Scale | sm_89 | fp8 | Custom kernel | `vllm/v1/kv_cache_interface.py:42` |
| Marlin FP8 MMA | sm_89 | e4m3/e5m2 | Marlin CUDA | `csrc/quantization/marlin/marlin_mma.h` |
| Marlin FP8 Kernels | sm_89 | fp8 | Marlin CUDA | `CMakeLists.txt:384-386` |
| MXFP8 | sm_100 | mxfp8 | Expert-sparse grouped MM | `CMakeLists.txt:500-502` |
| MXFP8 (test) | sm_100 | mxfp8 | FlashInfer | `tests/models/quantization/test_mxfp8.py:37` |
| NVFP4 KV Cache | sm_100 | nvfp4 | Custom + FlashInfer | `vllm/v1/kv_cache_interface.py:43` |
| MXFP4 | sm_100 | mxfp4 | FlashInfer | `vllm/model_executor/kernels/linear/mxfp4/flashinfer.py:27` |
| Marlin INT4/INT8 | sm_75 | int4/int8 | Marlin CUDA | `csrc/quantization/marlin/*` |
| Marlin BF16 | sm_80 | bf16 | Marlin CUDA | `CMakeLists.txt:375-377` |
| GPTQ (Marlin) | sm_75 | int4 | Marlin CUDA | `csrc/quantization/marlin/*` |
| GPTQ (AllSpark) | sm_80 | w8a16 | AllSpark CUDA | `CMakeLists.txt:687` |
| AWQ (Marlin) | sm_75 | int4 | Marlin CUDA | `csrc/quantization/marlin/*` |
| Machete W4A16 | sm_90 | w4a16 | Machete CUDA | `CMakeLists.txt:530`, `tests/quantization/test_cutlass_w4a16.py` |
| Scaled MM (FP8 matmul) | sm_90 | fp8 | Scaled MM CUDA | `CMakeLists.txt:708` |
| Scaled MM (Blackwell) | sm_120 | fp8 | Scaled MM CUDA | `CMakeLists.txt:740` |
| GGUF | sm_61 | various | GGUF CUDA | `csrc/quantization/gguf/*` |
| torchao quantization | sm_89 | various | torchao | `vllm/model_executor/layers/quantization/torchao.py` |
| DeepGEMM FP8 | sm_90 | fp8 | DeepGEMM | `vllm/utils/deep_gemm.py:89` |
| INT8 Per-Token-Head | sm_80 | int8 | Custom kernel | `vllm/v1/kv_cache_interface.py:41` |
| Petit NVFP4 | sm_100 | nvfp4 | Custom | `vllm/v1/metrics/perf.py:68` |
| ModelOpt MXFP8 | sm_100 | mxfp8 | ModelOpt | `vllm/v1/metrics/perf.py:59` |

---

## 2. FP8 (E4M3 / E5M2) Details

### 2.1 FP8 in KV Cache

**File:** `vllm/v1/kv_cache_interface.py`

| Line | Code | Purpose |
|------|------|---------|
| 40 | `FP8_PER_TENSOR = 1` | Per-tensor FP8 scale |
| 42 | `FP8_PER_TOKEN_HEAD = 3` | Per-token-head FP8 scale |
| 67 | `if kv_cache_dtype.startswith("fp8")` | FP8 KV cache detection |

**File:** `vllm/v1/worker/gpu_model_runner.py`

| Line | Code | Purpose |
|------|------|---------|
| 928 | `self.init_fp8_kv_scales()` | Initialize FP8 KV scales |
| 931 | `def init_fp8_kv_scales(self)` | FP8 scale initialization |

### 2.2 FP8 TurboQuant (SM-Dependent Format)

**File:** `vllm/v1/attention/ops/triton_turboquant_decode.py`

| Line | Code | SM Context |
|------|------|-----------|
| 26 | `Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9)` | FP8 format: e4b15 on sm_80-sm_89 |
| 85 | `0 = e4nv (Hopper+)` | FP8 format: e4nv on sm_90+ |

**File:** `vllm/v1/attention/ops/triton_turboquant_store.py`

| Line | Code | SM Context |
|------|------|-----------|
| 167 | `1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)` | Same format dispatch |

### 2.3 FP8 in Marlin MMA

**File:** `csrc/quantization/marlin/marlin_mma.h`

| Line | Instruction | Dtype A | Dtype B | SM Min |
|------|------------|---------|---------|--------|
| 86 | `mma.sync.m16n8k32.row.col.f32.e4m3.bf16.f32` | E4M3 | BF16 | sm_89 |
| 96 | `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32` | E4M3 | E4M3 | sm_89 |
| 105 | `mma.sync.m16n8k32.row.col.f32.e5m2.bf16.f32` | E5M2 | BF16 | sm_89 |
| 110 | `mma.sync.m16n8k32.row.col.f32.e5m2.e5m2.f32` | E5M2 | E5M2 | sm_89 |
| 115 | `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32` | E4M3 | E4M3 | sm_89 |
| 120 | `mma.sync.m16n8k32.row.col.f32.e5m2.e5m2.f32` | E5M2 | E5M2 | sm_89 |
| 126 | `mma.sync.m16n8k32.row.col.f32.e4m3.bf16.f32` | E4M3 | BF16 | sm_89 |

### 2.4 FP8 in Activation Quantization

**File:** `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh`

| Line | Guard | Purpose |
|------|-------|---------|
| 27 | `__CUDA_ARCH__ < 800` | BF16→FP8 conversion requires sm_80 |
| 194 | Same | Another conversion function |
| 485 | Same | Yet another conversion |

**File:** `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`

| Line | Code | SM Context |
|------|------|-----------|
| 296 | `nvcc keeps these as 2x LDG.E.128 on sm_100` | Blackwell load behavior note |

### 2.5 FP8 Scaled MM

**File:** `CMakeLists.txt`

| Line | Arch | Purpose |
|------|------|---------|
| 708 | `SCALED_MM_ARCHS "9.0a"` | FP8 scaled matmul on Hopper |
| 740 | `SCALED_MM_ARCHS "12.0f"` | FP8 scaled matmul on Blackwell |

---

## 3. NVFP4 / MXFP4 / MXFP8 (Blackwell sm_100+)

### 3.1 NVFP4 KV Cache

**File:** `vllm/v1/kv_cache_interface.py`

| Line | Code | Purpose |
|------|------|---------|
| 43 | `NVFP4 = 4` | Packed fp4 data + fp8 block scales |
| 54 | `def is_nvfp4(self)` | NVFP4 check |
| 65 | `if kv_cache_dtype == "nvfp4"` | NVFP4 detection |
| 168-170 | NVFP4 packed layout | fp4 data + fp8 block scales per head |
| 280-286 | NVFP4 dimension calculation | Full dim calculation for fp4+fp8 |
| 446-449 | NVFP4 last dim | Dimension for block table |

**File:** `vllm/utils/torch_utils.py`

| Line | Code | Purpose |
|------|------|---------|
| — | `nvfp4_kv_cache_full_dim` | Calculate full dimension for NVFP4 KV cache |

### 3.2 MXFP8

**File:** `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`

| Line | Code | Purpose |
|------|------|---------|
| 27 | `return False, "requires >=sm_100 (Blackwell)"` | MXFP8 requires Blackwell |

**File:** `tests/models/quantization/test_mxfp8.py`

| Line | Code | Purpose |
|------|------|---------|
| 37 | `reason="mxfp8 is not supported on this GPU type (requires sm_100+)."` | Test skip reason |
| 86 | Same | Another test |

**File:** `tests/quantization/test_compressed_tensors.py`

| Line | Code | Purpose |
|------|------|---------|
| 660 | `reason="MXFP8 requires Turing (sm_75+) or newer."` | MXFP8 min SM is sm_75 (for compressed tensors) |

### 3.3 MXFP4

**File:** `vllm/model_executor/kernels/linear/mxfp4/flashinfer.py`

| Line | Code | Purpose |
|------|------|---------|
| 27 | `return False, "FlashInfer + >=sm_100 (Blackwell) required"` | MXFP4 requires Blackwell |

### 3.4 NVFP4 + FlashInfer

**File:** `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`

| Line | Code | Purpose |
|------|------|---------|
| 40 | `return False, "FlashInfer + >=sm_100 required"` | NVFP4 FlashInfer requires Blackwell |

### 3.5 Expert-Sparse MXFP8 Grouped MM

**File:** `CMakeLists.txt`

| Line | Arch | Purpose |
|------|------|---------|
| 500 | `"10.0f;11.0f"` | ES MXFP8 on sm_100f/sm_110f |
| 502 | `"10.0a;10.1a;10.3a"` | ES MXFP8 alternative variants |

---

## 4. BF16 (Ampere sm_80+)

### 4.1 BF16 Support Guards

| File | Line | Guard | Context |
|------|------|-------|---------|
| `csrc/type_convert.cuh` | 72 | `__CUDA_ARCH__ >= 800` | BF16 type conversion |
| `csrc/quantization/marlin/marlin.cuh` | 60 | `__CUDA_ARCH__ < 800` | Skip BF16 on pre-Ampere |
| `csrc/quantization/marlin/marlin_dtypes.cuh` | 73 | `__CUDA_ARCH__ >= 800` | BF16 dtype availability |
| `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` | 27,194,485 | `__CUDA_ARCH__ < 800` | BF16→FP8 conversion |
| `csrc/moe/moe_wna16_utils.h` | 49,146 | `__CUDA_ARCH__ >= 800` | BF16 dequant/store |
| `vllm/platforms/cuda.py` | 177 | Ampere+ | BF16 supported |
| `vllm/platforms/cuda.py` | 180 | Pascal/Volta/Turing | BF16 NOT supported |

### 4.2 Marlin BF16 Kernels

**File:** `CMakeLists.txt`

| Line | Arch | Purpose |
|------|------|---------|
| 375 | `"8.0+PTX;9.0+PTX;12.0f"` | Marlin BF16: sm_80/sm_90/sm_120f |
| 377 | `"8.0+PTX;9.0+PTX;12.0a;12.1a"` | Marlin BF16 alternative |

---

## 5. INT4 / INT8 Quantization

### 5.1 Marlin INT4/INT8

**File:** `csrc/quantization/marlin/marlin_mma.h`

| Line | Instruction | Purpose |
|------|------------|---------|
| 49 | `mma.sync.m16n8k32.row.col.satfinite.s32.s8.s8.s32` | INT8 x INT8 MMA |
| 54 | `mma.sync.m16n8k32.row.col.satfinite.s32.s8.u8.s32` | INT8 x UINT8 MMA |

### 5.2 GGUF Quantization (DP4A)

**File:** `csrc/quantization/gguf/vecdotq.cuh`

| Line | Guard | Purpose |
|------|-------|---------|
| 48-470 | `__CUDA_ARCH__ >= 610` | DP4A dot product instruction (22 variants) |

DP4A (Dot Product Accumulate) is used for INT4/INT8 dot products in GGUF quantized formats:
- Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ variants

### 5.3 INT8 Per-Token-Head KV Cache

**File:** `vllm/v1/kv_cache_interface.py`

| Line | Code | Purpose |
|------|------|---------|
| 41 | `INT8_PER_TOKEN_HEAD = 2` | Per-token-head dynamic scales for INT8 |
| 61 | `if kv_cache_dtype == "int8_per_token_head"` | INT8 KV cache detection |

### 5.4 AllSpark GPTQ W8A16

**File:** `CMakeLists.txt`

| Line | Arch | Purpose |
|------|------|---------|
| 687 | `"8.0;8.6;8.7;8.9"` | AllSpark W8A16 kernels |

---

## 6. Machete Quantization (Hopper sm_90a)

**File:** `CMakeLists.txt`

| Line | Arch | Purpose |
|------|------|---------|
| 530 | `MACHETE_ARCHS "9.0a"` | Machete W4A16 requires sm_90a |

**File:** `tests/quantization/test_cutlass_w4a16.py`

| Line | Code | Purpose |
|------|------|---------|
| 6 | `MacheteLinearKernel on sm_90 GPUs` | Machete requires Hopper |
| 19 | `Machete W4A16 requires Hopper (sm_90).` | Test skip reason |

---

## 7. Quantization Method Bit Efficiency

**File:** `vllm/v1/metrics/perf.py`

| Method | Bits/Weight | Line |
|--------|-------------|------|
| fp8 / fbgemm_fp8 / ptpc_fp8 | 1.0 | 54-56 |
| modelopt_mxfp8 | 1.0 | 59 |
| mxfp4 | 0.5 | 61 |
| awq / awq_marlin | 0.5 | 62-63 |
| gptq / gptq_marlin | 0.5 | 64-65 |
| petit_nvfp4 | 0.5 | 68 |
| experts_int8 | 1.0 | 75 |

---

## 8. DeepSeek V4 FP8 MLA

**File:** `vllm/v1/kv_cache_interface.py`

| Line | Code | Purpose |
|------|------|---------|
| 355 | `if self.cache_dtype_str == "fp8_ds_mla"` | DeepSeek V4 FP8 MLA format |
| 357 | `448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token` | DS V4 MLA layout |
| 517 | Same | Another DS V4 MLA reference |
