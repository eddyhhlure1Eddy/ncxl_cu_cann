# Part 5: GPU Architecture Names in vLLM Source

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. NVIDIA Architecture Codenames (by Generation)

### 1.1 Pascal (SM 6.x)

| Codename | SM | Product | File | Line | Context |
|----------|-----|---------|------|------|---------|
| Pascal | 6.0 | — | `vllm/platforms/cuda.py` | 180 | `# Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported` |

### 1.2 Volta (SM 7.0/7.2)

| Codename | SM | Product | File | Line | Context |
|----------|-----|---------|------|------|---------|
| Volta | 7.0/7.2 | V100 | `csrc/launch_bounds_utils.h` | 28-29 | `2048 thr/SM: Volta (sm_70/72)` |
| V100 | 7.0 | Tesla V100 | `vllm/kernels/helion/utils.py` | 38-42 | V100 variants: `tesla_v100_sxm2_32gb`, `tesla_v100_sxm2_16gb`, `tesla_v100_pcie_32gb`, `tesla_v100_pcie_16gb` all map to `tesla_v100` |

### 1.3 Turing (SM 7.5)

| Codename | SM | Product | File | Line | Context |
|----------|-----|---------|------|------|---------|
| Turing | 7.5 | T4, RTX 2080 | `csrc/launch_bounds_utils.h` | 16-17 | `1024 thr/SM: Turing (sm_75)` |
| Turing | 7.5 | — | `vllm/platforms/cuda.py` | 180 | `Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported` |
| Turing | 7.5 | — | `vllm/v1/attention/ops/prefix_prefill.py` | 19 | `IS_TURING = current_platform.get_device_capability() == (7, 5)` |
| Turing | 7.5 | — | `vllm/v1/attention/ops/prefix_prefill.py` | 674 | `Turing does have tensor cores for float32 multiplication` |
| Turing | 7.5 | — | `tests/kernels/attention/test_mha_attn.py` | 87 | `Test Turing (pre-Ampere, sm_75): FlashAttention requires sm>=80` |
| Turing | 7.5 | — | `tests/quantization/test_compressed_tensors.py` | 660 | `MXFP8 requires Turing (sm_75+) or newer.` |

### 1.4 Ampere (SM 8.0/8.6/8.7)

| Codename | SM | Product | File | Line | Context |
|----------|-----|---------|------|------|---------|
| Ampere | 8.0 | A100 (GA100) | `csrc/launch_bounds_utils.h` | 28-29 | `2048 thr/SM: Ampere GA100 (sm_80)` |
| Ampere GA10x | 8.6/8.7 | A10, A16, A30, A40 | `csrc/launch_bounds_utils.h` | 20-21 | `1536 thr/SM: Ampere GA10x (sm_86/87)` |
| Ampere | 8.0 | — | `vllm/platforms/cuda.py` | 177 | `Ampere and Hopper or later NVIDIA GPUs.` |
| A100 | 8.0 | A100 | `vllm/model_executor/layers/fla/ops/utils.py` | 181 | `AMPERE = 166912  # A100` |
| A100 | 8.0 | A100-SXM4-80GB etc. | `vllm/kernels/helion/utils.py` | 32-37 | Maps `nvidia_a100_sxm4_80gb`, `nvidia_a100_sxm4_40gb`, `nvidia_a100_pcie_80gb`, `nvidia_a100_pcie_40gb`, `nvidia_a100_80gb_pcie` all to `nvidia_a100` |
| Ampere/Ada | <8.9 | — | `vllm/v1/attention/ops/triton_turboquant_decode.py` | 26 | `Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9)` |
| Ampere/Ada | <8.9 | — | `vllm/v1/attention/ops/triton_turboquant_store.py` | 167 | `1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)` |

### 1.5 Ada Lovelace (SM 8.9)

| Codename | SM | Product | File | Line | Context |
|----------|-----|---------|------|------|---------|
| Ada | 8.9 | L40, L4, RTX 4090 | `csrc/launch_bounds_utils.h` | 20-21 | `1536 thr/SM: Ada (sm_89)` |
| Ada | 8.9 | — | `vllm/v1/worker/gpu/sample/gumbel.py` | 119 | `on H100/Ada/Blackwell` |
| RTX 4090 | 8.9 | — | `vllm/model_executor/layers/fla/ops/utils.py` | 180 | `ADA = 101376  # RTX 4090` |
| RTX 4090 | 8.9 | — | `tests/kernels/helion/test_utils.py` | 18 | `("NVIDIA GeForce RTX 4090", "nvidia_geforce_rtx_4090")` |
| Ada Lovelace | 8.9 | — | `vllm/model_executor/layers/quantization/torchao.py` | 106 | `require GPU compute capability >= 8.9 (Ada Lovelace / Hopper) on NVIDIA` |
| Ada Lovelace | 8.9 | — | `vllm/model_executor/layers/quantization/torchao.py` | 127 | `requires GPU compute capability >= 8.9 (e.g., NVIDIA Ada Lovelace` |

### 1.6 Hopper (SM 9.0)

| Codename | SM | Product | File | Line | Context |
|----------|-----|---------|------|------|---------|
| Hopper | 9.0 | H100, H200 | `csrc/launch_bounds_utils.h` | 28-29 | `2048 thr/SM: Hopper (sm_90)` |
| Hopper | 9.0 | — | `vllm/platforms/cuda.py` | 177 | `Ampere and Hopper or later NVIDIA GPUs.` |
| Hopper | 9.0 | — | `vllm/v1/attention/backends/mla/prefill/selector.py` | 72 | `else: Hopper (SM90) and older` |
| Hopper | 9.0 | — | `vllm/v1/attention/backends/fa_utils.py` | 79 | `Hopper (SM90): prefer FA3` |
| Hopper | 9.0 | — | `vllm/v1/attention/ops/flashmla.py` | 59 | `FlashMLA Dense is only supported on Hopper devices.` |
| Hopper | 9.0 | — | `vllm/v1/attention/ops/flashmla.py` | 76 | `FlashMLA Sparse is only supported on Hopper and Blackwell devices.` |
| Hopper+ | 9.0+ | — | `vllm/v1/attention/ops/triton_turboquant_decode.py` | 85 | `0 = e4nv (Hopper+)` |
| H100 | 9.0 | — | `vllm/model_executor/layers/fla/ops/utils.py` | 182 | `HOPPER = 232448  # H100` |
| H100 | 9.0 | — | `vllm/model_executor/kernels/linear/base.py` | 200 | `compute_capability (e.g., 80 for A100, 90 for H100).` |
| H100 | 9.0 | PCIe, SXM5, 80GB, NVL | `vllm/kernels/helion/utils.py` | 24-27 | Maps `nvidia_h100_pcie`, `nvidia_h100_sxm5`, `nvidia_h100_80gb_hbm3`, `nvidia_h100_nvl` to `nvidia_h100` |
| H100 | 9.0 | — | `vllm/v1/attention/ops/vit_attn_wrappers.py` | 9 | `throughput in vision models by ~5% relative on H100` |
| H100+ | 9.0 | — | `vllm/v1/attention/backends/flash_attn_diffkv.py` | 215 | `FA3/4 on H100+ uses TMA` |
| H100+ | 9.0 | — | `vllm/v1/attention/backends/flash_attn.py` | 736 | `FA3/4 on H100+ uses TMA` |
| H100+ | 9.0 | — | `vllm/utils/torch_utils.py` | 124 | `CUDA TMA on H100+ requires all` |
| H100 | 9.0 | — | `vllm/config/vllm.py` | 1097, 1518 | `On H100 the CUDA kernel is faster than` |
| Hopper | 9.0 | — | `vllm/utils/flashinfer.py` | 933 | `cuDNN SDPA FP8 requires Hopper (SM 90) or newer.` |
| Hopper | 9.0 | — | `vllm/model_executor/warmup/kernel_warmup.py` | 70 | `FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs` |
| H100 | 9.0 | — | `csrc/quantization/machete/generate.py` | 536 | `Heuristic is currently tuned for H100s` |
| H100 | 9.0 | — | `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` | 770 | `Auto thresholds are calibrated on SM 9.0 (H100).` |
| H100 | 9.0 | — | `benchmarks/kernels/benchmark_machete.py` | 120 | `we construct enough weights to exceed L2 cache, which is 50mb on a H100` |
| H100/H200 | 9.0 | — | `vllm/engine/arg_utils.py` | 2304 | `values for non-H100/H200 GPUs.` |
| H100/MI300x | 9.0 | — | `vllm/engine/arg_utils.py` | 2317 | `For GPUs like H100 and MI300x, use larger default values.` |
| H200 | 9.0 | — | `vllm/model_executor/layers/fused_moe/fused_moe.py` | 1003-1005 | `if "H200" in device_name.split("_"): device_name = "NVIDIA_H200"` |
| H200 | 9.0 | NVL, 141GB HBM3e | `vllm/kernels/helion/utils.py` | 29-30 | Maps `nvidia_h200_nvl`, `nvidia_h200_141gb_hbm3e` to `nvidia_h200` |
| Hopper | 9.0 | — | `tests/quantization/test_cutlass_w4a16.py` | 6, 19 | `MacheteLinearKernel on sm_90 GPUs` / `Machete W4A16 requires Hopper (sm_90).` |
| Hopper/Blackwell | 9.0+ | — | `vllm/utils/deep_gemm.py` | 89, 98 | `only Hopper and Blackwell GPUs are supported.` / `E8M0 scale on a Hopper or Blackwell-class GPU.` |
| H100 | 9.0 | — | `tests/models/quantization/test_modelopt.py` | 6 | `these tests will only pass on H100` |
| H100+ | 9.0 | — | `tests/kernels/moe/test_moe_layer.py` | 469 | `flashinfer_nvlink needs H100+ GPUs` |
