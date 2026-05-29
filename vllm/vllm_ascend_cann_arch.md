# vLLM-Ascend CANN 架构深度挖掘报告

> 源码路径: `C:\Users\Administrator\Desktop\nccl\vllm-ascend`
> 生成时间: 2026-05-29
> 最新提交: `2a77209a`

---

## 1. CANN 架构概述

vLLM 对华为 Ascend NPU 的支持是通过 **vllm-ascend** 插件项目实现的，作为 vLLM 的 OOT (Out-Of-Tree) 硬件插件运行。CANN (Compute Architecture for Neural Networks) 是华为的 AI 计算框架。

### 架构关系

```
vLLM (核心)
  └── vllm-ascend (OOT 硬件插件)
        ├── torch_npu (PyTorch Ascend 后端)
        ├── HCCL (Huawei Collective Communication Library)
        ├── AscendC (自定义算子 DSL)
        ├── ATB (Ascend Transformer Boost)
        ├── ACL (Ascend Computing Language)
        └── op_plugin (NPU 算子插件)
```

---

## 2. Ascend 芯片系列 (SoC Version)

**File:** `vllm_ascend/utils.py:819-851`

| AscendDeviceType | 枚举值 | SoC Version 范围 | 芯片型号 | 定位 |
|-----------------|--------|-----------------|---------|------|
| **A2** | 0 | 220-225 | Ascend 910B (910B1/910B2/910B3) | 数据中心训练/推理 |
| **A3** | 1 | 250-255 | Ascend 910C | 新一代数据中心 |
| **A5** | 3 | 260 | Ascend 910D | 最新一代 (FP8原生) |
| **_310P** | 2 | 200-205 | Ascend 310P3 | 推理专用 (边缘) |

### SoC Version 检测代码

```python
# vllm_ascend/utils.py:841-851
soc_version = torch_npu.npu.get_soc_version()
if 220 <= soc_version <= 225:
    cur_device_type = AscendDeviceType.A2        # 910B
elif 250 <= soc_version <= 255:
    cur_device_type = AscendDeviceType.A3        # 910C
elif 200 <= soc_version <= 205:
    cur_device_type = AscendDeviceType._310P     # 310P3
elif soc_version == 260:
    cur_device_type = AscendDeviceType.A5        # 910D
```

### 芯片特性差异

| 特性 | A2 (910B) | A3 (910C) | A5 (910D) | 310P |
|------|-----------|-----------|-----------|------|
| FP8 E4M3 | 否 (INT8量化) | 否 (INT8量化) | **是** | 否 |
| ACLGraph | 是 | 是 | 是 | 部分 |
| HCCL | 是 | 是 | 是 | 部分 |
| AscendC 自定义算子 | 是 | 是 | 是 | 有限 |
| 动态Batch | 910B3支持 | 是 | 是 | 否 |
| MC2 MoE通信 | 是 | 是 | 是 | 否 |
| MXFP4/W4A4 | 是 | 是 | 是 (FP8路径) | 否 |

---

## 3. HCCL (华为集合通信库)

### 3.1 HCCL 库发现

**File:** `vllm_ascend/utils.py:398-415`

```python
def find_hccl_library() -> str:
    so_file = envs_ascend.HCCL_SO_PATH  # 环境变量覆盖
    so_file = "libhccl.so"               # 默认库
```

### 3.2 HCCL 环境变量

| 变量 | 用途 | 适用芯片 |
|------|------|---------|
| `HCCL_SO_PATH` | HCCL库路径 | 所有 |
| `HCCL_OP_EXPANSION_MODE=AIV` | 使用AIV展开模式 (替代AICPU) | A3 |
| `HCCL_INTRA_PCIE_ENABLE=1` | 启用PCIe通信 | A2 |
| `HCCL_INTRA_ROCE_ENABLE=0` | 禁用RoCE | A2 |
| `HCCL_BUFFSIZE` | HCCL缓冲区大小 | 所有 |

### 3.3 HCCL Process Group Registry

**File:** `vllm_ascend/patch/worker/_hccl_pg_registry.py`

| 行号 | 函数 | 用途 |
|------|------|------|
| 65 | `make_hccl_pg_key()` | 创建HCCL PG标识键 |
| 80 | `_normalize_hccl_pg_options()` | 归一化PG选项 |
| 1073 | `create_hccl_pg_options()` | 创建PG选项 (utils.py) |
| 1096-1100 | `hccl_config_map` | dp/dynamic_eplb组配置 |

### 3.4 NPU Communicator

**File:** `vllm_ascend/distributed/device_communicators/npu_communicator.py`

替代 CUDA 的 `custom_all_reduce`，提供NPU专用通信器。

### 3.5 PyHCCL Wrapper

**File:** `vllm_ascend/distributed/device_communicators/pyhccl.py` / `pyhccl_wrapper.py`

Python层HCCL封装，类似vLLM的PyNccl。

### 3.6 HCCL 共享内存 (MC2)

**File:** `csrc/mc2/dispatch_ffn_combine*/op_kernel/utils/hccl_shmem.hpp`

MC2 (Mixture of Communications) 用于 MoE 专家并行的AllToAll通信，使用HCCL共享内存。

---

## 4. torch_npu 算子 (CANN 核心)

### 4.1 量化算子

| 算子 | 文件 | 用途 | 芯片 |
|------|------|------|------|
| `torch_npu.npu_quant_matmul` | `quantization/methods/w8a8_*.py` | 量化矩阵乘法 | A2+ |
| `torch_npu.npu_dynamic_quant` | `quantization/methods/w8a8_dynamic.py` | 动态INT8量化 | A2+ |
| `torch_npu.npu_dynamic_mx_quant` | `quantization/methods/w4a4_mxfp4.py` | 动态MX量化 (FP4/FP8) | A2+ |
| `torch_npu.npu_weight_quant_batchmatmul` | `quantization/methods/w8a16.py` | 权重量化BatchGEMM | A2+ |
| `torch_npu.npu_convert_weight_to_int4pack` | `quantization/methods/w4a8.py` | INT4权重打包 | A2+ |
| `torch_npu.npu_kronecker_quant` | `quantization/methods/w4a4_flatquant.py` | Kronecker量化 | A2+ |
| `torch_npu.npu_quantize` | `quantization/methods/w4a8.py` | 通用量化 | A2+ |
| `torch_npu.npu_grouped_matmul_swiglu_quant` | `quantization/methods/w8a8_dynamic.py` | 分组量化GEMM+SwiGLU | A2+ |

### 4.2 注意力算子

| 算子 | 文件 | 用途 |
|------|------|------|
| `torch_npu.npu_prompt_flash_attention` | `ops/rel_pos_attention.py` | Prompt FlashAttention |
| `torch_npu._npu_rotary_embedding` | `ops/rotary_embedding.py` | 旋转位置编码 |
| `torch_npu.npu_mrope` | `ops/rotary_embedding.py:544` | 多模态RoPE |
| `torch_npu.npu_rotary_mul` | `ops/rotary_embedding.py:586` | 旋转乘法 |

### 4.3 内存/格式算子

| 算子 | 文件 | 用途 |
|------|------|------|
| `torch_npu.npu_format_cast` | `utils.py:235` | 张量格式转换 (ND→NZ) |
| `torch_npu.get_npu_format` | `xlite/xlite.py:353` | 获取NPU格式 |
| `torch_npu._C._weak_ref_tensor` | `utils.py:1030` | 弱引用张量 |
| `torch_npu.npu_prefetch` | `ops/weight_prefetch.py:214` | 权重预取 |
| `torch_npu.npu_sign_bits_unpack` | `worker/kvcomp_utils.py:484` | 符号位解包 |

### 4.4 流管理

| 算子 | 文件 | 用途 |
|------|------|------|
| `torch_npu.npu.Stream` | `utils.py:441-486` | NPU流 (5种专用流) |
| `torch_npu.Format.FRACTAL_NZ` | `xlite/xlite.py:354` | 5D NZ格式检查 |

### 4.5 数据类型

| 类型 | 用途 | 文件 |
|------|------|------|
| `torch_npu.float4_e2m1fn_x2` | MXFP4 (2x FP4) | `quantization/methods/w4a4_mxfp4.py` |
| `torch_npu.float8_e8m0fnu` | MX Block Scale (E8M0) | `quantization/methods/w4a4_mxfp4.py` |

---

## 5. ACL 格式 (张量内存布局)

**File:** `vllm_ascend/utils.py:52-53`

| 格式 | 值 | 描述 |
|------|-----|------|
| `ACL_FORMAT_FRACTAL_ND` | 2 | N-Dimensional 标准布局 |
| `ACL_FORMAT_FRACTAL_NZ` | 29 | N-Z Fractal 5D布局 (Ascend专用) |

NPU张量需要转换为FRACTAL_NZ格式才能高效执行矩阵运算:
```python
torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
```

---

## 6. AscendC 自定义算子

### 6.1 AscendC 构建

| 文件 | 用途 |
|------|------|
| `csrc/cmake/scripts/util/ascendc_gen_options.py` | AscendC 编译选项生成 |
| `csrc/cmake/scripts/util/ascendc_bin_param_build.py` | 二进制参数构建 |
| `csrc/cmake/scripts/util/ascendc_impl_build.py` | AscendC 实现构建 |
| `csrc/cmake/scripts/util/ascendc_ops_config.py` | 算子配置 |
| `csrc/cmake/third_party/ascend_protobuf.cmake` | Protobuf依赖 |

### 6.2 AscendC 算子列表

| 算子 | 目录 | 用途 |
|------|------|------|
| `copy_and_expand_eagle_inputs` | `csrc/moe/copy_and_expand_eagle_inputs/` | EAGLE推测解码输入展开 |
| `get_masked_input_and_mask` | `csrc/kernels/` | 掩码输入获取 |
| `npu_scaled_masked_softmax` | `csrc/common/src/framework/` | 缩放掩码Softmax |
| `npu_masked_softmax_with_relposbias` | `csrc/common/src/framework/` | 带相对位置偏置的Softmax |
| `npu_fused_attention_score` | `csrc/common/src/framework/` | 融合注意力分数 |
| `top_k_top_p_ascendc` | `sample/sampler.py:242` | AscendC TopK/TopP采样 |

### 6.3 MC2 通信算子

| 算子 | 目录 | 用途 |
|------|------|------|
| `dispatch_ffn_combine` | `csrc/mc2/dispatch_ffn_combine/` | MoE dispatch+combine (BF16) |
| `dispatch_ffn_combine_bf16` | `csrc/mc2/dispatch_ffn_combine_bf16/` | BF16专用MC2 |
| `dispatch_ffn_combine_w4_a8` | `csrc/mc2/dispatch_ffn_combine_w4_a8/` | W4A8量化MC2 |

---

## 7. 量化方法 (CANN 专用)

| 方法 | 文件 | 精度 | 芯片要求 |
|------|------|------|---------|
| W8A8 Static | `quantization/methods/w8a8_static.py` | INT8/INT8 | A2+ |
| W8A8 Dynamic | `quantization/methods/w8a8_dynamic.py` | INT8/INT8 | A2+ |
| W8A8 MXFP8 | `quantization/methods/w8a8_mxfp8.py` | MXFP8/INT8 | A2+ |
| W8A16 | `quantization/methods/w8a16.py` | INT8/FP16 | A2+ |
| W4A8 | `quantization/methods/w4a8.py` | INT4/INT8 | A2+ |
| W4A16 | `quantization/methods/w4a16.py` | INT4/FP16 | A2+ |
| W4A4 MXFP4 | `quantization/methods/w4a4_mxfp4.py` | MXFP4/MXFP4 | A2+ |
| W4A4 FlatQuant | `quantization/methods/w4a4_flatquant.py` | INT4/INT4 | A2+ |
| W4A4 LAOS Dynamic | `quantization/methods/w4a4_laos_dynamic.py` | INT4/INT4 | A2+ |
| W4A4 MXFP4 FlatQuant | `quantization/methods/w4a4_mxfp4_flatquant.py` | MXFP4/MXFP4 | A2+ |
| KV C8 | `quantization/methods/kv_c8.py` | INT8 KV Cache | A2+ |
| CompressedTensors | `quantization/compressed_tensors_config.py` | 多种 | A2+ |
| ModelSlim | `quantization/modelslim_config.py` | 多种 | A2+ |

### FP8 支持 (A5/910D 专用)

**File:** `vllm_ascend/attention/dsa_v1.py:1492`

```python
soc_version = get_ascend_device_type()
dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8
# A5 使用 FP8 E4M3，其他芯片使用 INT8
```

---

## 8. ACLGraph (CUDA Graph 等效)

| 文件 | 用途 |
|------|------|
| `vllm_ascend/worker/model_runner_v1.py:435` | `self.use_aclgraph = self._use_aclgraph()` |
| `vllm_ascend/platform.py:364` | `sp_aclgraph_sizes` 序列并行图大小 |
| `vllm_ascend/quantization/methods/w8a8_mxfp8.py:199` | `self.use_aclgraph` |

---

## 9. NPU 流管理 (5种专用流)

**File:** `vllm_ascend/utils.py:441-486`

| 流 | 变量 | 用途 |
|----|------|------|
| `_PREFETCH_STREAM` | `torch_npu.npu.Stream()` | 权重预取 |
| `_GLOBAL_STREAM` | `torch_npu.npu.Stream()` | 全局默认流 |
| `_SHARED_EXPERTS_CALCULATION_STREAM` | `torch_npu.npu.Stream()` | 共享专家计算 |
| `_CP_CHUNKEDPREFILL_COMM_STREAM` | `torch_npu.npu.Stream()` | 上下文并行通信 |
| `_ATNN_CALCULATION_STREAM` | `torch_npu.npu.Stream()` | ATNN计算 |

---

## 10. 芯片特定配置

### 10.1 A2 (910B) 特性

| 特性 | 配置 | 文件 |
|------|------|------|
| PCIe通信优化 | `HCCL_INTRA_PCIE_ENABLE=1` | `utils.py:1121` |
| 禁用RoCE | `HCCL_INTRA_ROCE_ENABLE=0` | `utils.py:1122` |
| MC2通信选择 | tokens适配MC2容量 | `ascend_forward_context.py:239` |
| 动态Batch | 仅910B3支持 | `scheduler_dynamic_batch.py:161` |
| AscendStore | A2专用内存后端 | `ascend_store/backend/memcache_backend.py:32` |

### 10.2 A3 (910C) 特性

| 特性 | 配置 | 文件 |
|------|------|------|
| HCCL默认AICPU | 推荐设AIV模式 | `utils.py:648` |
| 逻辑NPU映射 | `card_id*2 + chip_id` | `cpu_binding.py:477` |
| MC2通信 | 支持 | `ascend_forward_context.py:283` |

### 10.3 A5 (910D) 特性

| 特性 | 配置 | 文件 |
|------|------|------|
| FP8 E4M3量化 | `torch.float8_e4m3fn` | `dsa_v1.py:1492,2005,2295,2543` |
| DSA注意力FP8 | A5走FP8路径，其他走INT8 | `dsa_v1.py` (4处) |
| CP注意力FP8 | 上下文并行FP8路径 | `dsa_cp.py:1016,1072` |

### 10.4 310P3 特性

| 特性 | 配置 | 文件 |
|------|------|------|
| 推理专用 | `SOC_VERSION_INFERENCE_SERIES` | `utils.py:49` |
| 独立ModelRunner | `model_runner_310p.py` | `_310p/` |
| Conv3dLayer310 | 独立Conv3d | `utils.py:781` |
| NPU Input Batch | 独立输入批处理 | `_310p/npu_input_batch.py` |

---

## 11. vLLM 核心中对 Ascend 的引用

| 文件 | 行号 | 内容 |
|------|------|------|
| `vllm/docs/quickstart.md` | 79-86 | Ascend NPU 快速开始指南 |
| `vllm/docs/design/custom_op.md` | 265-267 | vllm-ascend 作为官方插件 |
| `vllm/docs/design/v1_guide.md` | 98 | vllm-ascend 链接 |
| `vllm/docs/governance/committers.md` | 187 | Ascend NPU 维护者 |
| `vllm/config/compilation.py` | 1111 | OOT硬件插件兼容 (vllm-ascend) |
| `vllm/model_executor/models/openpangu_vl.py` | 9 | vllm-ascend 项目 |
| `vllm/distributed/kv_transfer/*/mooncake/store/*.py` | 4-5 | 从vllm-ascend适配 |
| `.buildkite/scripts/hardware_ci/run-npu-test.sh` | 全文 | Ascend NPU CI测试 |
| `.buildkite/hardware_tests/ascend_npu.yaml` | 全文 | NPU测试配置 |

---

## 12. 架构对比: NVIDIA CUDA vs Huawei CANN

| 维度 | NVIDIA CUDA | Huawei CANN |
|------|-------------|-------------|
| 计算框架 | CUDA | CANN |
| 计算语言 | PTX / CUDA C | AscendC |
| 集合通信 | NCCL | HCCL |
| 运行时API | CUDA Driver/RT | ACL |
| 张量格式 | Row-Major (ND) | FRACTAL_NZ (5D) |
| 图捕获 | CUDA Graph | ACLGraph |
| 量化 | FP8 (E4M3/E5M2), INT4/8 | FP8 (E4M3, A5+), MXFP4, W4A4 |
| PyTorch后端 | torch.cuda | torch_npu |
| 算子插件 | cuDNN/cuBLAS/cutlass | op_plugin/ATB |
| 子代理通信 | NCCL P2P | HCCL P2P |
| SM架构 | sm_70 ~ sm_121 | A2(910B) ~ A5(910D) |
| Tensor Core | HMMA/IMMA/wgmma | Cube单元 |
