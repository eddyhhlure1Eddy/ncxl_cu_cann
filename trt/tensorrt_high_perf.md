# TensorRT 高性能特性深度挖掘报告

> 源码路径: `C:\Users\Administrator\Desktop\nccl\TensorRT`
> 生成时间: 2026-05-29
> 版本: v10.16.1 (commit 5302b288)

---

## 1. 数据类型体系 (12种)

**File:** `include/NvInferRuntimeBase.h:145-195`

| DataType | 枚举值 | 位宽 | 说明 |
|----------|--------|------|------|
| `kFLOAT` | 0 | 32 | IEEE FP32 |
| `kHALF` | 1 | 16 | IEEE FP16 (5e11m) |
| `kINT8` | 2 | 8 | 量化INT8 |
| `kINT32` | 3 | 32 | 整数 |
| `kBOOL` | 4 | 8 | 布尔 |
| `kUINT8` | 5 | 8 | 无符号整数 |
| **`kFP8`** | 6 | 8 | **FP8 E4M3 (1s+4e+3m)** |
| **`kBF16`** | 7 | 16 | **Brain Float (8e+8m)** |
| `kINT64` | 8 | 64 | 长整数 |
| **`kINT4`** | 9 | 4 | **4位整数 (W4A4量化)** |
| **`kFP4`** | 10 | 4 | **FP4 E2M1 (1s+2e+1m, NVFP4)** |
| **`kE8M0`** | 11 | 8 | **无符号E8M0 (仅指数, MXFP4缩放因子)** |

### 数据类型位宽计算 (子字节类型)

**File:** `samples/common/common.h:558`

```cpp
case DataType::kFP4: return (vol + 1) / 2;  // 2个FP4打包为1字节
```

---

## 2. BuilderFlag — 构建优化标志 (27个)

**File:** `include/NvInfer.h:10464-10652`

| BuilderFlag | 值 | 含义 | 状态 |
|-------------|-----|------|------|
| `kFP16` | 0 | 启用FP16层选择 | Deprecated (强类型替代) |
| `kINT8` | 1 | 启用INT8层选择 | Deprecated |
| `kDEBUG` | 2 | 逐层同步调试 | 活跃 |
| `kGPU_FALLBACK` | 3 | DLA回退到GPU | 活跃 |
| `kREFIT` | 4 | 可重构引擎 | 活跃 |
| `kDISABLE_TIMING_CACHE` | 5 | 禁用计时缓存 | 活跃 |
| **`kTF32`** | 6 | **TF32内积 (10b尾数, 默认开启)** | 活跃 |
| **`kSPARSE_WEIGHTS`** | 7 | **稀疏权重优化 (2:4结构化稀疏)** | 活跃 |
| `kSAFETY_SCOPE` | 8 | 安全范围限制 | Deprecated |
| `kOBEY_PRECISION_CONSTRAINTS` | 9 | 严格精度约束 | Deprecated |
| `kPREFER_PRECISION_CONSTRAINTS` | 10 | 优先精度约束 | Deprecated |
| `kDIRECT_IO` | 11 | 禁止IO重格式 | Deprecated |
| `kREJECT_EMPTY_ALGORITHMS` | 12 | 拒绝空算法集 | Deprecated |
| `kVERSION_COMPATIBLE` | 13 | 版本前向兼容 | 活跃 |
| `kEXCLUDE_LEAN_RUNTIME` | 14 | 排除精简运行时 | 活跃 |
| **`kFP8`** | 15 | **启用FP8层选择** | Deprecated |
| `kBF16` | 17 | 启用BF16层选择 | Deprecated |
| **`kWEIGHT_STREAMING`** | 21 | **权重流式加载 (GPU内存优化)** | 活跃 |
| **`kINT4`** | 22 | **启用INT4层选择** | Deprecated |
| **`kFP4`** | 26 | **启用FP4层选择 (NVFP4)** | Deprecated |
| `kSTRICT_NANS` | - | NaN严格模式 (与SPARSE互斥) | 活跃 |
| `kDISTRIBUTIVE_INDEPENDENCE` | - | 分配独立性 | 活跃 |

---

## 3. 量化体系 — Q/DQ Layer

### 3.1 IQuantizeLayer

**File:** `include/NvInfer.h:5530-5665`

| 量化输出类型 | 缩放类型 | 量化模式 | Block Size |
|-------------|---------|---------|-----------|
| `kINT8` | FP32 scalar/1D | per-tensor / per-channel | N/A |
| `kFP8` | FP32 scalar/1D | per-tensor / per-channel | N/A |
| **`kFP4`** | **`kFP8` (UE4M3)** | **Block量化 (NVFP4)** | **16** |
| **`kFP8`** | **`kE8M0`** | **Block量化 (MXFP8)** | **32** |
| `kINT4` | FP32 scalar/1D | per-channel / block | 可配置 |

### 3.2 Block量化公式

```cpp
// Block Quantization (NVFP4 / MXFP8 / INT4)
// 2D RS输入, R (dimension 0) 为blocking轴, B为block size
// scale shape: (R//B, S)
output[r,s] = clamp(round(input[r,s] / scale[r//B, s]) + zeroPt[r//B, s])

// setBlockShape() - 允许正数和-1(完全阻塞维度)
bool setBlockShape(Dims const& blockShape) noexcept;
```

### 3.3 IDynamicQuantizeLayer — 动态量化

**File:** `include/NvInfer.h:5857-5929`

```cpp
// 动态量化输出类型
setToType(DataType::kFP4);   // NVFP4量化 (默认)
setToType(DataType::kFP8);   // MXFP8量化

// 缩放因子类型
setScaleType(DataType::kFP8);   // NVFP4缩放 (默认)
setScaleType(DataType::kE8M0);  // MXFP8缩放
setScaleType(DataType::kFLOAT); // FP32缩放

// Block量化轴: 必须是最后或倒数第二维
setAxis(int32_t axis);
```

### 3.4 IDequantizeLayer

**File:** `include/NvInfer.h:5668-5729`

```cpp
// 反量化公式
output = (input - zeroPt) * scale

// 输入类型
input: kINT8 / kFP8 / kINT4 / kFP4
// 输出类型
output: kFLOAT / kHALF / kBF16

// Block反量化支持: kFP4, kFP8, kINT4
```

---

## 4. IAttention — 融合注意力层

**File:** `include/NvInfer.h:7025-7383`

### 4.1 IAttention 输入/输出

| Index | 输入 | 类型 | 形状 |
|-------|------|------|------|
| 0 | Query | kFLOAT/kHALF/kBF16 | [B, Hq, Sq, D] |
| 1 | Key | kFLOAT/kHALF/kBF16 | [B, Hkv, Sk, D] |
| 2 | Value | kFLOAT/kHALF/kBF16 | [B, Hkv, Sk, D] |
| 3 | Mask (可选) | kBOOL或BMM1输出类型 | [B, Hq, Sq, Sk] |
| 4 | NormQuantizeScale (可选) | kFLOAT/kHALF/kBF16 | 0D或1D |
| Output | Attention输出 | 同Query类型 | [B, Hq, Sq, D] |

### 4.2 IAttention 高级特性

| 特性 | API | 说明 |
|------|-----|------|
| 因果注意力 | `setCausal(bool)` | 因果推理 |
| 可分解 | `setDecomposable(bool)` | 无融合内核时分解为多内核 |
| 归一化操作 | `setNormalizationOperation(AttentionNormalizationOp)` | 默认kSOFTMAX |
| 归一化量化 | `setNormalizationQuantizeToType(kFP8/kINT8)` | 注意力归一化输出量化 |
| 量化缩放 | `setNormalizationQuantizeScale(ITensor&)` | 归一化量化缩放 |
| 元数据 | `setMetadata(ITensor&)` | 注意力元数据 |
| 注意力类型 | `kFP8` / `kFP4` | KV Cache量化类型 |

---

## 5. IKVCacheUpdateLayer — KV缓存更新

**File:** `include/NvInfer.h:7461-7545`

```cpp
enum class KVCacheMode : int32_t {
    // 10.15仅支持kLINEAR
};

class IKVCacheUpdateLayer : public ILayer {
    // 3输入: cache, update, writeIndices
    // 1输出: 更新后的cache
    bool setCacheMode(KVCacheMode cacheMode) noexcept;
    KVCacheMode getCacheMode() const noexcept;
};
```

---

## 6. Fused Multi-Head Attention (FMHA) Cubin矩阵

### 6.1 预编译Cubin — 7代SM全覆盖

**File:** `plugin/bertQKVToContextPlugin/fused_multihead_attention_v2/`

| SM架构 | GPU | Cubin文件数 | 支持精度 |
|--------|-----|-----------|---------|
| **sm72** | V100 | 多个 | INT8 |
| **sm75** | T4 | 多个 | INT8, FP16 |
| **sm80** | A100 | 多个 | INT8, FP16 |
| **sm86** | A30 | 少量 | INT8 |
| **sm87** | AGX Orin | 多个 | INT8, FP16 |
| **sm90** | H100/H200 | 19个 | **INT8, FP16** |
| **sm100** | B200/GB200 | 19个 | **INT8, FP16** |
| **sm120** | RTX 5090 | 19个 | **INT8, FP16** |

### 6.2 SM90/SM100/SM120 Cubin配置 (19个/代)

| 序列维度 | Head维度 | 精度 | IL(Interleaved)变体 |
|---------|---------|------|-------------------|
| 128 | 64 | INT8 | ✓ (il_int8) |
| 192 | 64 | INT8 | ✓ (il_int8) |
| 256 | 64 | INT8 | ✓ (il_int8) |
| 384 | 64 | INT8 | ✓ (il_int8) |
| 512 | 64 | INT8 | ✓ (il_int8) |
| 96 | 64 | INT8 | ✓ (il_int8) |
| 64 | 64 | INT8 | ✓ (il_int8) |
| 128 | 64 | FP16 | - |
| 192 | 64 | FP16 | - |
| 256 | 64 | FP16 | - |
| 384 | 64 | FP16 | - |
| 512 | 64 | FP16 | - |
| 96 | 64 | FP16 | - |
| 64 | 64 | FP16 | - |
| 256 | 32 | INT8 | sm80/sm75 only |
| 512 | 32 | INT8 | sm80/sm75 only |

### 6.3 Cubin大小参考 (SM120)

| 内核 | Cubin大小 |
|------|----------|
| `fmha_v2_int8_384_64_sm120` | 343,904 bytes |
| `fmha_v2_int8_256_64_sm120` | 268,384 bytes |
| `fmha_v2_int8_128_64_sm120` | ~250KB |
| `fmha_v2_int8_96_64_sm120` | 249,568 bytes |
| `fmha_v2_int8_64_64_sm120` | 113,248 bytes |
| `fmha_v2_fp16_384_64_sm120` | 类似 |
| `fmha_v2_il_int8_384_64_sm120` | 类似 |

---

## 7. Plugin 体系 — 47个官方插件

| 插件 | 关键特性 | 量化支持 |
|------|---------|---------|
| **bertQKVToContext** | **FMHA v1+v2, 7代SM cubin** | INT8 + FP16 + FP8 |
| **embLayerNorm** | Embedding + LayerNorm融合 | INT8 |
| **skipLayerNorm** | Skip + LayerNorm融合 | INT8 |
| **fcPlugin** | **cuBLASLt HMMA/FMA选择** | FP16/FP32 |
| **instanceNorm** | **PTX内联asm (cvt.rn.f16.f32, ld.global.cs.nc)** | FP16 |
| **groupNorm** | Group Normalization | FP16 |
| **efficientNMS** | NMS优化 | - |
| **batchedNMS** | Batched NMS | - |
| **disentangledAttention** | 解耦注意力 (Disentangled) | FP16 |
| **multiscaleDeformableAttn** | 多尺度可变形注意力 | - |
| **regionPlugin** | YOLO softmax tree | FP32 |
| **geluPlugin** | GELU激活 | FP16 |
| **leakyReluPlugin** | LeakyReLU | - |
| **clipPlugin** | Clip操作 | - |
| **modulatedDeformConv** | 调制可变形卷积 | - |
| **pillarScatter** | PointPillars scatter | - |
| **voxelGenerator** | 体素生成 | - |
| **coordConvAC** | CoordConv AC | - |
| **cropAndResize** | 裁剪缩放 | - |
| **pyramidROIAlign** | ROI Align | - |

---

## 8. PTX 内联汇编 (InstanceNorm 插件)

**File:** `plugin/instanceNormalizationPlugin/instanceNormCommon.h:64-366`

### 8.1 FP16 转换指令

```cuda
asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(src[2*i+0]));  // FP32→FP16
asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(src[2*i+1]));
asm volatile("mov.b32 %0, {%1, %2};" : "=r"(dst[i]) : "h"(lo), "h"(hi));  // 2xFP16打包

asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(src[i]));  // 拆包
asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dst[2*i+0]) : "h"(lo));  // FP16→FP32
asm volatile("cvt.f32.f16 %0, %1;" : "=f"(dst[2*i+1]) : "h"(hi));
```

### 8.2 缓存一致性加载/存储

```cuda
// 一致性流加载 (L2 cache streaming)
asm volatile("ld.global.cs.nc.s32 %0, [%1];" : "=r"(tmp) : "l"(gmem));
asm volatile("ld.global.cs.nc.v2.s32 {%0,%1}, [%2];" : ...);  // 64-bit向量

// 一致性流存储
asm volatile("st.global.cs.s32 [%0], %1;" ::"l"(gmem), "r"(tmp));
asm volatile("st.global.cs.v2.s32 [%0], {%1,%2};" :: ...);
```

### 8.3 锁同步 (InstanceNorm)

```cuda
asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(retired_ctas) : "l"(gmem_retired_ctas));
```

---

## 9. cuBLASLt HMMA — FC Plugin

**File:** `plugin/fcPlugin/fcPlugin.cpp:170-545`

```cpp
// 数值实现标志选择
auto numericImpl = Ctype == CUDA_R_16F 
    ? CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA   // FP16使用HMMA (Tensor Core)
    : CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA;   // FP32使用FMA

// 跳过HMMA-FP32累积内核 (精度考虑)
if (Ctype == CUDA_R_32F && numericImpl == CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
    // skip HMMA-fp32accu kernels

// FP16必须使用HMMA
if (mType == DataType::kHALF && p.numericImpl != CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
    // reject non-HMMA FP16
```

---

## 10. HardwareCompatibilityLevel — 跨架构引擎

**File:** `include/NvInfer.h:10950-10998`

| 级别 | 含义 | 限制 |
|------|------|------|
| `kNONE` | 不要求兼容 | 无限制 |
| **`kAMPERE_PLUS`** | **Ampere及更新GPU兼容** | 共享内存≤48KiB, 禁用cuDNN/cuBLAS/cuBLASLt |
| `kSAME_COMPUTE_CAPABILITY` | 相同计算能力兼容 | 禁用cuDNN/cuBLAS/cuBLASLt, Turing+ |

---

## 11. TacticSource — 算法来源

**File:** `include/NvInferRuntime.h:2917-2961`

| TacticSource | 说明 |
|-------------|------|
| kCUBLAS | cuBLAS矩阵乘法 |
| kCUBLASLT | cuBLASLt矩阵乘法 |
| kCUDNN | cuDNN卷积/注意力 |
| kEDGE_CONVOLUTIONS | 边缘卷积 (DLA) |

---

## 12. TilingOptimizationLevel — 片上缓存优化

**File:** `include/NvInfer.h:11002-11039`

| 级别 | 策略 | 编译时间影响 |
|------|------|------------|
| `kNONE` | 不优化 | 无 |
| `kFAST` | 快速启发式 | 略微增加 |
| `kMODERATE` | 混合启发式/性能分析 | 中等增加 |
| `kFULL` | 最广搜索空间 | 显著增加 |

---

## 13. Weight Streaming — 权重流式加载

**File:** `include/NvInfer.h:10599` + `NvInferRuntime.h:3918-4096`

```cpp
// BuilderFlag::kWEIGHT_STREAMING
// 运行时API:
uint64_t getWeightStreamingBudget() const noexcept;       // 当前预算
uint64_t setWeightStreamingBudget(uint64_t budget);        // 设置预算
uint64_t getWeightStreamingAutomaticBudget() const noexcept; // 自动预算
uint64_t getWeightStreamingScratchMemory() const noexcept;  // 所需暂存内存
double getWeightStreamingUtilization() const noexcept;       // 利用率
```

---

## 14. Quantization Toolkit — PyTorch量化工具

**File:** `tools/pytorch-quantization/`

### 14.1 校准器

| 校准器 | 方法 | 文件 |
|--------|------|------|
| `MaxCalibrator` | 最大值 | `calib/max.py:91` |
| `HistogramCalibrator` | 直方图 | `calib/histogram.py:37` |

### 14.2 AMAX计算方法

| 方法 | 函数 | 文件:行号 |
|------|------|----------|
| 熵 | `_compute_amax_entropy` | `histogram.py:182` |
| MSE | `_compute_amax_mse` | `histogram.py:257` |
| 百分位 | `_compute_amax_percentile` | `histogram.py:296` |
| 最大值 | `compute_amax` | `max.py:91` |

### 14.3 FP8仿真

**File:** `tools/pytorch-quantization/src/tensor_quant_gpu.cu:142-154`

```cuda
__global__ void fake_e4m3fy_kernel(const T* inputs, size_t n, T* outputs) {
    // FP8 E4M3伪量化: FP32→FP8→FP32
    outputs[idx] = static_cast<T>(static_cast<float>(
        static_cast<__nv_fp8_e4m3>(static_cast<float>(inputs[idx]))));
}
```

---

## 15. Polygraphy — 测试与调试工具

**File:** `tools/Polygraphy/`

| 特性 | 说明 |
|------|------|
| `compute_capabilities` | 指定目标计算能力构建引擎 |
| TRT Runner | 封装TensorRT推理 |
| ONNX Runner | ONNX模型对比 |
| 精度对比 | 逐层精度分析 |

---

## 16. TRT Engine Explorer — 引擎分析

**File:** `tools/experimental/trt-engine-explorer/`

### 16.1 FP4渲染支持

**File:** `trex/activations.py:73-127`

```python
"Row major linear FP4 format (kLINEAR)": "FP4",
"FP4": "FP4",
"FP4E2M1": "FP4",
# FP4位宽=0.5
'FP4': ('FP4', 0.5),
```

### 16.2 计算效率分析

```python
# compute_efficiency = MACs / latency
# memory_efficiency
# arithmetic_intensity
```

---

## 17. TensorRT vs vLLM/SGLang 高性能特性对比

| 维度 | TensorRT | vLLM | SGLang |
|------|----------|------|--------|
| **FP4 (NVFP4)** | ✓ (kFP4 DataType, Q/DQ Layer, Block量化) | ✓ (FlashInfer b12x) | ✓ (CUTLASS Sm120/Sm100) |
| **FP8 (E4M3)** | ✓ (kFP8 DataType, IAttention量化) | ✓ (FP8 KV Cache) | ✓ (FP8 GEMM) |
| **INT4 (W4A4)** | ✓ (kINT4 DataType) | ✓ (AWQ/GPTQ) | ✓ (Marlin MoE) |
| **E8M0 (MX缩放)** | ✓ (kE8M0 DataType) | ✗ | ✗ |
| **Block量化** | ✓ (NVFP4 BS=16, MXFP8 BS=32) | ✗ | ✗ |
| **FMHA Cubin** | ✓ (7代SM, 19+配置/代) | ✗ (FlashInfer) | ✗ (FlashInfer/FA3) |
| **SM120 Cubin** | ✓ (19个预编译cubin) | ✗ | ✓ (JIT编译) |
| **IAttention** | ✓ (原生API, 量化归一化) | ✗ | ✗ |
| **KV Cache Update** | ✓ (IKVCacheUpdateLayer API) | ✓ (自定义) | ✓ (自定义) |
| **TF32** | ✓ (BuilderFlag, 默认开启) | ✗ (PyTorch控制) | ✗ |
| **2:4稀疏** | ✓ (kSPARSE_WEIGHTS) | ✗ | ✗ |
| **Weight Streaming** | ✓ (kWEIGHT_STREAMING) | ✗ | ✗ |
| **Tiling优化** | ✓ (4级) | ✗ | ✗ |
| **跨架构引擎** | ✓ (AMPERE_PLUS/SAME_CC) | ✗ | ✗ |
| **PTX内联asm** | ✓ (InstanceNorm, FC) | ✓ (大量) | ✓ (大量) |
| **cuBLASLt HMMA** | ✓ (FC Plugin) | ✓ | ✓ |
| **动态量化** | ✓ (IDynamicQuantizeLayer) | ✗ | ✗ |
