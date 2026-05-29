# Comfy-Kitchen NVFP4 深度挖掘报告

> comfy-kitchen v0.2.9 — NVIDIA官方ComfyUI量化内核库, 完整NVFP4 (E2M1) Block量化实现

## 1. NVFP4 体系总览

comfy-kitchen实现了NVFP4的**全栈量化管线**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Python层 (nvfp4.py / scaled_mm_v2.py / __init__.py)              │
│  TensorCoreNVFP4Layout → QuantizedTensor → torch.compile调度       │
├─────────────────────────────────────────────────────────────────────┤
│  三后端并行                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ CUDA Backend  │  │ Triton Backend│  │ Eager Backend (参考)     │ │
│  │ .cu 内核      │  │ PTX内联汇编   │  │ PyTorch原生运算          │ │
│  │ cuBLASLt GEMM │  │ tl.inline_asm │  │ _f32_to_floatx_unpacked │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  CUDA C++ 内核层                                                    │
│  quantize_nvfp4.cu    → 量化 (FP16/BF16→FP4+FP8 block scale)      │
│  dequantize_nvfp4.cu  → 反量化 (FP4+block scale→FP16/BF16/FP32)   │
│  cublas_gemm_nvfp4.cu → cuBLASLt FP4 Block GEMM                   │
│  float_utils.cuh      → FP4/FP8工具函数 + Swizzled布局             │
│  cublaslt_runtime.h   → cuBLASLt动态加载(支持CUDA 13+)             │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. NVFP4 E2M1 数据格式详解

### 2.1 格式定义

| 属性 | E2M1 (FP4) | E4M3 (FP8 Block Scale) |
|------|-----------|----------------------|
| 总位数 | 4 bit | 8 bit |
| 符号位 | 1 bit | 1 bit |
| 指数位 | 2 bit | 4 bit |
| 尾数位 | 1 bit | 3 bit |
| 指数偏置 | 1 | 7 |
| 最大值 | **6.0** | **448.0** |
| 最小正规数 | 0.5 | 2^(-6) = 0.015625 |
| 特殊值 | 无NaN/Inf | 无Inf |

### 2.2 E2M1 编码表 (FP4全部16个值)

```
符号 | 指数 | 尾数 | 值          符号 | 指数 | 尾数 | 值
  0  |  00  |  0   | +0          1  |  00  |  0   | -0
  0  |  00  |  1   | +0.5        1  |  00  |  1   | -0.5
  0  |  01  |  0   | +1.0        1  |  01  |  0   | -1.0
  0  |  01  |  1   | +1.5        1  |  01  |  1   | -1.5
  0  |  10  |  0   | +2.0        1  |  10  |  0   | -2.0
  0  |  10  |  1   | +3.0        1  |  10  |  1   | -3.0
  0  |  11  |  0   | +4.0        1  |  11  |  0   | -4.0
  0  |  11  |  1   | +6.0        1  |  11  |  1   | -6.0  ← MAX
```

**含金量**: 仅4bit就覆盖了0.5~6.0的6个正数量化级(含0), 精度只有1位尾数 — 这意味着NVFP4**必须配合block scale才能实用**，单独使用精度极差。

### 2.3 CUDA头文件定义 (`cuda_fp4.h`, CUDA >= 12.8)

```cpp
#include <cuda_fp4.h>   // CUDA 12.8+

__nv_fp4x2_e2m1         // 打包FP4x2类型 (1 byte = 2个FP4)
__nv_fp4x2_storage_t    // 底层存储 (uint8_t)
__nv_cvt_float2_to_fp4x2(float2, __NV_E2M1, cudaRoundNearest)  // 量化
__nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t, __NV_E2M1)   // 反量化
```

## 3. NVFP4 Block量化算法 (核心含金量)

### 3.1 两级缩放体系

NVFP4采用**Per-Tensor Scale + Per-Block Scale**二级缩放:

```
原始值 = FP4_value × block_scale_FP8_E4M3 × per_tensor_scale_FP32
```

其中:
- `per_tensor_scale = amax(input) / (FP4_MAX × FP8_E4M3_MAX) = amax / (6.0 × 448.0)`
- `block_scale = amax(block) / FP4_MAX / per_tensor_scale` (存储为FP8 E4M3)
- 编码时: `encode_scale = 1.0 / (block_scale_fp8 × per_tensor_scale)`

### 3.2 量化内核算法 (`quantize_nvfp4.cu:37-59`)

```
Algorithm:
  1. Load kValsPerThread(=4) values per thread → local absmax
  2. __shfl_down_sync reduction: kThreadsPerGroup(=4) threads → block absmax
  3. decode_scale = absmax / FP4_MAX(6.0) / global_decode_scale
  4. Clamp to FP8 range → store as __nv_fp8_e4m3 in swizzled layout
  5. encode_scale = 1.0 / (decode_scale_fp8 × global_decode_scale)
  6. Multiply values by encode_scale, convert to FP4 E2M1
```

**关键参数**:
- `kBlockSize = 16` (NVFP4固定, MXFP8为32)
- `kValsPerThread = 4` (每个线程处理4个值)
- `kThreadsPerGroup = 4` (4个线程=1个block=16个值)
- 2次shuffle迭代: `log2(4) = 2`

### 3.3 反量化内核算法 (`quantize_nvfp4.cu:212-314`)

```
Algorithm:
  1. 计算block_linear_idx → 2D坐标 (row_idx, col_idx)
  2. 从swizzled布局读取 block_scale_fp8
  3. decode_scale = block_scale_fp8 × global_decode_scale
  4. __nv_cvt_fp4x2_to_halfraw2 解包FP4 → half2
  5. 乘以decode_scale → 输出FP16/BF16/FP32
```

**HiFirst语义** (`float_utils.cuh:84-91`):
```cpp
// hi_first=true (默认, 匹配cuBLAS):
//   高nibble = val0(偶数索引), 低nibble = val1(奇数索引)
// hi_first=false:
//   低nibble = val0, 高nibble = val1
store_fp4x2<OType, true>(output, idx, val0, val1)
    → float2{val1, val0}  // 交换以匹配__nv_cvt_float2_to_fp4x2的$1=高nibble语义
```

## 4. Swizzled Block Scale布局 (最高含金量)

### 4.1 为什么要Swizzle?

cuBLASLt的FP4 GEMM要求block scale按特定内存布局存储, 以优化Tensor Core的scale访存模式。Swizzle将线性地址映射为分形块, 减少bank冲突并提高向量化加载效率。

### 4.2 Swizzled布局算法 (`float_utils.cuh:110-125`)

```cpp
__device__ __forceinline__ size_t
scale_factor_swizzled_offset(size_t row_idx, size_t col_idx, uint32_t col_length) {
  constexpr uint32_t kTotalRowsPerBaseBlock = 128;  // 128行一组
  constexpr uint32_t kRowsPerBaseBlockCol = 32;     // 32行一子块
  constexpr uint32_t kColsPerBaseBlockCol = 4;      // 4列一组

  const size_t rb = row_idx / 128;           // 128行块索引
  const size_t rem = row_idx % 128;          // 块内行偏移
  const size_t d4 = rem / 32;               // 32行子块索引 (0~3)
  const size_t d3 = rem % 32;               // 子块内行偏移 (0~31)
  const size_t cbg = col_idx / 4;           // 4列组索引
  const size_t d5 = col_idx % 4;            // 组内列偏移 (0~3)

  const size_t cbg_cnt = (col_length + 3) / 4;  // 总列组数
  return ((rb * cbg_cnt + cbg) * 32 + d3) * 16 + d4 * 4 + d5;
}
```

**布局维度分解**:
```
原始: (row_idx, col_idx) — row_idx ∈ [0, M/16), col_idx ∈ [0, K/16)

Swizzled: 5D → 1D
  rb  = row_idx / 128           → 128行基块
  cbg = col_idx / 4             → 4列组
  d3  = (row_idx % 128) % 32    → 32行内偏移
  d4  = (row_idx % 128) / 32    → 4个32行子块
  d5  = col_idx % 4             → 4列内偏移

最终: ((rb*cbg_cnt + cbg)*32 + d3)*16 + d4*4 + d5
```

**Scale张量分配**:
```python
# __init__.py:250-252
scale_rows = roundup(num_rows, 128)        # 向上取整到128
scale_cols = roundup(num_cols // 16, 4)    # 每16元素1个scale, 向上取整到4
sx_uint8 = torch.zeros((scale_rows, scale_cols), dtype=torch.uint8)
```

### 4.3 Python参考实现 (`float_utils.py:292-326`)

```python
def to_blocked(input_matrix, flatten: bool = True) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix  # pad to (n_row_blocks*128, n_col_blocks*4)
    # Step 1: (n_rb, 128, n_cb, 4) → permute(0,2,1,3)
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    # Step 2: reshape to (-1, 4, 32, 4) → transpose(1,2) → (-1, 32, 16)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    # 最终shape: (padded_rows, padded_cols) 其中 padded_cols = n_col_blocks * 4
```

**逆变换** (`float_utils.py:329-352`):
```python
def from_blocked(blocked_matrix, num_rows, num_cols):
    # 逆向操作: reshape(-1,32,16) → (-1,32,4,4) → transpose → reshape → permute
```

## 5. cuBLASLt FP4 Block GEMM (最高含金量)

### 5.1 GEMM配置 (`cublas_gemm_nvfp4.cu:64-288`)

**要求**: CUDA >= 12.9

```cpp
// 核心配置
cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// ★ FP4 Block Scale模式 (CUDA 12.9新增)
cublasLtMatmulDescSetAttribute(operationDesc,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
    &CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,  // ★ 每16元素1个UE4M3 scale
    sizeof(A_scale_mode));

// ★ 数据类型: CUDA_R_4F_E2M1 (FP4 E2M1)
cudaDataType_t Atype = CUDA_R_4F_E2M1;
cudaDataType_t Btype = CUDA_R_4F_E2M1;

// ★ Scale指针 (A和B各有独立的block scale)
cublasLtMatmulDescSetAttribute(operationDesc,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_decode_scale_ptr, ...);
cublasLtMatmulDescSetAttribute(operationDesc,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_decode_scale_ptr, ...);

// alpha/beta: 设备指针, FP32
cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
```

### 5.2 维度映射 (Row-Major → Column-Major)

```python
# __init__.py:391-396  Python端
m, k_a = a.shape    # a: (M, K_packed), K_packed = K/2 (2个FP4=1个uint8)
n, k_b = b.shape    # b: (N, K_packed)
k = 2 * k_a         # 实际K维度 (FP4元素数)

# cublas_gemm_nvfp4.cu:106-110  C++端
const int m = transa ? A_rows : A_cols;   // TN: m = A_rows = N (weight行)
const int k = (transa ? A_cols : A_rows) * 2;  // K翻倍 (2 FP4/byte)
const int n = transb ? B_cols : B_rows;   // N = B_rows = M (activation行)
```

### 5.3 尾部融合 (Epilogue)

```cpp
// BIAS融合
if (bias_ptr != nullptr) {
    epilogue = CUBLASLT_EPILOGUE_BIAS;  // D = alpha*AB + beta*C + bias
    cublasLtMatmulDescSetAttribute(operationDesc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, ...);
}
```

**Bias约束**: `bias_size == m` (输出特征维度N), 只支持FP16/BF16。

### 5.4 工作空间

```python
# Hopper+: 33 MiB, 其他: 4 MiB
workspace_size = 32 * 1024 * 1024  # 32MB (C++默认)
```

### 5.5 约束条件

```python
# __init__.py:1009-1037  scaled_mm_nvfp4约束
"min_compute_capability": (10, 0),       # ★ SM100+ (Blackwell)
"a": { "dtypes": {torch.uint8}, "DivisibleBy(dim=1, factor=16)" },  # K对齐16
"b": { "dtypes": {torch.uint8}, "DivisibleBy(dim=1, factor=16)" },  # K对齐16
"block_scale_a": { "dtypes": {torch.float8_e4m3fn} },               # ★ E4M3
"block_scale_b": { "dtypes": {torch.float8_e4m3fn} },               # ★ E4M3
"out_dtype": { "dtypes": {torch.float16, torch.bfloat16} },         # ★ 无FP32输出
```

**Scale shape验证**:
```python
roundup_m = roundup(m, 128)
roundup_n = roundup(n, 128)
roundup_sk = roundup(k // 16, 4)  # K/16 = block数, 对齐到4

assert block_scale_a.shape == (roundup_m, roundup_sk)
assert block_scale_b.shape == (roundup_n, roundup_sk)
```

## 6. Triton后端: PTX内联汇编 (含金量极高)

### 6.1 量化: `cvt.rn.satfinite.e2m1x2.f32` (`quantization.py:299-320`)

```python
packed_bytes_u16 = tl.inline_asm_elementwise(
    asm="""
    {
        .reg .b8 fp4_byte;
        .reg .b16 result;
        cvt.rn.satfinite.e2m1x2.f32 fp4_byte, $1, $2;  // ★ SM100 Blackwell指令
        mov.b16 result, {fp4_byte, 0};
        mov.u16 $0, result;
    }
    """,
    constraints="=h,f,f",        // 输出: u16, 输入: f32, f32
    args=[asm_arg_hi, asm_arg_lo],
    dtype=tl.uint16,
    is_pure=True,
    pack=1,
)
```

**指令语义**:
- `cvt.rn.satfinite.e2m1x2.f32 dst, src1, src2`
  - `src1` → 高nibble (bit 7:4)
  - `src2` → 低nibble (bit 3:0)
  - `.satfinite`: 饱和到FP4最大值6.0 (不产生Inf)
  - `.rn`: 最近舍入

### 6.2 反量化: `cvt.rn.f16x2.e2m1x2` (`quantization.py:464-480`)

```python
x_f16x2_packed = tl.inline_asm_elementwise(
    asm="""
    {
        .reg .b8 byte0, byte1, byte2, byte3;
        mov.b32 {byte0, byte1, byte2, byte3}, $4;  // 拆包4个字节
        cvt.rn.f16x2.e2m1x2 $0, byte0;  // ★ 每字节→2个FP16
        cvt.rn.f16x2.e2m1x2 $1, byte1;
        cvt.rn.f16x2.e2m1x2 $2, byte2;
        cvt.rn.f16x2.e2m1x2 $3, byte3;
    }
    """,
    constraints="=r,=r,=r,=r,r",   // 4个u32输出, 1个u32输入
    args=[packed_data],
    dtype=tl.uint32,
    is_pure=True,
    pack=4,                          // ★ 4路并行
)
```

**吞吐量**: 1条`cvt.rn.f16x2.e2m1x2`指令 = 1字节→2个FP16, pack=4 = 4字节→8个FP16

### 6.3 SM100 PTX指令总结

| PTX指令 | 方向 | 语义 |
|---------|------|------|
| `cvt.rn.satfinite.e2m1x2.f32` | 量化 | 2×FP32 → 1字节(2×FP4), 饱和到6.0 |
| `cvt.rn.f16x2.e2m1x2` | 反量化 | 1字节(2×FP4) → 2×FP16 |
| `__nv_cvt_float2_to_fp4x2` | 量化(CUDA C++) | float2 → `__nv_fp4x2_storage_t` |
| `__nv_cvt_fp4x2_to_halfraw2` | 反量化(CUDA C++) | `__nv_fp4x2_storage_t` → `__half2_raw` |

## 7. Python Eager后端: 位操作参考实现 (`float_utils.py`)

### 7.1 FP32→FP4 量化 (`_f32_to_floatx_unpacked`, 派生自PyTorch AO)

纯Python实现的sub-byte浮点转换, 支持**任意ebits/mbits组合**:

```python
def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    # 三路分支: saturate / denormal / normal
    # 1. saturate: x >= max_normal → clamp
    # 2. denormal: x < min_normal → FP32加法+减法技巧
    # 3. normal: 调整指数+舍入
    # 最终: 拼接符号位
```

**FP4 E2M1特化** (mbits=1的快速路径):
```python
# float_utils.py:209-210  FP4 E2M1优化
if mbits == 1:
    result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32
    # 比通用denormal循环快2x
```

### 7.2 FP4→FP32 反量化 (`_floatx_unpacked_to_f32`)

```python
def _floatx_unpacked_to_f32(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    # 1. 提取符号位
    # 2. 计算normal路径: exp_biased_f32 | mantissa_f32
    # 3. 处理denormal (mbits=1快速路径)
    # 4. 添加符号
```

### 7.3 打包/解包工具

```python
# 2个uint4 → 1个uint8 (nibble打包)
pack_uint4(uint8_data, hi_first=True)   # 0xAB: hi=偶数, lo=奇数
unpack_uint4(uint8_data, hi_first=True) # 逆操作

# 交换nibble顺序 (hi_first ↔ !hi_first)
swap_nibbles(packed)  # 0xAB → 0xBA

# E8M0格式 (MXFP8 block scale)
f32_to_e8m0(x)  # FP32 → power-of-2 (8位指数)
e8m0_to_f32(x)  # 8位指数 → FP32 (2^(exp-127))
```

## 8. TensorCoreNVFP4Layout 量化张量系统 (`nvfp4.py`)

### 8.1 Layout设计

```python
class TensorCoreNVFP4Layout(QuantizedLayout):
    MIN_SM_VERSION = (10, 0)  # ★ Blackwell required

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        scale: torch.Tensor        # per_tensor_scale (FP32, scalar)
        block_scale: torch.Tensor  # per_block_scale (FP8 E4M3, swizzled)
        transposed: bool = False   # 逻辑转置标记 (无需物理转置)
```

### 8.2 16×16自动Padding

```python
@classmethod
def get_padded_shape(cls, orig_shape):
    rows, cols = orig_shape
    return (roundup(rows, 16), roundup(cols, 16))  # ★ 16对齐

@classmethod
def get_storage_shape(cls, orig_shape):
    padded = cls.get_padded_shape(orig_shape)
    return (padded[0], padded[1] // 2)  # 2个FP4 = 1个uint8
```

### 8.3 逻辑转置 (零拷贝)

```python
@register_layout_op(torch.ops.aten.t.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_transpose(qt, args, kwargs):
    # ★ 不执行物理转置, 只翻转transposed标记
    new_params = TensorCoreNVFP4Layout.Params(
        ...,
        transposed=not input_tensor._params.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreNVFP4Layout", new_params)
```

### 8.4 量化Linear层调度

```python
@register_layout_op(torch.ops.aten.linear.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_linear(qt, args, kwargs):
    # Fast path: 两个NVFP4 QuantizedTensor → scaled_mm_nvfp4
    result = ck.scaled_mm_nvfp4(
        input_qdata, weight_qdata,
        tensor_scale_a=scale_a, tensor_scale_b=scale_b,
        block_scale_a=block_scale_a, block_scale_b=block_scale_b,
        bias=bias, out_dtype=out_dtype,
    )
    # 输出裁剪回原始shape
    return _slice_to_original_shape(result, orig_m, orig_n)
```

## 9. NVFP4Linear 推理示例 (`samples/nvfp4_linear.py`)

```python
class NVFP4Linear(nn.Module):
    def _load_from_state_dict(self, state_dict, ...):
        weight = state_dict.pop(prefix + "weight")              # uint8 (N, K/2)
        weight_scale = state_dict.pop(prefix + "weight_scale_2") # FP32 scalar
        weight_block_scale = state_dict.pop(prefix + "weight_scale")  # uint8→FP8 E4M3
        # 封装为QuantizedTensor
        self.weight = nn.Parameter(
            QuantizedTensor(weight, "TensorCoreNVFP4Layout", params), requires_grad=False)

    def forward(self, x):
        if not isinstance(x, QuantizedTensor):
            x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout", scale=self.input_scale)
        return torch.nn.functional.linear(x, self.weight, self.bias)
```

**State Dict键映射**:
| 后缀 | 张量 | 类型 | Shape |
|------|------|------|-------|
| `weight` | 量化权重 | uint8 | (N, K/2) |
| `weight_scale` | Block scale | float8_e4m3fn | (roundup(N,128), roundup(K/16,4)) |
| `weight_scale_2` | Per-tensor scale | float32 | (1,) |
| `input_scale` | 输入量化scale | float32 | (1,) |

## 10. NVFP4 vs MXFP4 vs SVDQuant 对比

| 特性 | NVFP4 | MXFP4 | SVDQuant W4A4 |
|------|-------|-------|---------------|
| 数据格式 | E2M1 (FP4) | E2M1 (FP4) | INT4 |
| Block大小 | **16** | **32** | **64** |
| Block Scale格式 | **FP8 E4M3** (448.0) | **E8M0** (2^n) | BF16/FP16 |
| Per-Tensor Scale | **FP32** (必须) | 无 | 无 |
| GEMM后端 | cuBLASLt (CUDA 12.9+) | cuBLASLt (未支持) | 自定义MMA内核 |
| SM要求 | **SM100+** | SM100+ | SM80+ |
| 输出类型 | FP16/BF16 | FP16/BF16 | FP16/BF16 |
| 尾部融合 | BIAS | - | BIAS + LoRA-up |
| Shuffle归约 | `__shfl_down_sync` | `__shfl_xor_sync` | - |
| Swizzle布局 | 128×4分形 | 128×4分形 | 无 |

**NVFP4独有特性**:
1. **FP8 E4M3 block scale**: 可表示非2的幂的缩放因子 (448.0范围内任意值), 比E8M0精度高得多
2. **Per-tensor global scale**: 额外的FP32标量缩放, 充当"超级缩放", 防止FP8 E4M3溢出
3. **Block size=16**: 比MXFP8的32更细粒度, 量化误差更小
4. **`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`**: cuBLASLt专用scale模式

## 11. cuBLASLt动态加载架构 (`cublaslt_runtime.h`)

comfy-kitchen不链接cuBLASLt, 而是**运行时动态加载**:

```cpp
class CublasLtRuntime {
    // Windows: cublasLt64_13.dll → cublasLt64.dll
    // Linux:   libcublasLt.so.13 → libcublasLt.so
    void* handle_ = LoadLibraryA("cublasLt64_13.dll");

    // 12个函数指针, 运行时解析
    cublasLtCreate_t cublasLtCreate;
    cublasLtMatmul_t cublasLtMatmul;
    // ... 等12个
};
```

**优势**: 无编译期cuBLAS依赖, 支持多CUDA版本共存, 跨平台(Windows/Linux)。

## 12. 内存布局与量化流水线总结

```
量化全流程:
  Input (M, K) FP16/BF16
    → pad 16×16 → (M_pad, K_pad)
    → per_tensor_scale = amax / (6.0 × 448.0)
    → 每block(16元素): block_scale = amax_block / 6.0 / per_tensor_scale
    → clamp(block_scale, 448.0) → FP8 E4M3
    → encode_scale = 1.0 / (block_scale_fp8 × per_tensor_scale)
    → val × encode_scale → FP4 E2M1 (cvt.rn.satfinite.e2m1x2.f32)
    → 2×FP4 packed into 1×uint8 (hi_first)
    → Output: qdata (M_pad, K_pad//2) uint8 + block_scales (M_pad_128, K_pad//16_4) FP8

GEMM全流程:
  A_qdata (M, K//2) uint8  +  A_block_scales FP8 E4M3
  B_qdata (N, K//2) uint8  +  B_block_scales FP8 E4M3
  alpha = A_tensor_scale × B_tensor_scale  (FP32, device)
  cuBLASLt: TN layout, CUDA_R_4F_E2M1, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
  → Output: (M, N) FP16/BF16 + optional BIAS

反量化全流程:
  qdata (M, K//2) uint8
    → __nv_cvt_fp4x2_to_halfraw2 / cvt.rn.f16x2.e2m1x2 → half2
    → × (block_scale_fp8 × global_scale)
    → Output (M, K) FP16/BF16/FP32
```

## 13. 关键源文件索引

| 文件 | 行数 | 含金量 |
|------|------|--------|
| `backends/cuda/ops/quantize_nvfp4.cu` | 437 | ★★★★★ 量化/反量化CUDA内核 |
| `backends/cuda/ops/cublas_gemm_nvfp4.cu` | 344 | ★★★★★ cuBLASLt FP4 GEMM |
| `backends/cuda/float_utils.cuh` | 132 | ★★★★ FP4/FP8工具 + Swizzle |
| `backends/triton/quantization.py` | 760 | ★★★★★ PTX内联汇编量化/反量化 |
| `tensor/nvfp4.py` | 264 | ★★★★ QuantizedLayout + torch.compile调度 |
| `float_utils.py` | 382 | ★★★★ 位操作参考实现 + E8M0 + Swizzle Python版 |
| `scaled_mm_v2.py` | 99 | ★★★ PyTorch scaled_mm V2适配 |
| `backends/cuda/cublaslt_runtime.h` | 248 | ★★★ cuBLASLt动态加载 |
| `backends/cuda/__init__.py` | 1061 | ★★★★ Python绑定 + 约束 + scaled_mm_nvfp4 |
| `samples/nvfp4_linear.py` | 106 | ★★★ NVFP4Linear示例 |
