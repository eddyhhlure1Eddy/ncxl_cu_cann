# SGLang H200/SM120 架构与 PTX 指令深度挖掘报告

> 源码路径: `C:\Users\Administrator\Desktop\nccl\sglang`
> 生成时间: 2026-05-29
> 最新提交: `ec075d8bc`

---

## 1. GPU 架构检测体系

### 1.1 Python 层架构检测

**File:** `python/sglang/srt/utils/common.py:259-285`

| 函数 | 架构范围 | CUDA版本要求 | 对应GPU |
|------|---------|------------|---------|
| `is_hopper_with_cuda_12_3` | major=9 | ≥12.3 | H100, H200 |
| `is_blackwell_supported` / `is_blackwell` | major=10,11,12 | ≥12.8 | B200, GB200, RTX 5090 |
| `is_sm90_supported` | major=9 | ≥12.3 | H100, H200 |
| `is_sm100_supported` | major=10 | ≥12.8 | B200 |
| `is_sm120_supported` | major=12 | ≥12.8 | RTX 5090 (sm_120), DGX Spark (sm_121) |

### 1.2 CUDA C++ 层架构检测

**File:** `sgl-kernel/include/utils.h:254-258`

```cpp
int sm_major = 0, sm_minor = 0;
cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
return sm_major * 10 + sm_minor;
```

### 1.3 JIT 内核架构宏

**File:** `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh:104-117`

```cpp
#if !defined(SGL_CUDA_ARCH)
#error "SGL_CUDA_ARCH is not defined. JIT compilation must inject -DSGL_CUDA_ARCH via load_jit()."
#endif
#define SGL_ARCH_HOPPER_OR_GREATER    (SGL_CUDA_ARCH >= 900)
#define SGL_ARCH_BLACKWELL_OR_GREATER ((SGL_CUDA_ARCH >= 1000) && (CUDA_VERSION >= 12090))
// Blackwell+: 256-bit (32 bytes) vector, Pre-Blackwell: 128-bit (16 bytes)
inline constexpr std::size_t kMaxVecBytes = SGL_ARCH_BLACKWELL_OR_GREATER ? 32 : 16;
```

### 1.4 sgl-kernel Compute Capability 分发

**File:** `sgl-kernel/python/sgl_kernel/load_utils.py:15-65`

```python
compute_capability = _get_compute_capability()  # e.g. 90, 100, 120
if compute_capability == 90:
    variant_name = "SM90 (Hopper optimized)"
elif compute_capability is not None:
    variant_name = f"SM{compute_capability} (precise math for compatibility)"
```

---

## 2. H200 (Hopper) 特有实现

### 2.1 H200 CI 测试矩阵

**File:** `test/run_suite.py:50-146`

| 套件 | GPU | 规模 |
|------|-----|------|
| `per-commit-8-gpu-h200` | H200 | 8-GPU |
| `per-commit-8-gpu-h200-deepep` | H200 | 8-GPU DeepEP |
| `nightly-8-gpu-h200` | H200 | 8-GPU |
| `nightly-8-gpu-h200-basic` | H200 | 8-GPU大模型 |
| `nightly-8-gpu-common` | H200+B200 | 通用 |
| `nightly-kernel-8-gpu-h200` | H200 | 内核单元测试 |
| `weekly-8-gpu-h200` | H200 | 周测试 |

### 2.2 Hopper 专用特性应用

| 特性 | 文件 | 行号 | 说明 |
|------|------|------|------|
| PDL (Programmatic Dependent Launch) | `jit_kernel/include/sgl_kernel/utils.cuh` | 136-162 | `griddepcontrol.wait` / `griddepcontrol.launch_dependents` |
| SBO SM分配 | `batch_overlap/single_batch_overlap.py` | 99 | `communicate_num_sms = 3` (Hopper) |
| EAGLE推测解码 | `server_args.py` | 2705,2730 | `is_hopper_with_cuda_12_3()` 启用优化 |
| 注意力后端 | `layers/attention/attention_registry.py` | 286 | Hopper走FA3/Triton路径 |

### 2.3 PDL (程序化依赖启动) — Hopper 专属

**File:** `jit_kernel/include/sgl_kernel/utils.cuh:140-162`

```cuda
// 等待前序内核完成 (仅Hopper)
template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if SGL_ARCH_HOPPER_OR_GREATER
    if constexpr (kUsePDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
}

// 触发依赖内核 (仅Hopper)
template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if SGL_ARCH_HOPPER_OR_GREATER
    if constexpr (kUsePDL) {
        asm volatile("griddepcontrol.launch_dependents;" :::);
    }
#endif
}
```

---

## 3. SM120 (Consumer Blackwell) NVFP4 实现

### 3.1 NVFP4 GEMM 内核分发

**File:** `jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh:122-145`

```cpp
const int sm_version = getSMVersion(A.device().device_id);
if (sm_version >= 120) {
    // SM120+: 使用 sm120 专用配置
    if (is_type<fp16_t>(D.dtype())) {
        cutlass_fp4_f16_gemm_dispatch_sm120(...);
    } else if (is_type<bf16_t>(D.dtype())) {
        cutlass_fp4_bf16_gemm_dispatch_sm120(...);
    } else {
        Panic("Unsupported output data type of nvfp4 mm sm120");
        // 注意: SM120不支持FP32输出！
    }
} else {
    // SM100: 使用 sm100 专用配置 (支持FP16/BF16/FP32)
    cutlassFp4GemmDispatchSm100<OutType>(...);
}
```

### 3.2 SM120 FP4 GEMM 配置

**File:** `jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm120.cuh:20-228`

| 配置 | ClusterShape | MmaTileShape | PerSmTileShape | 适用M范围 |
|------|-------------|-------------|---------------|----------|
| `sm120_fp4_config_small_m` | (1,1,1) | (128,128,256) | (128,128,256) | M ≤ 32 |
| `sm120_fp4_config_M256` | (1,1,1) | (128,128,128) | (128,128,128) | 32 < M ≤ 256 |
| `sm120_fp4_config_default` | (1,1,1) | (256,128,128) | (256,128,128) | M > 256 |

### 3.3 SM120 vs SM100 FP4 GEMM 关键差异

| 维度 | SM100 | SM120 |
|------|-------|-------|
| ArchTag | `cutlass::arch::Sm100` | `cutlass::arch::Sm120` |
| OperatorClass | `OpClassBlockScaledTensorOp` | `OpClassBlockScaledTensorOp` |
| EpilogueSchedule | `TmaWarpSpecialized1Sm/2Sm` | `EpilogueScheduleAuto` |
| MainloopSchedule | `KernelTmaWarpSpecialized1Sm/2SmNvf4Sm100` | `KernelScheduleAuto` |
| ClusterShape | `Shape<int, int, _1>` (动态) | `Shape<_1, _1, _1>` (固定) |
| EpilogueTile | `Shape<_128, _64>` (手动) | `EpilogueTileAuto` |
| FP32输出 | ✓ | ✗ |
| Cluster动态fallback | ✓ (`preferred_cluster` / `fallback_cluster`) | ✗ (单cluster) |
| 对齐A/B | 32 | 32 |
| ScaleFactor类型 | `float_ue4m3_t` | `float_ue4m3_t` |
| ElementA/B | `nv_float4_t<float_e2m1_t>` | `nv_float4_t<float_e2m1_t>` |

### 3.4 SM100 FP4 GEMM 配置 (4级M分发)

**File:** `jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm100.cuh:22-101`

| 配置 | ClusterShape | MmaTileShape | Epilogue | 适用M范围 |
|------|-------------|-------------|----------|----------|
| `KernelConfigM128` | (1,4,1)/(1,2,1) | (128,256,256) | (128,64) 1Sm | M ≤ 128 |
| `KernelConfigM256` | (2,4,1)/(2,1,1) | (256,256,256) | (128,64) 2Sm | M ≤ 256 |
| `KernelConfigDefault` | (2,4,1)/(2,1,1) | (256,256,256) | (128,64) 2Sm | 256 < M ≤ 1024 |
| `KernelConfigLargeM` | (1,4,1)/(1,2,1) | (256,256,256) | (128,64) 2Sm | M > 1024 |
| `KernelConfigFp32` | (1,4,1)/(1,2,1) | (128,128,256) | Auto 1Sm | FP32输出 |

---

## 4. Blackwell (SM100/SM120) 系统级特性

### 4.1 DeepSeek-V4 多流BS限制

**File:** `python/sglang/srt/models/deepseek_v4.py:316-318`

```python
from sglang.srt.utils import is_blackwell_supported
self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64
```

### 4.2 SBO SM分配

**File:** `batch_overlap/single_batch_overlap.py:99,118`

```python
communicate_num_sms = 32 if is_blackwell() else 3  # Hopper: 3 SM, Blackwell: 32 SM
if is_blackwell():
    # Blackwell通信占用32 SM
```

### 4.3 EAGLE推测解码 Blackwell 限制

**File:** `python/sglang/srt/speculative/draft_utils.py:50,79`

```python
if not is_blackwell():
    # 非Blackwell架构的特殊处理
```

### 4.4 GlmMoeDSA 模型 Blackwell 优化

**File:** `server_args.py:1806`

```python
if model_arch == "GlmMoeDsaForCausalLM" and is_blackwell_supported():
    # Blackwell专用DSA注意力路径
```

### 4.5 FP4 KV Cache 量化

**File:** `layers/quantization/fp4_kv_cache_quant_method.py:105-137`

```python
class NVFP4KVMethod:
    def __init__(self, num_layers, device, sm_version=120):
        self.sm_version = sm_version
    
    def load_scales_from_model(self, model_runner, sm_version=None):
        if sm_version is not None:
            self.sm_version = sm_version
        # SM100特殊路径
        if self.sm_version == 100:
            # SM100的scale加载方式不同
```

### 4.6 NVFP4 W4A4 MoE 量化

**File:** `layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py:42`

```python
if not is_blackwell_supported():
    raise ValueError("NVFP4 MoE requires Blackwell (SM100/SM120)")
```

### 4.7 Blackwell Flash Attention 3

**File:** `layers/attention/attention_registry.py:264-286`

```python
from sglang.srt.utils import is_blackwell, is_npu
if is_blackwell():
    # 使用Blackwell优化注意力路径
```

**File:** `layers/attention/vision.py:1036`

```python
if backend == "fa3" and is_blackwell_supported():
    # FA3 + Blackwell特殊配置
```

---

## 5. PTX 内联汇编指令全表

### 5.1 系统级内存操作 (AllReduce/Custom AR)

**File:** `sgl-kernel/csrc/allreduce/custom_all_reduce.cuh:153-186`

| PTX指令 | 行号 | 语义 | 架构要求 |
|---------|------|------|---------|
| `st.release.sys.global.u32` | 157 | 系统级Release存储 | sm_70+ |
| `ld.acquire.sys.global.u32` | 171 | 系统级Acquire加载 | sm_70+ |
| `membar.sys` | 159 | 系统级内存屏障 | sm_70以下 |
| `st.volatile.global.u32` | 159,179 | Volatile存储 | 所有 |
| `ld.volatile.global.u32` | 173,184 | Volatile加载 | 所有 |
| `membar.gl` | 173 | GPU级内存屏障 | sm_70以下 |

### 5.2 Tensor Core MMA 指令

**File:** `sgl-kernel/csrc/gemm/` 各文件

| PTX指令 | 文件 | 行号 | 类型 |
|---------|------|------|------|
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | `marlin/marlin_template.h` | 89 | FP16 Tensor Core |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `marlin/marlin_template.h` | 95 | BF16 Tensor Core |
| `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` | `qserve_w4a8_per_group_gemm.cu` | 136 | INT8 Tensor Core |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `dsv3_fused_a_gemm.cu` | 40 | BF16 Tensor Core |

### 5.3 异步内存拷贝 (cp.async)

**File:** `sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu:67,86,91`

| PTX指令 | 行号 | 语义 |
|---------|------|------|
| `cp.async.cg.shared.global.L2::128B` | 67 | 128B异步拷贝 (L2缓存提示) |
| `cp.async.commit_group` | 86 | 提交异步拷贝组 |
| `cp.async.wait_group N` | 91 | 等待N组完成 |

**File:** `sgl-kernel/csrc/gemm/marlin/marlin.cuh:63-91`

```cuda
// Marlin GEMM异步拷贝
asm volatile(
    "{  .reg .pred p;\n"
    "  setp.ne.b32 p, %2, 0;\n"
    "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}" :: ... );
asm volatile("cp.async.commit_group;\n" ::);
asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
```

### 5.4 LDMATRIX (矩阵加载)

**File:** `sgl-kernel/csrc/gemm/marlin/marlin_template.h:156-162`

```cuda
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem));
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
             : "=r"(a[0]), "=r"(a[1]) : "r"(smem));
asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
             : "=r"(a[0]) : "r"(smem));
```

### 5.5 mbarrier (内存屏障初始化/等待)

**File:** `sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu:96,107,153`

```cuda
asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(thread_count));
asm volatile(
    "{\n"
    "  .reg .b64 rem;\n"
    "  mbarrier.arrive.shared.b64 rem, [%0];\n"
    "  mbarrier.test_wait.shared.b64 %1, rem, %2;\n"
    "}\n" :: ... );
asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(smem_int_ptr));
```

### 5.6 bar.sync (CTA 屏障)

**File:** `sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu:432,465`

```cuda
asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));
```

### 5.7 TMA (Tensor Memory Accelerator) — Blackwell/Hopper

**File:** `jit_kernel/include/sgl_kernel/deepseek_v4/topk/ptx.cuh:29-30`

```cuda
SGL_DEVICE void tma_load(void* dst, const void* src, uint32_t num_bytes, uint64_t* mbar) {
    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global, 
                              dst, src, num_bytes, mbar);
}
```

**File:** `sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh:233-256`

```cuda
cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());  // 等待TMA完成
cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);             // 异步代理fence
cuda::ptx::cp_async_bulk(                                            // TMA bulk拷贝
    cuda::ptx::space_global, cuda::ptx::space_shared, dst, src, num_bytes, mbar);
cuda::ptx::cp_async_bulk_commit_group();                             // 提交TMA组
```

### 5.8 LOP3 (三输入逻辑操作)

**File:** `sgl-kernel/csrc/gemm/awq_kernel.cu:14` + `marlin/dequant.h:77`

```cuda
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
```

### 5.9 PRMT (字节排列)

**File:** `sgl-kernel/csrc/gemm/marlin/dequant.h:86`

```cuda
asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(res) : "r"(a), "n"(start_byte), "n"(mask));
```

### 5.10 FMA / 算术指令

| PTX指令 | 文件 | 行号 | 类型 |
|---------|------|------|------|
| `fma.rn.f32x2` | `dsv3_router_gemm_float_out.cu` | 30 | FP32x2 FMA |
| `fma.rn.f16x2` | `awq_kernel.cu` | 69,73 | FP16x2 FMA |
| `sub.f16x2` | `awq_kernel.cu` | 67,71 | FP16x2 减法 |
| `mul.rn.f16x2` | `awq_kernel.cu` | 152-158 | FP16x2 乘法 |
| `rcp.approx.ftz.f32` | `es_sm100_mxfp8_blockscaled_group_quant.cuh` | 30 | FP32近似倒数 |

### 5.11 L2缓存控制存储

**File:** `sgl-kernel/csrc/elementwise/utils.cuh:28-77`

```cuda
// L1 no-allocate存储 (避免污染L1缓存)
asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v));
asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" :: ...);
asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" :: ...);

// L1 no-allocate + L2 128B提示加载
asm volatile("ld.global.nc.L1::no_allocate.L2::128B.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v2.s32 {%0, %1}, [%2];" :: ...);
asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];" :: ...);

// 预取到L2
asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
```

### 5.12 锁与原子操作 (Marlin GEMM)

**File:** `sgl-kernel/csrc/gemm/marlin/marlin_template.h:234-263`

```cuda
asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  // 获取锁
asm volatile("fence.acq_rel.gpu;\n");                                               // GPU fence
asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val)); // 原子加
```

### 5.13 KV Cache I/O (NC传输)

**File:** `sgl-kernel/csrc/kvcacheio/transfer.cu:29-30`

```cuda
asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");  // 非一致性加载
asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");    // 全局一致性存储
```

### 5.14 量化存储 (per_token_group_quant)

**File:** `sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu:94-105`

```cuda
asm volatile("st.global.v4.s32 [%0], {%1, %2, %3, %4};" :: ...);  // 128-bit向量存储
asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];" :: ...);  // 非一致性128-bit加载
```

### 5.15 cp.async.cg (QServe GEMM)

**File:** `sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu:100-135`

```cuda
__asm__ __volatile__(
    "{  .reg .pred p;\n"
    "  setp.ne.b32 p, %2, 0;\n"
    "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
    "}" :: ... );
```

---

## 6. AMD GPU 汇编指令 (Quick AllReduce)

**File:** `sgl-kernel/csrc/allreduce/quick_all_reduce_base.h:87-266`

| AMD指令 | 行号 | 语义 | NVIDIA等效 |
|---------|------|------|-----------|
| `s_setreg_imm32_b32 0xdc1` | 90,92 | FP16溢出模式控制 | N/A |
| `v_pk_add_f16` | 109-112,193 | 打包FP16加法 | `add.f16x2` |
| `v_pk_max_f16` | 131 | 打包FP16最大值 | N/A |
| `v_pk_min_f16` | 150 | 打包FP16最小值 | N/A |
| `v_pk_add_i16` | 209 | 打包INT16加法 | N/A |
| `v_pk_fma_f16` | 221 | 打包FP16 FMA | `fma.rn.f16x2` |
| `v_pk_mul_f16` | 240 | 打包FP16乘法 | `mul.rn.f16x2` |
| `llvm.amdgcn.raw.buffer.load.v4i32` | 80-81 | 缓冲区加载 | `ld.global.nc.v4` |
| `llvm.amdgcn.raw.buffer.store.v4i32` | 83-85 | 缓冲区存储 | `st.global.v4` |

---

## 7. __CUDA_ARCH__ 守卫汇总

| 守卫 | 值 | 文件 | 含义 |
|------|-----|------|------|
| `__CUDA_ARCH__ >= 700` | sm_70+ | `custom_all_reduce.cuh:156,170` | st.release/ld.acquire |
| `__CUDA_ARCH__ >= 800` | sm_80+ | `mscclpp_allreduce.cuh:51,73,98,120` | BF16向量操作 |
| `__CUDA_ARCH__` | sm_80+ | `custom_all_reduce.cu:85` | BF16 AllReduce |
| `__gfx942__` | MI300 | `quick_all_reduce_base.h:23,88` | CDNA3 MUBUF语义 |
| `__gfx908__` | MI100 | `quick_all_reduce_base.h:27` | CDNA1 |
| `__gfx90a__` | MI250 | `quick_all_reduce_base.h:27` | CDNA2 |
| `CUTLASS_ARCH_MMA_SM100_SUPPORTED` | sm_100 | `nvfp4_scaled_mm_sm100.cuh:20` | SM100 FP4 GEMM |
| `CUTLASS_ARCH_MMA_SM120_SUPPORTED` | sm_120 | `nvfp4_scaled_mm_sm120.cuh:20` | SM120 FP4 GEMM |
| `CUTLASS_ARCH_MMA_SM121_SUPPORTED` | sm_121 | `nvfp4_scaled_mm_sm120.cuh:20` | SM121 FP4 GEMM |
| `SGL_CUDA_ARCH >= 900` | sm_90+ | `utils.cuh:112` | Hopper PDL |
| `SGL_CUDA_ARCH >= 1000` | sm_100+ | `utils.cuh:113` | Blackwell 256-bit vector |
| `sm_version >= 80` | sm_80+ | `qserve_w4a8_per_group_gemm.cu:748` | QServe GEMM |
| `sm_version >= 120` | sm_120+ | `nvfp4_scaled_mm_kernels.cuh:125` | NVFP4 SM120路径 |

---

## 8. CUTLASS SM120 Block-Scaled GEMM 深度分析

### 8.1 nv_float4_t<float_e2m1_t> — NVFP4 数据类型

```cpp
// SM120/SM100 Block-Scaled GEMM核心类型
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // 2x FP4 打包为uint8
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementSFA = cutlass::float_ue4m3_t;  // Block Scale Factor (E4M3无符号)
using ElementSFB = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
```

### 8.2 Scale Factor Swizzling

```cpp
// SM120: 使用 Sm1xxBlkScaledConfig
auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
```

### 8.3 SM120 FP4 输出类型限制

```cpp
// SM120: 仅支持 FP16 和 BF16 输出
if (host::is_type<fp16_t>(D.dtype())) cutlass_fp4_f16_gemm_dispatch_sm120(...);
else if (host::is_type<bf16_t>(D.dtype())) cutlass_fp4_bf16_gemm_dispatch_sm120(...);
else Panic("Unsupported output data type of nvfp4 mm sm120");
// 对比 SM100: 还支持 FP32 输出
```

---

## 9. Marlin MoE NVFP4 内核

**File:** `jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh:319-456`

```cuda
// NVFP4: cases for nvfp4(e2m1) (group_blocks == 1)
#define NVFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)    \
  if (W_TYPE == kFE2M1f && N_BLOCKS == 8 && K_BLOCKS == 8 && NUM_THREADS == 256) \
    return true;                                                      \
  if (W_TYPE == kFE2M1f && N_BLOCKS == 8 && K_BLOCKS == 4 && NUM_THREADS == 128) \
    return true;

#define NVFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)   \
  if (W_TYPE == kFE2M1f && N_BLOCKS == 16 && K_BLOCKS == 4 && NUM_THREADS == 256) \
    return true;                                                       \
  if (W_TYPE == kFE2M1f && N_BLOCKS == 8 && K_BLOCKS == 4 && NUM_THREADS == 128) \
    return true;

#define NVFP4_GET_IF(W_TYPE)            \
  NVFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  NVFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  NVFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  NVFP4_GET_IF_M234(W_TYPE, 8, 4, 128)
```

**关键约束:**
```cuda
RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, 
             "global_scale can only be used for nvfp4 format.");
RuntimeCheck(
    !(b_q_type == kFE2M1f && group_size == 16), 
    "the global_scale parameter must be passed for nvfp4 format.");
"float4_e2m1f only supports group_size == 16 (NVFP4) or group_size == 32 (MXFP4)."
```

---

## 10. DeepSeek-V4 TopK TMA 加载

**File:** `jit_kernel/include/sgl_kernel/deepseek_v4/topk/`

| 文件 | TMA用法 |
|------|---------|
| `ptx.cuh:29-30` | `tma_load()` 封装 `cuda::ptx::cp_async_bulk` |
| `cluster.cuh:81` | `ptx::tma_load(smem->score_buffer[stage], scores + offset, size_bytes, bar)` |
| `streaming.cuh:56` | `ptx::tma_load(smem->score_buffer[buf_idx], scores + offset, size_bytes, bar)` |
| `register.cuh:81` | `ptx::tma_load(smem->score_buffer, scores + kMax1PassLength, size_bytes, &smem->mbarrier)` |

---

## 11. CuteDSL MoE (Blackwell 专用)

**File:** `test/registered/moe/test_flashinfer_a2a_cutedsl_v2.py:26`

```python
SKIP_REASON = "Requires Blackwell (B200, sm_100a) or above."
```

CuteDSL FP4 MoE 需要 `sm_100a` (B200) 或更高。

---

## 12. Blackwell/SM120 架构对比总结

| 维度 | H200 (sm_90a) | B200 (sm_100a) | RTX 5090 (sm_120) | DGX Spark (sm_121) |
|------|--------------|---------------|-------------------|-------------------|
| PDL | ✓ | ✓ | ✓ | ✓ |
| TMA | ✓ | ✓ | ✓ | ✓ |
| NVFP4 GEMM | ✗ | ✓ (Sm100) | ✓ (Sm120) | ✓ (Sm121) |
| FP4 KV Cache | ✗ | ✓ | ✓ | ✓ |
| Block-Scaled Tensor Core | ✗ | ✓ | ✓ | ✓ |
| Cluster Launch | ✓ | ✓ (动态) | ✓ (固定1x1x1) | ✓ (固定1x1x1) |
| FP32输出FP4 GEMM | N/A | ✓ | ✗ | ✗ |
| SBO通信SM数 | 3 | 32 | 32 | 32 |
| Multi-stream BS Limit | 64 | 128 | 128 | 128 |
| CuteDSL MoE | ✗ | ✓ | ✓ | ✓ |
| FlashInfer b12x | ✗ | ✗ | ✓ | ✓ |
| kMaxVecBytes | 16 | 32 | 32 | 32 |
