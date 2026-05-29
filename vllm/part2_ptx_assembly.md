# Part 2: PTX Assembly & Inline PTX Instructions in vLLM

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. Summary of PTX Instruction Categories

| Category | Instruction | SM Min | Primary Files |
|----------|------------|--------|---------------|
| Async Copy | `cp.async` | sm_80 | `csrc/quantization/marlin/marlin.cuh`, `csrc/quantization/activation_kernels.cu` |
| Async Copy Commit | `cp.async.commit_group` | sm_80 | Same as above |
| Async Copy Wait | `cp.async.wait_group` / `cp.async.wait_all` | sm_80 | Same as above |
| Load Matrix | `ldmatrix.sync.aligned.m8n8` | sm_75 | `csrc/quantization/marlin/marlin_template.h` |
| Load Global | `ld.global.acquire.gpu.b32` | sm_70 | `csrc/persistent_topk.cuh`, `csrc/quantization/marlin/marlin_template.h` |
| Load Global CG | `ld.global.cg.v4.u32` | sm_70 | `csrc/persistent_topk.cuh` |
| Load Volatile | `ld.volatile.global.v4.f32` | sm_70 | `csrc/minimax_reduce_rms_kernel.cu` |
| Store Release | `st.release.gpu.global.b32` | sm_70 | `csrc/persistent_topk.cuh` |
| Fence | `fence.acq_rel.gpu` | sm_70 | `csrc/persistent_topk.cuh`, `csrc/quantization/marlin/marlin_template.h` |
| Atomic Reduce | `red.relaxed.gpu.global.add.s32` | sm_70 | Same as fence |
| Logic Op | `lop3.b32` | sm_70 | `csrc/quantization/marlin/dequant.h`, `csrc/moe/moe_wna16_utils.h` |
| Permute | `prmt.b32` | sm_70 | Same as lop3 |
| RCP Approx | `rcp.approx.ftz.f32` | sm_70 | `csrc/moe/mxfp8_moe/mxfp8_experts_quant.cuh` |
| FMA | `fma.rn.f32x2` | sm_70 | `csrc/moe/dsv3_router_gemm_float_out.cu` |
| Grid Dep Control | `griddepcontrol.wait` / `griddepcontrol.launch_dependents` | sm_90 | Multiple MOE kernels |
| LDSM | `ldsm_4` (ldmatrix wrapper) | sm_75 | `csrc/libtorch_stable/quantization/gptq_allspark/allspark_utils.cuh` |

---

## 2. Async Memory Copy (cp.async) - sm_80+

### 2.1 Marlin Kernel Async Copy

**File:** `csrc/quantization/marlin/marlin.cuh`

```cpp
// Line 110-161: cp.async for loading weight tiles from global to shared memory
asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n"  // Copy global→shared
    :: "r"(smem_ptr), "l"(gmem_ptr), "n"(bytes)
);

// Line 169: Commit async group
asm volatile("cp.async.commit_group;\n" ::);

// Line 174: Wait for N groups to complete
asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
```

### 2.2 Activation Kernel Async Copy

**File:** `csrc/quantization/activation_kernels.cu`

```cpp
// Line 153-186: cp.async for activation quantization
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n"
    :: "r"(smem_ptr), "l"(gmem_ptr), "n"(bytes)
);
asm volatile("cp.async.commit_group;\n" ::);
asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
asm volatile("cp.async.wait_all;\n" ::);
#endif
```

---

## 3. Matrix Load (ldmatrix) - sm_75+

### 3.1 Marlin Template ldmatrix

**File:** `csrc/quantization/marlin/marlin_template.h`

```cpp
// Line 87-96: ldmatrix for loading matrix fragments from shared memory
// x2 variant (load 2 8x8 matrices)
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(r0), "=r"(r1) : "r"(smem_addr));

// x1 variant (load 1 8x8 matrix)
asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
    : "=r"(r0) : "r"(smem_addr));
```

### 3.2 MOE WNA16 ldmatrix

**File:** `csrc/moe/marlin_moe_wna16/marlin_template.h`

```cpp
// Line 95-104: Same ldmatrix patterns for MOE quantized kernels
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" ...);
asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" ...);
```

### 3.3 AllSpark LDSM Wrapper

**File:** `csrc/libtorch_stable/quantization/gptq_allspark/allspark_utils.cuh`

```cpp
// Line 244-246: ldsm_4 wrapper for loading 4 matrix elements
__device__ __forceinline__ void ldsm_4(T& r0, T& r1, T& r2, T& r3, ...) {
  static_assert(sizeof(T) == 4, "ldsm_4: invalid T");
  // Wraps ldmatrix.m8n8.x4
}
```

**Used in:** `csrc/libtorch_stable/quantization/gptq_allspark/allspark_qgemm_w8a16.cu:289`

---

## 4. Memory Fence & Atomic Operations - sm_70+

### 4.1 Fence + Atomic Reduce

**File:** `csrc/persistent_topk.cuh`

```cpp
// Line 600-624: Memory fence and atomic reduction for persistent TopK
// Line 600: Load with acquire semantics
asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(val) : "l"(ptr));

// Line 604: Cache-global load
asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));

// Line 611: GPU-wide acquire-release fence
asm volatile("fence.acq_rel.gpu;\n");

// Line 612: Relaxed atomic add
asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));

// Line 623: Another fence
asm volatile("fence.acq_rel.gpu;\n");

// Line 624: Release store
asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
```

### 4.2 Marlin Fence + Reduce

**File:** `csrc/quantization/marlin/marlin_template.h`

```cpp
// Line 183-203: Lock-based synchronization with fence + atomic reduce
asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(lock) : "l"(lock_ptr));
asm volatile("fence.acq_rel.gpu;\n");
asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));
```

### 4.3 MOE WNA16 Fence + Reduce

**File:** `csrc/moe/marlin_moe_wna16/marlin_template.h`

```cpp
// Line 191-211: Same pattern as Marlin for MOE kernels
asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" ...);
asm volatile("fence.acq_rel.gpu;\n");
asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" ...);
```

---

## 5. Bit Manipulation (lop3, prmt) - sm_70+

### 5.1 lop3.b32 (Logic Operation 3-input)

**File:** `csrc/quantization/marlin/dequant.h`

```cpp
// Line 77: 3-input logic operation for dequantization
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(result) : "r"(a), "r"(b), "r"(c), "n"(lookup));
```

**File:** `csrc/moe/moe_wna16_utils.h`

```cpp
// Line 84: Same lop3 pattern for MOE weight dequantization
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" ...);
```

### 5.2 prmt.b32 (Permute Bytes)

**File:** `csrc/quantization/marlin/dequant.h`

```cpp
// Line 88: Byte permute for weight rearrangement
asm volatile("prmt.b32 %0, %1, %2, %3;\n"
    : "=r"(result) : "r"(a), "r"(b), "r"(ctrl));
```

**File:** `csrc/moe/moe_wna16_utils.h:93`

---

## 6. Grid Dependency Control (Hopper sm_90+)

### 6.1 Persistent TopK / Softplus

**File:** `csrc/moe/topk_softplus_sqrt_kernels.cu`

```cpp
// Line 171: Wait for dependent grids
asm volatile("griddepcontrol.wait;");

// Line 298, 423: Launch dependent grids
asm volatile("griddepcontrol.launch_dependents;");
```

### 6.2 Grouped TopK

**File:** `csrc/moe/grouped_topk_kernels.cu`

```cpp
// Line 562: Grid dependency wait
asm volatile("griddepcontrol.wait;");

// Line 607, 668: Launch dependents
asm volatile("griddepcontrol.launch_dependents;");
```

### 6.3 MiniMax Reduce RMS

**File:** `csrc/minimax_reduce_rms_kernel.cu`

```cpp
// Line 249, 384: Grid dependency control
asm volatile("griddepcontrol.wait;");

// Line 313, 596: Launch dependents
asm volatile("griddepcontrol.launch_dependents;");
```

### 6.4 DSV3 Router GEMM

**File:** `csrc/moe/dsv3_router_gemm_float_out.cu`

```cpp
// Line 84: Grid dependency wait
asm volatile("griddepcontrol.wait;");
```

---

## 7. Miscellaneous PTX Instructions

### 7.1 RCP Approximate

**File:** `csrc/moe/mxfp8_moe/mxfp8_experts_quant.cuh`

```cpp
// Line 36: Fast reciprocal approximation
asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
```

### 7.2 FMA x2

**File:** `csrc/moe/dsv3_router_gemm_float_out.cu`

```cpp
// Line 32: Dual FMA for fused multiply-add
asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
    : "=f"(d), "=f"(e) : "f"(a), "f"(b), "f"(c), "f"(d));
```

### 7.3 Volatile Load

**File:** `csrc/minimax_reduce_rms_kernel.cu`

```cpp
// Line 116: Vector volatile load (4 floats)
asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(addr));

// Line 124: Scalar volatile load
asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
```

### 7.4 Vector Load CG

**File:** `csrc/persistent_topk.cuh`

```cpp
// Line 69: Cache-global vector load (4 u32)
asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(ptr));
```

### 7.5 LDG.E.128 Note (sm_100)

**File:** `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`

```cpp
// Line 296: Comment about sm_100 behavior
// nvcc keeps these as 2x LDG.E.128 on sm_100; the per-thread cost...
```

---

## 8. AMD ROCm PTX-equivalent (s_waitcnt, v_pk_*)

**File:** `csrc/rocm/skinny_gemms.cu`

```cpp
// Line 1550, 1636, 1674, 1886, 2081: AMD wait counter instructions
asm volatile("s_waitcnt 0");
asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
```

**File:** `csrc/quickreduce/base.h` and `csrc/quickreduce/quick_reduce_impl.cuh`

```cpp
// Line 93: Set register immediate
asm volatile("s_setreg_imm32_b32 0xdc1, 1;" ::);

// Line 114-258: Packed FP16 operations
asm volatile("v_pk_add_f16 %0, %1, %2" ...);  // Packed FP16 add
asm volatile("v_pk_max_f16 %0, %1, %2" ...);  // Packed FP16 max
asm volatile("v_pk_min_f16 %0, %1, %2" ...);  // Packed FP16 min
asm volatile("v_pk_add_i16 %0, %1, %2" ...);  // Packed INT16 add
asm volatile("v_pk_fma_f16 %0, %1, %2 %3" ...);  // Packed FP16 FMA
asm volatile("v_pk_mul_f16 %0, %1, %2" ...);  // Packed FP16 multiply
```
