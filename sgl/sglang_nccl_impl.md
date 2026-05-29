# SGLang NCCL 集合通信深度挖掘报告

> 源码路径: `C:\Users\Administrator\Desktop\nccl\sglang`
> 生成时间: 2026-05-29
> 最新提交: `ec075d8bc`

---

## 1. 通信架构全景

SGLang 实现了 **6 种 AllReduce 后端**，通过 `GroupCoordinator` 统一调度，按优先级自动选择：

```
AllReduce 优先级 (parallel_state.py:562-639):
  1. CustomAllReduce (ca)      — NVIDIA NVLink 自定义内核，IPC共享内存
  2. QuickAllReduce (qr)       — AMD MI300 特有，gfx942+ DPP指令
  3. PyMscclpp (pymscclpp)     — Microsoft MSCCL++ 单边通信
  4. TorchSymmMem (torch_symm) — PyTorch 对称内存 AllReduce
  5. PyNccl (pynccl)           — 直接调用 libnccl.so
  6. torch.distributed         — PyTorch 默认 NCCL 后端 (fallback)
```

### 通信架构选择矩阵

| AllReduce后端 | Eager模式 | Graph模式 | 平台 | 大小限制 |
|--------------|-----------|----------|------|---------|
| Custom AR | ✓ | ✓ | NVIDIA | ≤512KB (small), ≤256KB (large) |
| Quick AR | ✓ | ✓ | AMD gfx942+ | 动态阈值 |
| PyMscclpp | ✗ | ✓ | NVIDIA+InfiniBand | 动态阈值 |
| TorchSymmMem | ✗ | ✓ | NVIDIA | 动态阈值 |
| PyNccl | ✗ | ✓ (outplace) | NVIDIA/AMD/MUSA | 无限制 |
| torch.distributed | ✓ | ✗ | 所有 | 无限制 |

**File:** `python/sglang/srt/distributed/parallel_state.py:522-546`

```
allreduce \ Mode   |  Eager  |  Graph  |
--------------------------------------------
quick allreduce    | enabled | enabled |
custom allreduce   | enabled | enabled |
PyNccl             | disabled| enabled |
PyMscclpp          | disabled| enabled |
TorchSymmMem       | disabled| enabled |
torch.distributed  | enabled | disabled|
```

---

## 2. PyNccl — NCCL 直接封装

### 2.1 NCCL 库发现

**File:** `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py:39-64`

| 环境变量 | 用途 | 默认值 |
|---------|------|-------|
| `SGLANG_NCCL_SO_PATH` | 指定NCCL库路径 | 无 |
| (自动) | CUDA: `libnccl.so.2` | - |
| (自动) | ROCm: `librccl.so.1` | - |
| (自动) | MUSA: `libmccl.so.2` | - |

### 2.2 NCCL API 导出

**File:** `pynccl_wrapper.py:160-308`

| NCCL C API | Python函数签名 | 用途 |
|------------|---------------|------|
| `ncclGetErrorString` | `(ncclResult_t) -> c_char_p` | 错误字符串 |
| `ncclGetVersion` | `(int*) -> ncclResult_t` | 版本号 |
| `ncclGetUniqueId` | `(ncclUniqueId*) -> ncclResult_t` | 获取通信唯一ID |
| `ncclCommInitRank` | `(ncclComm_t*, int, ncclUniqueId, int) -> ncclResult_t` | 初始化通信器 |
| `ncclAllReduce` | `(sendbuf, recvbuf, count, dtype, op, comm, stream)` | 全归约 |
| `ncclAllGather` | `(sendbuf, recvbuf, count, dtype, comm, stream)` | 全收集 |
| `ncclReduce` | `(sendbuf, recvbuf, count, dtype, op, root, comm, stream)` | 规约到root |
| `ncclReduceScatter` | `(sendbuf, recvbuf, count, dtype, op, comm, stream)` | 规约散射 |
| `ncclSend` | `(sendbuf, count, dtype, dest, comm, stream)` | 点对点发送 |
| `ncclRecv` | `(recvbuf, count, dtype, src, comm, stream)` | 点对点接收 |
| `ncclBroadcast` | `(sendbuf, recvbuf, count, dtype, root, comm, stream)` | 广播 |
| `ncclCommDestroy` | `(ncclComm_t) -> ncclResult_t` | 销毁通信器 |
| `ncclGroupStart` | `() -> ncclResult_t` | 组操作开始 |
| `ncclGroupEnd` | `() -> ncclResult_t` | 组操作结束 |
| `ncclCommWindowRegister` | `(comm, buff, size, win, flags)` | 对称内存窗口注册 |
| `ncclCommWindowDeregister` | `(comm, win)` | 窗口注销 |

### 2.3 NCCL 数据类型映射

**File:** `pynccl_wrapper.py:87-122`

| NCCL类型 | 值 | PyTorch dtype |
|---------|-----|--------------|
| `ncclInt8` | 0 | `torch.int8` |
| `ncclUint8` | 1 | `torch.uint8` |
| `ncclInt32` | 2 | `torch.int32` |
| `ncclUint32` | 3 | `torch.uint32` (not mapped) |
| `ncclInt64` | 4 | `torch.int64` |
| `ncclUint64` | 5 | `torch.uint64` (not mapped) |
| `ncclFloat16` | 6 | `torch.float16` |
| `ncclFloat32` | 7 | `torch.float32` |
| `ncclFloat64` | 8 | `torch.float64` |
| `ncclBfloat16` | 9 | `torch.bfloat16` |

### 2.4 NCCL 规约操作映射

| NCCL Op | 值 | PyTorch ReduceOp |
|---------|-----|-----------------|
| `ncclSum` | 0 | `ReduceOp.SUM` |
| `ncclProd` | 1 | `ReduceOp.PRODUCT` |
| `ncclMax` | 2 | `ReduceOp.MAX` |
| `ncclMin` | 3 | `ReduceOp.MIN` |
| `ncclAvg` | 4 | `ReduceOp.AVG` |

### 2.5 PyNcclCommunicator 生命周期

**File:** `pynccl.py:29-127`

```python
# 1. 创建: 绑定到非NCCL ProcessGroup
self.nccl = NCCLLibrary(library_path)
self.unique_id = self.nccl.ncclGetUniqueId()  # rank 0
self.comm = self.nccl.ncclCommInitRank(world_size, unique_id, rank)

# 2. 默认禁用, 仅在CUDA Graph中使用
self.disabled = True
# 使用: with pynccl_comm.change_state(enable=True):
#     pynccl_comm.all_reduce(tensor)

# 3. Warmup: 小规模AllReduce
data = torch.zeros(1, device=device)
self.all_reduce(data)
```

---

## 3. Custom AllReduce — NVLink IPC 内核

### 3.1 算法选择

**File:** `sgl-kernel/csrc/allreduce/custom_all_reduce.cuh:32-36`

| 常量 | 值 | 含义 |
|------|-----|------|
| `kAllReduceGPUSmall` | 4 | ≤4 GPU: 单次算法 |
| `kAllReduceGPULarge` | 8 | ≤8 GPU: 双次算法 |
| `kAllReduceSmallThreshold` | 512KB | 小张量阈值 |
| `kAllReduceLargeThreshold` | 256KB | 大张量阈值 |

### 3.2 信号同步机制 (PTX 系统级原子)

**File:** `custom_all_reduce.cuh:153-186`

```cuda
// Release语义 (sm_70+)
asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));

// Acquire语义 (sm_70+)
asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));

// 旧架构fallback (sm_70以下)
asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
```

### 3.3 多GPU屏障同步

**File:** `custom_all_reduce.cuh:192-218`

```cuda
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
    // 双缓冲peer_counter避免竞态
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
        st_flag_release(peer_counter_ptr, val);
        while (ld_flag_acquire(self_counter_ptr) != val);
    } else {
        st_flag_volatile(peer_counter_ptr, val);
        while (ld_flag_volatile(self_counter_ptr) != val);
    }
}
```

### 3.4 CUDA初始化

**File:** `custom_all_reduce.cu:14-27`

```cpp
fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, 
                       torch::Tensor& rank_data, int64_t rank, bool full_nvlink) {
    // 最多8 GPU, 必须偶数
    if (world_size > 8) throw "world size > 8 is not supported";
    if (world_size % 2 != 0) throw "Odd num gpus is not supported";
    return new CustomAllreduce(ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), 
                               rank, world_size, full_nvlink);
}
```

---

## 4. Quick AllReduce — AMD MI300 专用

### 4.1 AMD 特有 PTX (实际是 AMD GCN 汇编)

**File:** `sgl-kernel/csrc/allreduce/quick_all_reduce_base.h`

| 指令 | 行号 | 含义 |
|------|------|------|
| `s_setreg_imm32_b32 0xdc1, 1` | 90 | 开启FP16溢出模式 (gfx942) |
| `s_setreg_imm32_b32 0xdc1, 0` | 92 | 关闭FP16溢出模式 |
| `v_pk_add_f16` | 109-112 | 打包FP16加法 (8路并行) |
| `v_pk_max_f16` | 131 | 打包FP16最大值 |
| `v_pk_min_f16` | 150 | 打包FP16最小值 |
| `v_pk_add_i16` | 209 | 打包INT16加法 |
| `v_pk_fma_f16` | 221 | 打包FP16 FMA (用于sub) |
| `v_pk_mul_f16` | 240 | 打包FP16乘法 |

### 4.2 CDNA架构内存语义

```cpp
#if defined(__gfx942__)          // CDNA3 (MI300)
#define MUBUF_ACQUIRE 16         // scope bits sc0, sc1
#define MUBUF_RELEASE 16
#elif defined(__gfx908__) || defined(__gfx90a__)  // CDNA1/CDNA2
#define MUBUF_ACQUIRE 1          // glc bit
#define MUBUF_RELEASE 0
#endif
```

### 4.3 Buffer Resource 描述符

```cpp
union BufferResource {
    int32x4_t descriptor;
    struct {
        void* address;     // 48b地址 + 16b stride
        uint32_t range;    // 缓冲区字节范围
        uint32_t config;   // 0x00020000U (DFMT=32b)
    };
};
```

---

## 5. MSCCL++ — 微软单边通信

### 5.1 通信模式

**File:** `sgl-kernel/csrc/allreduce/mscclpp_allreduce.cuh`

| 模式 | 函数 | 用途 |
|------|------|------|
| 单节点 | `allreduce_LL_1node` | NVLink直连，LLPacket协议 |
| 双节点 | `allreduce_LL_2node` | NVLink + IB, PortChannel跨节点 |

### 5.2 单节点 AllReduce 算法 (3步)

```cuda
// Step 1: 写入scratch缓冲区 (putPackets)
memChan.putPackets(scratchOffset, srcOffset, nelemsPerRank*sizeof(int), tid, ...);

// Step 2: 读取+规约+写回 (read + add_vectors + write)
for (int index = 0; index < nPeers; index++) {
    uint2 val = dstPkt[idx].read(flag);
    data = add_vectors<TYPE>(val, data);
}
data = add_vectors<TYPE>(data, src[idx]);
// 写回所有peer的result scratch
for (int index = 0; index < nPeers; index++) {
    memChans[index].write(offset, packet);
}

// Step 3: 从result scratch读取最终结果
uint2 data = dstPkt[idx + dstOffset].read(flag);
```

### 5.3 双节点扩展

增加 `PortChannel` (InfiniBand) 进行跨节点 reduce-scatter + all-gather：
- Step 1: 本地 NVLink putPackets
- Step 2: 本地 reduce-scatter
- Step 3: IB PortChannel 发送到远端节点
- Step 4: 远端 all-gather 回传
- Step 5: 本地写入最终结果

### 5.4 NCCL环境变量 (测试中)

**File:** `test_mscclpp_allreduce.cu:20-21`

```
NCCL_RUNTIME_CONNECT=0
NCCL_IB_GID_INDEX=3
NCCL_DEBUG=WARN
```

---

## 6. TorchSymmMem — PyTorch 对称内存

**File:** `python/sglang/srt/distributed/device_communicators/torch_symm_mem.py:32-135`

| 方法 | 行号 | 用途 |
|------|------|------|
| `__init__` | 54 | 初始化对称内存通信器 |
| `should_torch_symm_mem_allreduce` | 112 | 判断是否使用对称内存AR |
| `all_reduce` | 135 | 执行对称内存AllReduce |

---

## 7. ProcessGroup 创建与 NCCL 初始化

### 7.1 分布式后端选择

**File:** `parallel_state.py:269-316`

| 平台 | 设备后端 | CPU后端 |
|------|---------|--------|
| NVIDIA | `nccl` | `gloo` |
| AMD | `nccl` (RCCL) | `gloo` |
| MUSA | `nccl` (MCCL) | `gloo` |
| Mooncake | `mooncake` | `mooncake-cpu` |

### 7.2 通信器初始化顺序

```python
# parallel_state.py:368-444
1. PyNcclCommunicator(group=cpu_group, device=device)     # if use_pynccl
2. PyMscclppCommunicator(group=cpu_group, device=device)  # if use_pymscclpp
3. dispatch_custom_allreduce(group=cpu_group, device=device)  # if use_custom_allreduce
4. QuickAllReduce(group=cpu_group, device=device)          # if is_hip() and qr_rocm_arch_available()
5. TorchSymmMemCommunicator(group=cpu_group, device=device) # if use_torch_symm_mem
6. HpuCommunicator(group=device_group)                     # if use_hpu
7. XpuCommunicator(group=device_group)                     # if use_xpu
8. NpuCommunicator(group=device_group)                     # if use_npu
```

### 7.3 AllReduce 调度决策树

```python
# parallel_state.py:562-639
def all_reduce(self, input_):
    if world_size == 1: return input_
    if input_.is_cpu:
        if shm_available: shm_allreduce(input_)
        else: torch.distributed.all_reduce(input_)
    if hpu_communicator: return hpu.all_reduce(input_)
    if xpu_communicator: return xpu.all_reduce(input_)
    if npu_communicator: return npu.all_reduce(input_)
    if pynccl and symmetric_memory: pynccl.all_reduce(input_)  # in-place
    # 以下为outplace决策:
    if ca_comm and ca_comm.should_custom_ar(input_):     method = "ca"
    elif qr_comm and qr_comm.should_quick_allreduce():    method = "qr"
    elif pymscclpp and pymscclpp.should_mscclpp():       method = "pymscclpp"
    elif torch_symm_mem and should_torch_symm_mem():     method = "torch_symm_mem"
    elif piecewise_cuda_graph and pynccl:                method = "pynccl"
    if method: return outplace_all_reduce(input_, method)
    else: inplace_all_reduce(input_)  # fallback to torch.distributed
```

---

## 8. NCCL 环境变量汇总

| 变量 | 来源 | 用途 |
|------|------|------|
| `SGLANG_NCCL_SO_PATH` | `pynccl_wrapper.py:48` | 指定NCCL库路径 |
| `NCCL_CUMEM_ENABLE` | `test_update_weights_from_distributed.py:184` | 禁用CUMEM (避免意外行为) |
| `NCCL_NVLS_ENABLE` | `test_update_weights_from_distributed.py:185` | 禁用NVLS (避免意外行为) |
| `NCCL_RUNTIME_CONNECT` | `test_mscclpp_allreduce.cu:20` | MSCCL++测试: 延迟连接 |
| `NCCL_IB_GID_INDEX` | `test_mscclpp_allreduce.cu:21` | IB GID索引 |
| `NCCL_DEBUG` | `test_mscclpp_allreduce.cu:21` | NCCL调试级别 |

---

## 9. 张量并行通信

### 9.1 AllReduce (TP)

**File:** `parallel_state.py:562-639`

张量并行的核心通信是 AllReduce，由 `GroupCoordinator.all_reduce()` 统一调度。

### 9.2 AllGather / ReduceScatter

**File:** `parallel_state.py:177-196`

```python
@register_custom_op(mutates_args=["output"])
def reg_all_gather_into_tensor(output, input, group_name):
    group._all_gather_into_tensor(output, input)

@register_custom_op(mutates_args=["output"])
def reg_reduce_scatter_tensor(output, input, group_name):
    group._reduce_scatter_tensor(output, input)
```

### 9.3 PyNccl AllGather 实现

**File:** `pynccl.py:181-223`

```python
def all_gather(self, output_tensor, input_tensor, sizes=None):
    if sizes is not None:
        # 不均匀切分: 使用 ncclGroupStart/End + ncclBroadcast
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            self.nccl.ncclBroadcast(...)
        self.nccl.ncclGroupEnd()
    else:
        # 均匀切分: 直接 ncclAllGather
        self.nccl.ncclAllGather(...)
```

### 9.4 PyNccl ReduceScatter 实现

**File:** `pynccl.py:252-297`

```python
def reduce_scatter(self, output_tensor, input_tensor, op=SUM, sizes=None):
    if sizes is not None:
        # 不均匀切分: ncclGroupStart/End + ncclReduce
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            self.nccl.ncclReduce(...)
        self.nccl.ncclGroupEnd()
    else:
        # 均匀切分: ncclReduceScatter
        self.nccl.ncclReduceScatter(...)
```

---

## 10. MoE Expert Parallel 通信

### 10.1 DeepEP AlltoAll

**File:** `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:33,769`

```python
from sglang.srt.utils import is_blackwell
# Blackwell上的特殊处理:
if is_blackwell():
    # 特殊通信优化路径
```

### 10.2 FlashInfer AllReduce Fusion

SGLang 支持在MoE层中将 AllReduce 融合到 FlashInfer 注意力内核中：

```bash
--enable-flashinfer-allreduce-fusion
```

### 10.3 AIter AllReduce Fusion (AMD)

```bash
--enable-aiter-allreduce-fusion
```

---

## 11. 上下文并行 (Context Parallelism)

### 11.1 CP AllGather (非阻塞)

**File:** `pynccl.py:225-250`

```python
def cp_all_gather_into_tensor(self, output, input, stream, sizes=None):
    """上下文并行: 使用pynccl非阻塞allgather"""
    self.nccl.ncclAllGather(
        buffer_type(input.data_ptr()),
        buffer_type(output.data_ptr()),
        input.numel(),
        ncclDataTypeEnum.from_torch(input.dtype),
        self.comm,
        cudaStream_t(stream.cuda_stream),
    )
```

---

## 12. 其他硬件通信器

| 通信器 | 文件 | 后端 |
|--------|------|------|
| `HpuCommunicator` | `hpu_communicator.py:12` | Intel Gaudi (HABANA) |
| `XpuCommunicator` | `xpu_communicator.py:12` | Intel XPU (oneAPI) |
| `NpuCommunicator` | `npu_communicator.py` | 华为 Ascend NPU (HCCL) |
| `MessageQueue` | `shm_broadcast.py:171` | CPU共享内存广播 |

### NpuCommunicator 特殊方法

```python
# parallel_state.py:641-653
def quant_all_reduce(self, input_):
    """NPU专用: 量化AllReduce"""
    if self.npu_communicator is not None:
        return self.npu_communicator.quant_all_reduce(input_)
```

---

## 13. CUDA Graph 与 NCCL 交互

### 13.1 Graph 捕获上下文

**File:** `parallel_state.py:499-559`

```python
@contextmanager
def graph_capture(self, graph_capture_context=None, stream=None):
    with self.device_module.stream(stream), maybe_ca_context:
        with maybe_pynccl_context, maybe_pymscclpp_context:
            yield graph_capture_context
```

### 13.2 分段CUDA Graph (Piecewise CUDA Graph)

**File:** `parallel_state.py:628-630`

```python
elif is_in_piecewise_cuda_graph() and self.pynccl_comm is not None:
    # 分段Graph: 使用pynccl outplace allreduce
    outplace_all_reduce_method = "pynccl"
```

---

## 14. 通信与计算重叠

### 14.1 SBO (Single Batch Overlap)

**File:** `python/sglang/srt/batch_overlap/single_batch_overlap.py:99,118`

```python
communicate_num_sms = 32 if is_blackwell() else 3
if is_blackwell():
    # Blackwell: 32 SM用于通信
```

### 14.2 SM分区 (Spatial Multiplexing)

**File:** `python/sglang/srt/multiplex/pdmux_context.py:55-76`

```python
def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    # 不同架构的SM分区约束
def divide_sm(total_sms, compute_capability, groups):
    # 按架构约束分配SM给通信/计算
```

---

## 15. 对比: SGLang vs vLLM NCCL 实现

| 维度 | SGLang | vLLM |
|------|--------|------|
| PyNccl封装 | 自有`pynccl_wrapper.py` + `pynccl.py` | 类似 |
| Custom AR | 自有内核 (从vLLM v0.8.2适配) | 原创实现 |
| Quick AR | 自有AMD实现 (quickreduce) | 无 |
| MSCCL++ | 自有内核 + Python封装 | 无 |
| TorchSymmMem | 有 | 有 |
| 分段Graph | 支持pynccl outplace | 支持 |
| 量化AllReduce | NPU专用 | 无 |
| AllReduce Fusion | FlashInfer + AIter | FlashInfer |
| Mooncake后端 | 有 | 有 |
| 硬件后端数 | 6 (CUDA/ROCm/MUSA/HPU/XPU/NPU) | 4 |
