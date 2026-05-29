# Part 4: Triton Kernel SM Configurations in vLLM

Deep-mined from `C:\Users\Administrator\Desktop\nccl\vllm`

---

## 1. Summary of Triton Kernel Categories

| Category | File Count | SM-Sensitive | Key Files |
|----------|-----------|-------------|-----------|
| Attention Ops | ~15 | Yes | `vllm/v1/attention/ops/triton_*.py` |
| Sampling Ops | ~10 | Yes | `vllm/v1/worker/gpu/sample/*.py` |
| Speculative Decoding | ~8 | No | `vllm/v1/spec_decode/*.py`, `vllm/v1/worker/gpu/spec_decode/*.py` |
| Input Batch / Buffer | ~6 | No | `vllm/v1/worker/gpu/input_batch.py`, `buffer_utils.py` |
| Block Table | ~3 | No | `vllm/v1/worker/gpu/block_table.py` |
| Mamba Utils | ~2 | No | `vllm/v1/worker/mamba_utils.py` |
| TopK/TopP | ~2 | No | `vllm/v1/sample/ops/topk_topp_triton.py` |
| Rejection Sampler | ~4 | No | `vllm/v1/sample/rejection_sampler.py` |
| Metrics / Logits | ~2 | No | `vllm/v1/worker/gpu/metrics/logits.py` |
| Rope | ~1 | No | `vllm/v1/worker/gpu/mm/rope.py` |
| MLA Sparse | ~1 | Yes | `vllm/v1/attention/ops/xpu_mla_sparse.py` |

Total: **~505 `@triton.jit` decorated kernels** found across vLLM.

---

## 2. SM-Sensitive Triton Kernels

### 2.1 TurboQuant Attention (FP8 KV Cache)

**File:** `vllm/v1/attention/ops/triton_turboquant_decode.py`

| Line | Code | SM Context |
|------|------|-----------|
| 26 | `Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9)` | FP8 format selection by SM |
| 43 | `@triton.jit` kernel `_turboquant_decode_kernel` | Main decode kernel |
| 85 | `0 = e4nv (Hopper+)` | Hopper uses native e4nv format |
| 321 | `@triton.jit` kernel | Another TQ decode kernel |

**File:** `vllm/v1/attention/ops/triton_turboquant_store.py`

| Line | Code | SM Context |
|------|------|-----------|
| 25 | `@triton.jit` | Store kernel |
| 144 | `@triton.jit` | Store kernel variant |
| 167 | `1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)` | FP8 format dispatch by SM |
| 220 | `@triton.jit` | Merge kernel |

### 2.2 Reshape and Cache Flash (FP8 KV)

**File:** `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

| Line | Code | SM Context |
|------|------|-----------|
| 8 | `get_fp8_min_max` | FP8 quantization bounds |
| 14 | `FP8_MIN, FP8_MAX = get_fp8_min_max()` | FP8 range constants |
| 16 | `_NATIVE_KV_CACHE_DTYPES` | Supported dtype strings |
| 102-113 | `key_load.dtype.is_fp8()` | FP8 path in kernel |
| 250 | `torch.int8: (127.0, -128.0)` | INT8 quantization bounds |
| 335 | `kv_cache_dtype: str` | KV cache dtype parameter |
| 365 | `current_platform.fp8_dtype()` | Platform-specific FP8 dtype |
| 384-385 | `torch.float8_e4m3fn, torch.float8_e5m2` | FP8 dtype variants |

### 2.3 Decode Attention (FP8 KV)

**File:** `vllm/v1/attention/ops/triton_decode_attention.py`

| Line | Code | SM Context |
|------|------|-----------|
| 53 | `@triton.jit` | Main decode attention kernel |
| 59 | `@triton.jit` | Helper kernel |
| 137 | `k.dtype.is_fp8()` | FP8 key handling |
| 157 | `v.dtype.is_fp8()` | FP8 value handling |
| 261 | `@triton.jit` | Full decode kernel |
| 371, 382, 401 | `is_fp8()` checks | Multiple FP8 paths |
| 492 | `triton.cdiv(head_num, min(BLOCK_H, kv_group_num))` | Block size cdiv |
| 548 | `@triton.jit` | Another decode kernel |

### 2.4 Prefill Attention

**File:** `vllm/v1/attention/ops/triton_prefill_attention.py`

| Line | Code | SM Context |
|------|------|-----------|
| 36 | `@triton.jit` | Prefill attention kernel |
| 221 | `grid = (batch, head, triton.cdiv(max_input_len, BLOCK))` | Grid configuration |

### 2.5 Unified Attention Helpers

**File:** `vllm/v1/attention/ops/triton_attention_helpers.py`

| Line | Code | SM Context |
|------|------|-----------|
| 3 | `Shared @triton.jit helpers for unified attention` | Shared helpers |
| 21-321 | Multiple `@triton.jit` functions | 11 helper kernels total |

### 2.6 MLA Sparse

**File:** `vllm/v1/attention/ops/xpu_mla_sparse.py`

| Line | Code | SM Context |
|------|------|-----------|
| 9 | `@triton.jit` | MLA sparse kernel |
| 215 | `triton.cdiv(num_heads_q, min(BLOCK_H, kv_group_num))` | Block configuration |

### 2.7 Merge Attention States

**File:** `vllm/v1/attention/ops/triton_merge_attn_states.py`

| Line | Code | SM Context |
|------|------|-----------|
| 9 | `float8_info = torch.finfo(current_platform.fp8_dtype())` | Platform FP8 dtype |
| 57 | `@triton.jit` | Merge kernel with FP8 constants |

---

## 3. Sampling Triton Kernels

### 3.1 Gumbel Sampling

**File:** `vllm/v1/worker/gpu/sample/gumbel.py`

| Line | Code | SM Context |
|------|------|-----------|
| 10 | `Triton requires globals accessed from @triton.jit` | Triton global access note |
| 19 | `@triton.jit` | Gumbel sample kernel |
| 52 | `num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)` | Block grid |
| 63 | `@triton.jit` | Another Gumbel kernel |
| 78 | `@triton.jit` | Gumbel helper |
| 119 | `on H100/Ada/Blackwell` | SM-specific comment |
| 142 | `@triton.jit` | Gumbel final kernel |
| 207 | `num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)` | Block grid |

### 3.2 Penalties

**File:** `vllm/v1/worker/gpu/sample/penalties.py`

| Line | Code | SM Context |
|------|------|-----------|
| 106 | `@triton.jit` | Penalty kernel |
| 199 | `num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)` | Block grid |
| 218 | `@triton.jit` | Another penalty kernel |
| 285 | `num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)` | Block grid |

### 3.3 Min-P, Logprob, Logit Bias, Bad Words

| File | Line | Kernel |
|------|------|--------|
| `vllm/v1/worker/gpu/sample/min_p.py` | 8 | `@triton.jit` min-p filtering |
| `vllm/v1/worker/gpu/sample/logprob.py` | 13, 55, 173 | 3 `@triton.jit` logprob kernels |
| `vllm/v1/worker/gpu/sample/logit_bias.py` | 147 | `@triton.jit` logit bias kernel |
| `vllm/v1/worker/gpu/sample/bad_words.py` | 100 | `@triton.jit` bad words kernel |
| `vllm/v1/worker/gpu/sample/prompt_logprob.py` | 154 | `@triton.jit` prompt logprob |

---

## 4. Speculative Decoding Triton Kernels

### 4.1 Rejection Sampler

**File:** `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`

| Line | Kernel |
|------|--------|
| 9 | `@triton.jit` - sampler util 1 |
| 20 | `@triton.jit` - sampler util 2 |
| 47 | `@triton.jit` - sampler util 3 |
| 156 | `@triton.jit` - sampler util 4 |
| 306 | `@triton.jit` - sampler util 5 |
| 533 | `triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)` |
| 620 | `triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)` |

**File:** `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`

| Line | Kernel |
|------|--------|
| 20 | `@triton.jit` - rejection sampler |

### 4.2 EAGLE Speculator

**File:** `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`

| Line | Kernel |
|------|--------|
| 601 | `@triton.jit` |
| 723 | `@triton.jit` |
| 796 | `@triton.jit` |

### 4.3 V1 Sample Rejection Sampler

**File:** `vllm/v1/sample/rejection_sampler.py`

| Line | Kernel |
|------|--------|
| 707 | `@triton.jit(do_not_specialize=["max_spec_len"])` |
| 761 | `@triton.jit(do_not_specialize=["max_spec_len"])` |
| 830 | `@triton.jit(do_not_specialize=["replace_from", "replace_to"])` |
| 853 | `@triton.jit` |

### 4.4 N-gram Proposer

**File:** `vllm/v1/spec_decode/ngram_proposer_gpu.py`

| Line | Code |
|------|------|
| 231 | `"triton.autotune_pointwise": True` |

---

## 5. Infrastructure Triton Kernels

### 5.1 Input Batch Management

**File:** `vllm/v1/worker/gpu/input_batch.py`

| Line | Kernel |
|------|--------|
| 161 | `@triton.jit` |
| 221 | `@triton.jit` |
| 279 | `@triton.jit` |
| 374 | `@triton.jit` |
| 423 | `@triton.jit` |
| 518 | `@triton.jit` |
| 550 | `@triton.jit` |

### 5.2 Block Table

**File:** `vllm/v1/worker/gpu/block_table.py`

| Line | Kernel |
|------|--------|
| 183 | `@triton.jit(do_not_specialize=["num_reqs"])` |
| 223 | `@triton.jit` |
| 288 | `@triton.jit` |

### 5.3 Buffer Utils

**File:** `vllm/v1/worker/gpu/buffer_utils.py`

| Line | Kernel |
|------|--------|
| 193 | `@triton.jit` |

### 5.4 Context Parallelism

**File:** `vllm/v1/worker/gpu/cp_utils.py`

| Line | Code |
|------|------|
| 22 | `num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)` |
| 35 | `@triton.jit` |

### 5.5 Worker Utils

**File:** `vllm/v1/worker/utils.py`

| Line | Kernel |
|------|--------|
| 40 | `@triton.jit` |

### 5.6 Mamba Utils

**File:** `vllm/v1/worker/mamba_utils.py`

| Line | Kernel |
|------|--------|
| 25 | `@triton.jit` |
| 176 | `@triton.jit` |

### 5.7 TopK/TopP

**File:** `vllm/v1/sample/ops/topk_topp_triton.py`

| Line | Kernel |
|------|--------|
| 70 | `@triton.jit` |
| 93 | `@triton.jit` |

### 5.8 Metrics

**File:** `vllm/v1/worker/gpu/metrics/logits.py`

| Line | Kernel |
|------|--------|
| 9 | `@triton.jit` |

### 5.9 Rope

**File:** `vllm/v1/worker/gpu/mm/rope.py`

| Line | Kernel |
|------|--------|
| 149 | `@triton.jit` |

### 5.10 Spec Decode Utils

**File:** `vllm/v1/spec_decode/utils.py`

| Line | Kernel |
|------|--------|
| 28 | `@triton.jit` |
| 136 | `@triton.jit` |
| 179 | `@triton.jit` |
| 308 | `@triton.jit` |
| 458 | `@triton.jit` |

### 5.11 Block Table (V1 worker)

**File:** `vllm/v1/worker/block_table.py`

| Line | Kernel |
|------|--------|
| 325 | `@triton.jit` |
