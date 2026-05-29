# TensorRT SM120 (Consumer Blackwell) FMHA Cubin 深度分析


## 1. 提取概述

| 项目 | 数量 |
|------|------|
| Cubin文件 | 32个 (v1=13, v2=19) |
| PTX源码 | 32个, 4.33 MB, 135,883行 |
| SASS反汇编 | 32个, 20.19 MB, 174,418行 |
| PTX版本 | **8.7** |
| Target | **sm_120** |
| 内部代号 | **gb20x** (Consumer Blackwell) |
| ELF架构 | 64-bit, Little-Endian, e_machine=0xBE |

## 2. Cubin ELF内部结构

每个cubin包含以下ELF段:

| 段名 | 类型 | 内容 |
|------|------|------|
| `.nv_debug_ptx_txt` | PROGBITS | **嵌入的PTX源码** (null分隔, 4.9 MB总计) |
| `.text.<kernel>` | PROGBITS | SASS机器码 (1.4 MB总计) |
| `.nv.info.<kernel>` | SHT_NV_INFO (0x70000000) | NV元信息 |
| `.nv.constant0.<kernel>` | PROGBITS | 常量缓存数据 (992 bytes/kernel) |
| `.nv.shared.<kernel>` | NOBITS | Shared memory布局 (1024 bytes) |
| `.nv.callgraph` | SHT_NV_CALLGRAPH | 调用图 |
| `.debug_line` | PROGBITS | DWARF行号信息 |
| `.debug_frame` | PROGBITS | DWARF帧信息 |
| `.debug_str` | PROGBITS | DWARF字符串 |
| `.nv_debug_line_sass` | PROGBITS | SASS行号映射 |
| `.note.nv.tkinfo` | NOTE | Token信息 |
| `.note.nv.cuver` | NOTE | CUDA版本 |

**关键发现: cubin内嵌了完整的PTX 8.7源码！** 通过提取`.nv_debug_ptx_txt`段并将null字节替换为换行符，获得可读的PTX文本。

## 3. SM120内核清单

### 3.1 FMHA v1 (13个cubin)

| 文件 | 精度 | 序列长度 | 内核数 | SASS大小 |
|------|------|---------|--------|---------|
| fp16_128_64 | FP16 | 128 | 2 (nl+main) | 62,976 |
| fp16_256_64 | FP16 | 256 | 2 | 47,616 |
| fp16_384_64 | FP16 | 384 | 2 | 45,568 |
| fp16_512_64 | FP16 | 512 | 2 | 47,104 |
| fp16_64_64 | FP16 | 64 | 2 | 22,656 |
| fp16_96_64 | FP16 | 96 | 1 | 31,744 |
| int8_128_64 | INT8 | 128 | 1 | 54,656 |
| int8_192_64 | INT8 | 192 | 2 | 41,472 |
| int8_256_64 | INT8 | 256 | 2 | 50,048 |
| int8_384_64 | INT8 | 384 | 2 | 40,832 |
| int8_512_64 | INT8 | 512 | 2 | 48,512 |
| int8_64_64 | INT8 | 64 | 1 | 18,560 |
| int8_96_64 | INT8 | 96 | 2 | 44,288 |

### 3.2 FMHA v2 (19个cubin)

| 文件 | 精度 | 序列长度 | 交织 | 内核数 | SASS大小 |
|------|------|---------|------|--------|---------|
| fp16_128/256/384/512/64/96_64 | FP16 | 64~512 | - | 1~2 | 20,352~53,120 |
| int8_128/192/256/384/512/64/96_64 | INT8 | 64~512 | - | 1~2 | 16,640~62,336 |
| il_int8_128/192/256/384/64/96_64 | INT8 | 64~512 | ✓ | 1~2 | 17,792~68,352 |

## 4. PTX 8.7 指令分析 (135,883行)

### 4.1 PTX Top-30 指令

| PTX指令 | 出现次数 | 说明 |
|---------|---------|------|
| `mul.ftz.f32` | 13,101 | FTZ浮点乘 (Softmax核心) |
| `mov.u32` | 9,253 | 32位移动 |
| `add.s32` | 7,989 | 32位整数加 |
| `selp.f32` | 7,720 | 浮点条件选择 (替代分支) |
| `add.ftz.f32` | 4,596 | FTZ浮点加 |
| `ex2.approx.ftz.f32` | 4,480 | 2^x近似 (exp计算) |
| `sub.ftz.f32` | 4,352 | FTZ浮点减 |
| `mov.b32` | 4,031 | 位级移动 |
| `setp.gt.ftz.f32` | 3,960 | 浮点大于比较 |
| `and.b32` | 2,946 | 按位与 |
| `cvt.rni.sat.s32.f32` | 2,856 | 浮点→INT8截断 |
| `cvt.rn.f32.s32` | 2,832 | 整数→浮点 |
| `add.s64` | 2,058 | 64位地址计算 |
| `mov.f32` | 1,672 | 浮点移动 |
| `setp.eq.ftz.f32` | 1,656 | 浮点等于比较 |
| `abs.ftz.f32` | 1,536 | 浮点绝对值 |
| `st.shared.v4.b32` | 1,520 | Shared memory 128-bit存储 |
| `shl.b32` | 1,457 | 左移 |
| `cvt.pack.sat.s8.s32.b32` | 1,428 | INT32→INT8打包 |
| `setp.eq.s32` | 1,295 | 整数等于比较 |
| `cvt.f32.f16` | 1,264 | FP16→FP32转换 |
| `prmt.b32` | 1,232 | 字节重排 |
| `or.b32` | 1,087 | 按位或 |
| `shfl.sync.bfly.b32` | 992 | Warp butterfly洗牌 |
| `ld.global.v4.u32` | 912 | Global 128-bit加载 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 902 | **矩阵加载 (Tensor Core)** |
| `ld.shared.v4.b32` | 808 | Shared 128-bit加载 |
| `xor.b32` | 806 | 按位异或 |
| `bar.sync` | 797 | Barrier同步 |

### 4.2 关键PTX指令说明

| 指令 | Blackwell特性 |
|------|--------------|
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 从shared memory加载8x8矩阵碎片到Tensor Core寄存器 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 转置矩阵加载 |
| `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` | 半矩阵转置加载 |
| `cvt.pack.sat.s8.s32.b32` | INT32打包为INT8 (量化核心) |
| `cvt.rn.f16x2.f32` | FP32→FP16x2打包转换 |
| `mul.f16x2` | FP16x2并行乘法 |
| `add.f16x2` | FP16x2并行加法 |
| `shfl.sync.bfly.b32` | Warp内butterfly洗牌 (AllReduce) |
| `shfl.sync.idx.b32` | Warp内索引洗牌 |
| `ex2.approx.ftz.f32` | 快速exp2近似 (Softmax) |
| `rcp.approx.ftz.f32` | 快速倒数近似 |

## 5. SASS指令分析 (174,418行, 99种opcodes)

### 5.1 SASS Top-30 指令

| SASS指令 | 出现次数 | 说明 |
|----------|---------|------|
| `FMUL.FTZ` | 13,072 | FTZ浮点乘 |
| `FADD.FTZ` | 8,044 | FTZ浮点加 |
| `FSEL` | 7,656 | 浮点条件选择 |
| `MUFU.EX2` | 4,480 | Special function: exp2 |
| `MOV` | 3,897 | 寄存器移动 |
| `FSETP.GT.FTZ.AND` | 3,896 | 浮点比较> |
| `I2FP.F32.S32` | 2,832 | 整数→浮点 |
| `IADD3` | 2,631 | 3输入整数加法 (Blackwell新增) |
| `LOP3.LUT` | 2,597 | 3输入查表逻辑 (Blackwell增强) |
| **`HMMA.16816.F16`** | **2,528** | **FP16 Tensor Core矩阵乘** |
| **`IMMA.16832.S8.S8`** | **2,284** | **INT8 Tensor Core矩阵乘** |
| `CS2R` | 1,874 | 特殊寄存器读取 (时钟/cycle) |
| `IADD.64` | 1,590 | 64位地址计算 |
| `FSETP.NEU.FTZ.AND` | 1,544 | 浮点不等比较 |
| **`F2IP.S8.F32.NTZ`** | **1,384** | **FP32→INT8量化 (SM120新指令)** |
| `HADD2.F32` | 1,264 | FP16x2加法(结果FP32) |
| `PRMT` | 1,246 | 字节重排 |
| `ISETP.GE.U32.AND` | 1,087 | 无符号整数比较≥ |
| `FFMA.FTZ` | 1,032 | FTZ浮点FMA |
| `SHFL.BFLY` | 992 | Warp butterfly洗牌 |

### 5.2 SM120 Blackwell独有SASS指令

| SASS指令 | 次数 | 说明 | Blackwell代际 |
|----------|------|------|--------------|
| **`F2IP.S8.F32.NTZ`** | 1,384 | FP32→INT8 量化转换 | **SM120新** |
| **`F2FP.F16.F32.PACK_AB`** | 632 | FP32→FP16 打包转换 | **SM120新** |
| **`HMMA.16816.F16`** | 2,528 | FP16 16x8x16 Tensor Core MMA | Blackwell (HMMA新格式) |
| **`IMMA.16832.S8.S8`** | 2,284 | INT8 16x8x32 Tensor Core MMA | Blackwell (IMMA新格式) |
| **`BAR.SYNC.DEFER_BLOCKING`** | 797 | 延迟屏障同步 | **Blackwell新** |
| **`LDCU` / `LDCU.64` / `LDCU.128`** | 516 | Uniform常量缓存加载 | **Blackwell新 (UR)*** |
| **`LDGSTS.E.BYPASS.128`** | 133 | Global→Shared 异步拷贝 (BYPASS) | **Blackwell新** |
| **`UIADD3`** | 158 | Uniform 3输入加法 | **Blackwell新 (UR)** |
| **`ULEA`** | 73 | Uniform 地址计算 | **Blackwell新 (UR)** |
| **`S2UR`** | 136 | Special寄存器→Uniform寄存器 | **Blackwell新** |
| **`R2UR`** | 9 | 通用寄存器→Uniform寄存器 | **Blackwell新** |
| **`UMOV`** | 63 | Uniform寄存器移动 | **Blackwell新** |
| **`UIMAD`** | 32 | Uniform 乘加 | **Blackwell新** |
| **`UISETP`** | 29 | Uniform 整数比较 | **Blackwell新** |
| **`USHF.L.U32`** | 81 | Uniform 左移 | **Blackwell新** |
| **`USHF.L.U64.HI`** | 65 | Uniform 64位左移 | **Blackwell新** |
| **`UVIMNMX`** | 8 | Uniform 向量最小最大 | **Blackwell新** |
| **`UPRMT`** | 15 | Uniform 字节重排 | **Blackwell新** |
| **`VIMNMX.S32/U32`** | 48 | 向量最小最大 | **Blackwell新** |
| **`PLOP3.LUT`** | 42 | 谓词3输入查表逻辑 | **Blackwell新** |
| **`LDGDEPBAR`** | 101 | 全局内存依赖屏障 | **Blackwell增强** |
| **`BSSY` / `BSYNC`** | 85 | Barrier设置/同步 | Blackwell增强 |
| **`SR_CgaCtaId`** | 69 | CGA内CTA标识 | **Blackwell CGA** |

> *UR = Uniform Register, Blackwell新增的Uniform数据路径寄存器，独立于主ALU，减少寄存器压力

## 6. 关键Blackwell架构特性详解

### 6.1 Uniform Register (UR) 数据路径

SM120 (Consumer Blackwell) 引入了**Uniform Register**子系统，这是SM100/SM90所没有的：

```
S2UR UR6, SR_CgaCtaId     ; 特殊寄存器→UR
R2UR UR4, R0               ; 通用寄存器→UR
UMOV UR5, 0x1000           ; UR立即数移动
UIADD3 UR4, UPT, UPT, UR6, 0x2000, URZ  ; UR 3输入加法
ULEA UR6, UR6, 0x400, 0x18 ; UR 地址计算
UIMAD UR5, UR6, UR4, URZ   ; UR 乘加
UISETP.GE.AND UP0, UPT, UR4, 0x100, UPT ; UR 整数比较
LDCU.64 UR8, c[0x0][0x358] ; UR 常量缓存加载
USHF.L.U32 UR4, UR6, 0x2   ; UR 左移
UVIMNMX.S32 UR5, UR6, 0x20, UPT ; UR 向量最小最大
UPRMT UR0, UR0, 0x0, UR1   ; UR 字节重排
```

**UR的优势**: 地址计算、循环边界、stride等uniform数据放入UR，释放通用寄存器(R0~R255)给计算密集型数据。

### 6.2 Descriptor-Based寻址 (desc[])

SM120大量使用**描述符寻址**替代传统立即数偏移：

```sass
LDG.E R120, desc[UR8][R4.64]         ; 描述符基址+偏移
LDG.E R121, desc[UR8][R4.64+0x4]     ; 带偏移的描述符寻址
LDGSTS.E.BYPASS.128 [R3], desc[UR8][R20.64], !P3  ; 异步拷贝+描述符
```

描述符(UR8/UR12/UR18)通过`LDCU.64`从常量缓存加载，包含base address+属性信息。**共1,417次desc[]使用**。

### 6.3 Tensor Core矩阵乘 (HMMA/IMMA)

**FP16 Tensor Core** (FMHA FP16内核):
```sass
HMMA.16816.F16 R18, R64, R68, RZ   ; 16x8x16 FP16 MMA, C=0初始化
```
- 格式: `HMMA.16816.F16 D, A, B, C`
- 16x8x16 = M=16, N=8, K=16
- 输入/输出均为FP16

**INT8 Tensor Core** (FMHA INT8内核):
```sass
IMMA.16832.S8.S8 R88, R140.ROW, R44.COL, RZ  ; 16x8x32 INT8 MMA
```
- 格式: `IMMA.16832.S8.S8 D, A.ROW, B.COL, C`
- 16x8x32 = M=16, N=8, K=32
- 输入S8, 输出S32
- A矩阵按ROW布局, B矩阵按COL布局

### 6.4 新增格式转换指令

**F2IP: FP32→INT8 量化** (1,384次)
```sass
F2IP.S8.F32.NTZ R20, R171, R20, RZ  ; FP32→S8 量化
```
- 这是SM120的**专用量化指令**
- 对应PTX: `cvt.rni.sat.s32.f32` + `cvt.pack.sat.s8.s32.b32`
- NTZ = Non-Terminating Zero舍入

**F2FP: FP32→FP16 打包** (632次)
```sass
F2FP.F16.F32.PACK_AB R68, R125, R84  ; 两个FP32→FP16x2打包
```
- PACK_AB: 将两个FP32值打包为一个FP16x2寄存器
- 对应PTX: `cvt.rn.f16x2.f32`

### 6.5 异步数据传输

**LDGSTS: Global→Shared 异步拷贝**
```sass
LDGSTS.E.BYPASS.128 [R3], desc[UR8][R20.64], !P3     ; 异步128字节拷贝
LDGSTS.E.BYPASS.128 [R245+0x1000], desc[UR12][R16.64], !P2  ; 带shared offset
```
- BYPASS: 绕过L1缓存 (直通shared memory)
- 128: 128字节传输宽度
- !P3: 条件谓词 (P3为假时执行)
- 对应Hopper的`cp.async`, Blackwell进一步优化

**LDGDEPBAR: 依赖屏障**
```sass
LDGDEPBAR ;  ; 等待LDGSTS完成
```

### 6.6 BAR.SYNC.DEFER_BLOCKING

```sass
BAR.SYNC.DEFER_BLOCKING 0x0  ; 延迟屏障同步
```
- **DEPER_BLOCKING**: Blackwell新屏障语义
- 与Hopper的`BAR.SYNC`不同: DEFER允许其他CTA继续执行
- 共797次使用, 是FMHA内核同步的核心原语

### 6.7 CGA (Cooperative Group Array)

```sass
S2UR UR6, SR_CgaCtaId        ; 读取CGA内CTA ID
ULEA UR6, UR6, 0x400, 0x18   ; 基于CGA ID计算地址偏移
```
- **SR_CgaCtaId**: 当前线程在CGA内的CTA编号
- 用于多CTA协作寻址shared memory

### 6.8 HADD2: FP16x2算术

```sass
HADD2.F32 R68, R125, R84    ; FP16x2加法, 结果提升为FP32
HMUL2 R68, R125, R84         ; FP16x2乘法
```
- `.F32`修饰: 结果以FP32精度输出
- 混合精度策略: FP16存储 + FP32计算

## 7. PTX→SASS指令映射

| PTX指令 | SASS指令 | 说明 |
|---------|----------|------|
| `mul.ftz.f32` | `FMUL.FTZ` | 浮点乘 |
| `add.ftz.f32` | `FADD.FTZ` | 浮点加 |
| `selp.f32` | `FSEL` | 条件选择 |
| `ex2.approx.ftz.f32` | `MUFU.EX2` | exp2 |
| `rcp.approx.ftz.f32` | `MUFU.RCP` | 倒数 |
| `cvt.rni.sat.s32.f32` | `I2FP.F32.S32` (反) | 整数↔浮点 |
| `cvt.pack.sat.s8.s32.b32` | `F2IP.S8.F32.NTZ` | INT8量化 |
| `cvt.rn.f16x2.f32` | `F2FP.F16.F32.PACK_AB` | FP16打包 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | `LDSM.16.M88.4` | 矩阵加载 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | `LDSM.16.MT88.4` | 转置矩阵加载 |
| `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` | `LDSM.16.MT88.2` | 半转置加载 |
| `shfl.sync.bfly.b32` | `SHFL.BFLY` | Warp洗牌 |
| `and.b32` / `or.b32` / `xor.b32` | `LOP3.LUT` | 3输入查表逻辑 |
| `bar.sync` | `BAR.SYNC.DEFER_BLOCKING` | 屏障同步(Blackwell升级) |
| `mul.f16x2` | `HMUL2` | FP16x2乘法 |
| `add.f16x2` | `HADD2` / `HADD2.F32` | FP16x2加法 |
| `cvt.f32.f16` | `HADD2.F32` (隐式) | FP16→FP32 |

## 8. Softmax计算流水线

从SASS指令序列可以重建FMHA的Softmax计算流水线:

```
1.  HMMA.16816.F16 / IMMA.16832.S8.S8    ; QK^T矩阵乘
2.  FADD.FTZ                               ; 累加注意力分数
3.  FMUL.FTZ                               ; 缩放 (1/sqrt(d))
4.  FSETP.GT.FTZ.AND                       ; 比较找最大值
5.  FSEL                                    ; 条件选择最大值
6.  FSUB.FTZ (隐含)                         ; 减去最大值 (数值稳定)
7.  MUFU.EX2                                ; exp2(x - max)
8.  FADD.FTZ                               ; 求和 (softmax分母)
9.  MUFU.RCP                                ; 1/sum
10. FMUL.FTZ                               ; softmax权重
11. HMMA.16816.F16 / IMMA.16832.S8.S8      ; Attn*V矩阵乘
12. F2IP.S8.F32.NTZ (INT8) / F2FP.F16.F32.PACK_AB (FP16)  ; 量化/打包输出
```

## 9. 与SGLang SM120实现的对比

| 特性 | TensorRT SM120 | SGLang SM120 |
|------|---------------|-------------|
| 内核形式 | 预编译cubin | JIT CUTLASS模板 |
| Tensor Core | HMMA.16816/IMMA.16832 | CUTLASS Block-Scaled (NVFP4) |
| Uniform Register | ✓ (LDCU/UIADD3/ULEA/S2UR) | ✗ (CUTLASS抽象层隐藏) |
| 描述符寻址 | ✓ desc[] (1,417次) | ✗ |
| BAR.DEFER | ✓ (797次) | ✗ |
| LDGSTS.BYPASS | ✓ (133次) | ✗ |
| F2IP量化 | ✓ (1,384次) | ✗ (NVFP4不同量化) |
| PTX版本 | 8.7 | N/A (JIT) |
| CGA | ✓ (SR_CgaCtaId) | ✓ (griddepcontrol) |
| 目标应用 | FMHA (BERT注意力) | GEMM (LLM推理) |

**TensorRT的SM120内核比SGLang更底层**: 直接使用UR/描述符/DEER/异步拷贝等Blackwell硬件特性,而SGLang通过CUTLASS抽象层间接使用。

## 10. 文件清单

### 提取工具
- `extract_sm120_cubins.py` — 从C++字节数组提取ELF二进制
- `extract_ptx_sass.py` — 从cubin提取PTX文本+SASS反汇编

### 输出目录结构
```
ncxl/sm120_cubins/
├── manifest.txt                              — 32个cubin清单
├── *.cubin                                   — 32个原始ELF二进制 (7.99 MB)
├── ptx/                                      — 32个PTX源码文件 (4.33 MB)
│   ├── fused_multihead_attention_fp16_128_64_kernel.sm120.ptx
│   ├── fused_multihead_attention_v2_int8_384_64_kernel.sm120.ptx
│   └── ...
└── sass/                                     — 32个SASS反汇编文件 (20.19 MB)
    ├── fused_multihead_attention_fp16_128_64_kernel.sm120.sass
    ├── fused_multihead_attention_v2_int8_384_64_kernel.sm120.sass
    └── ...
```

### 文件大小Top-5

| Cubin | ELF大小 | PTX行数 | SASS行数 |
|-------|---------|---------|---------|
| v2_il_int8_384_64 | 353.3 KB | 6,047 | 8,558 |
| v1_fp16_128_64 | 380.6 KB | 6,867 | 7,886 |
| v2_il_int8_128_64 | 327.2 KB | 5,257 | 7,630 |
| v2_int8_384_64 | 335.8 KB | 5,759 | 7,806 |
| v1_int8_128_64 | 354.9 KB | 5,360 | 6,841 |

## 11. 核心结论

1. **SM120 cubin内嵌了PTX 8.7源码**: NVIDIA编译器在cubin中保留了debug PTX，通过`.nv_debug_ptx_txt`段可以完整提取
2. **Uniform Register是SM120最大的架构创新**: 15种UR指令 (S2UR/R2UR/UMOV/UIADD3/ULEA/UIMAD/UISETP/USHF/UVIMNMX/UPRMT/LDCU等)
3. **Descriptor寻址(desc[])**: 1,417次使用，是SM120内存访问的新范式
4. **F2IP.S8.F32.NTZ**: SM120专用的FP32→INT8量化指令，1,384次
5. **BAR.SYNC.DEFER_BLOCKING**: 797次，Blackwell独有的延迟屏障
6. **Tensor Core格式**: HMMA.16816.F16 (FP16) 和 IMMA.16832.S8.S8 (INT8)
7. **LDGSTS.E.BYPASS.128**: Global→Shared异步直通拷贝，133次
8. **CGA支持**: SR_CgaCtaId (69次)，多CTA协作计算
