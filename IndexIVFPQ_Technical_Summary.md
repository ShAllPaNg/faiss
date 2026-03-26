# IndexIVFPQ 技术总结

> 基于 Faiss 源码分析的 IndexIVFPQ 向量检索技术详解

---

## 目录

1. [IndexIVFPQ 检索时的代码位置](#1-indexivfpq-检索时的代码位置)
2. [IndexIVFPQ 四大核心流程](#2-indexivfpq-四大核心流程)
3. [一级二级量化器详解](#3-一级二级量化器详解)
4. [Faiss 量化器家族](#4-faiss-量化器家族)
5. [SPTAG vs Faiss IndexIVFPQ](#5-sptag-vs-faiss-indexivfpq)

---

## 1. IndexIVFPQ 检索时的代码位置

### 1.1 读盘 (IO 操作)

| 文件 | 函数/位置 | 说明 |
|------|----------|------|
| `faiss/index_io.h:75-81` | `read_index()` | 从文件/IO读取索引的入口 |
| `faiss/index_io.cpp` | 具体实现 | 反序列化 IndexIVFPQ 结构 |
| `faiss/index_io.h:113-117` | `read_InvertedLists()` | 读取倒排列表数据 |

### 1.2 PQ 距离计算

| 文件 | 函数/位置 | 说明 |
|------|----------|------|
| `IndexIVFPQ.cpp:651-717` | `precompute_list_tables_L2()` | **预计算L2距离表**，核心逻辑 |
| `IndexIVFPQ.cpp:861-932` | `scan_list_with_table()` | 使用预计算表扫描编码，计算距离 |
| `utils/pq_code_distance.h:63-82` | `distance_single_code()` | **单个PQ编码距离计算**（标量版） |
| `utils/pq_code_distance.h:84-116` | `distance_four_codes()` | **四个PQ编码并行计算**（SIMD优化） |
| `IndexIVFPQ.cpp:654-656` | `compute_distance_table()` | PQ量化器计算残差距离表 |

### 1.3 排序 (堆维护)

| 文件 | 函数/位置 | 说明 |
|------|----------|------|
| `impl/ResultHandler.h:283-287` | `HeapHandler::add_result()` | **堆排序入口**，维护top-k |
| `impl/ResultHandler.h:127-129` | `add_result()` | 通用添加结果接口 |
| `utils/Heap.h:84-138` | `heap_push()` | **堆插入核心实现** |
| `utils/Heap.h:47-81` | `heap_pop()` | 堆弹出 |
| `utils/Heap.h:161-168` | `minheap_push()` | 最小堆封装（L2距离） |

---

## 2. IndexIVFPQ 四大核心流程

### 2.1 构建索引流程

```
train() ──────────────────────────────────────────────────→ add()
    │                                                     │
    ├─→ 训练一级量化器 (IndexIVF.cpp:1212)              ├─→ quantizer->assign() 分配桶
    │   └─→ K-means聚类得到 nlist 个coarse centroid         │
    │                                                     ├─→ encode_vectors() PQ编码
    ├─→ 计算残差向量 (IndexIVF.cpp:1236)                   │   └─→ pq.compute_codes()
    │   residual = x - centroid                           │
    ├─→ 训练PQ量化器 (IndexIVFPQ.cpp:69)                   └─→ invlists->add_entry() 存入倒排列表
    │   └─→ 对残差向量做PQ聚类，得到 M×ksub 个centroid
    │
    └─→ precompute_table() (IndexIVFPQ.cpp:477)
        └─→ 重建预计算表以加速搜索
```

**关键代码位置:**

| 阶段 | 文件 | 函数 | 行号 |
|------|------|------|------|
| 训练入口 | `IndexIVF.cpp:1212` | `IndexIVF::train()` | 训练一级+二级量化器 |
| 训练PQ | `IndexIVFPQ.cpp:69` | `train_encoder()` | 对残差训练PQ |
| 计算残差 | `IndexIVF.cpp:1236` | `compute_residual_n()` | x - centroid |
| 添加向量 | `IndexIVF.cpp:190` | `add_with_ids()` | 分配桶+编码 |
| PQ编码 | `IndexIVFPQ.cpp:167` | `encode_vectors()` | 残差PQ编码 |
| 存入倒排 | `IndexIVF.cpp:248` | `add_core()` | 写入invlists |

**核心代码片段:**

```cpp
// IndexIVF.cpp:1212 - 训练流程
void IndexIVF::train(idx_t n, const float* x) {
    train_q1(n, x, verbose, metric_type);  // 训练一级量化器(IVF)

    if (by_residual) {
        quantizer->assign(n, tv.x, assign.data());
        quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());
        train_encoder(n, residuals.data(), assign.data());  // 训练PQ
    }
    is_trained = true;
}

// IndexIVFPQ.cpp:167 - 编码向量
void IndexIVFPQ::encode_vectors(idx_t n, const float* x,
                                const idx_t* list_nos, uint8_t* codes) {
    if (by_residual) {
        auto to_encode = compute_residuals(quantizer, n, x, list_nos);
        pq.compute_codes(to_encode.get(), codes, n);  // PQ编码
    }
}
```

### 2.2 保存索引到磁盘流程

```
write_index(index, file)
    │
    ├─→ 写入fourcc标识 ("IwPQ")
    ├─→ write_ivf_header()
    │   ├─→ write_index_header()  (d, ntotal, is_trained, metric)
    │   ├─→ nlist, nprobe
    │   ├─→ write_index(quantizer)  递归保存一级量化器
    │   └─→ write_direct_map()
    │
    ├─→ by_residual, code_size
    ├─→ write_ProductQuantizer(&pq)  保存PQ参数
    │
    └─→ write_InvertedLists(invlists)  保存倒排列表
        └─→ 逐个保存每个list的codes和ids
```

**关键代码位置:**

| 阶段 | 文件 | 函数 | 行号 |
|------|------|------|------|
| 入口 | `index_write.cpp:454` | `write_index()` | 分发到具体类型 |
| IVFPQ序列化 | `index_write.cpp:777` | `IndexIVFPQ分支` | 写入"IwPQ" |
| IVF头部 | `index_write.cpp:444` | `write_ivf_header()` | 写入通用IVF字段 |
| PQ序列化 | `index_write.cpp:785` | `write_ProductQuantizer()` | 写入PQ centroids |
| 倒排列表 | `index_write.cpp:786` | `write_InvertedLists()` | 写入编码数据 |

### 2.3 从磁盘加载索引流程

```
read_index(file)
    │
    ├─→ 读取fourcc识别类型 ("IwPQ"/"IvPQ")
    │
    └─→ read_ivfpq()
        ├─→ 创建 IndexIVFPQ 对象
        ├─→ read_ivf_header()
        │   ├─→ 读取 d, ntotal, is_trained, metric
        │   ├─→ 读取 nlist, nprobe
        │   └─→ read_index(quantizer)  递归加载一级量化器
        │
        ├─→ 读取 by_residual, code_size
        ├─→ read_ProductQuantizer(&pq)  加载PQ参数
        │
        ├─→ read_InvertedLists()  加载倒排列表
        │
        └─→ 如果is_trained && by_residual:
            └─→ precompute_table()  重建预计算表
```

**关键代码位置:**

| 阶段 | 文件 | 函数 | 行号 |
|------|------|------|------|
| 入口 | `index_read.cpp:1121` | `read_index_up()` | 读取fourcc分发 |
| IVFPQ加载 | `index_read.cpp:1072` | `read_ivfpq()` | 具体加载逻辑 |
| IVF头部 | `index_read.cpp:1088` | `read_ivf_header()` | 读取通用IVF字段 |
| PQ加载 | `index_read.cpp:1091` | `read_ProductQuantizer()` | 加载PQ centroids |
| 倒排列表 | `index_read.cpp:1098` | `read_InvertedLists()` | 加载编码数据 |
| 重建预计算 | `index_read.cpp:1107` | `precompute_table()` | 搜索优化 |

### 2.4 检索Query向量流程

```
search(n, x, k, distances, labels)
    │
    ├─→ quantizer->search()  粗量化
    │   └─→ 找到每个query最近的 nprobe 个coarse centroid
    │       返回: keys[n×nprobe], coarse_dis[n×nprobe]
    │
    ├─→ invlists->prefetch_lists()  预取倒排列表
    │
    └─→ search_preassigned()
        │
        ├─→ 对每个query i:
        │   ├─→ 初始化大小为k的堆 (保存top-k结果)
        │   │
        │   └─→ 对每个探测的桶 j (0..nprobe-1):
        │       │
        │       ├─→ get_InvertedListScanner()  获取扫描器
        │       │   └─→ 根据 nbits 选择 PQCodeDist 模板
        │       │
        │       ├─→ precompute_list_tables_L2()
        │       │   └─→ 预计算: dis0 + sim_table[M×ksub]
        │       │
        │       ├─→ scan_list_with_table()  扫描编码
        │       │   ├─→ PQCodeDist::distance_single_code()
        │       │   │   └─→ 查表计算PQ距离
        │       │   │       dis = dis0 + Σ tab[m][code[m]]
        │       │   │
        │       │   └─→ res.add(j, dis)  添加到结果
        │       │       └─→ heap_push()  维护top-k堆
        │       │
        │       └─→ 继续下一个桶...
        │
        └─→ 输出: distances[n×k], labels[n×k]
```

**关键代码位置:**

| 阶段 | 文件 | 函数 | 行号 |
|------|------|------|------|
| 搜索入口 | `IndexIVF.cpp:304` | `search()` | 主搜索函数 |
| 粗量化 | `IndexIVF.cpp:333` | `quantizer->search()` | 找nprobe个候选桶 |
| 精细搜索 | `IndexIVF.cpp:401` | `search_preassigned()` | 扫描候选桶 |
| 获取扫描器 | `IndexIVFPQ.cpp:1256` | `get_InvertedListScanner()` | 返回扫描器对象 |
| 预计算表 | `IndexIVFPQ.cpp:651` | `precompute_list_tables_L2()` | 计算距离查找表 |
| 扫描编码 | `IndexIVFPQ.cpp:861` | `scan_list_with_table()` | 遍历PQ编码 |
| PQ距离计算 | `pq_code_distance.h:63` | `distance_single_code()` | 查表计算距离 |
| 堆排序 | `ResultHandler.h:283` | `HeapHandler::add_result()` | 维护top-k |

**距离计算公式 (L2 with by_residual):**

```
d²(q, x) = ||q - centroid||²           // dis0 (粗距离)
           + ||(q-centroid) - PQ(x)||²  // PQ精距离

其中 PQ距离通过查表计算:
  sim_table[m][c] = 距离表第m个子量化器的第c个质心距离
  dis = dis0 + Σ sim_table[m][code[m]]  for m in 0..M-1
```

---

## 3. 一级二级量化器详解

### 3.1 量化层次结构

```
原始向量 x (128维)
    │
    ├─→ 【一级量化器】Coarse Quantizer
    │   目的: 将向量空间划分为 nlist 个区域(Voronoi cell)
    │   方法: K-means 聚类
    │   输出: coarse_id (桶编号, 如 0~315)
    │
    └─→ 【二级量化器】Product Quantizer
        目的: 压缩存储残差向量 (x - centroid)
        方法: 分组PQ量化
        输出: PQ编码 (如 M=16, nbits=8 → 16字节)
```

### 3.2 为什么需要两级量化？

| 层级 | 作用 | 受益 |
|------|------|------|
| **一级量化** | 缩小搜索范围 | 从全库搜索 → 只搜索 nprobe 个桶 |
| **二级量化** | 压缩存储 | 从 float×d 字节 → PQ编码字节 |

**举例**: 100万向量，128维
- **一级量化**: 划分为316个桶，搜索时只查16个桶 (只查5%数据)
- **二级量化**: 每个向量从 512字节 (128×4) → 16字节 (压缩32倍)

### 3.3 存储结构

```
IndexIVFPQ 索引结构:
┌──────────────────────────────────────────────────────┐
│ Coarse Quantizer (一级量化器)                         │
│   - nlist 个 centroid (每个128维浮点)                 │
│   - 用于粗分类, 找到最近的 nprobe 个桶               │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│ Product Quantizer (二级量化器)                        │
│   - M×ksub 个 centroid (每个 dsub 维)                │
│   - 用于编码残差向量                                  │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│ Inverted Lists (倒排列表)                             │
│   List 0: [id1, PQ_code1], [id2, PQ_code2], ...      │
│   List 1: [id3, PQ_code3], ...                       │
│   ...                                                 │
│   List nlist-1: ...                                  │
└──────────────────────────────────────────────────────┘
```

### 3.4 构建与检索的量化顺序

#### 构建索引阶段: 先粗分类，后PQ量化

```
原始向量 x
    │
    ├─→ 先用 Coarse Quantizer (K-means)
    │   └─→ 得到 nlist 个 centroid, 分配 coarse_id
    │
    └─→ 计算残差 residual = x - centroid[coarse_id]
        │
        └─→ 用 PQ 量化器编码 residual
            └─→ 得到 PQ_code
```

#### 检索阶段: 也是先粗分类，后PQ计算

```
search(query):
    │
    ├─→ 【第一步】粗量化
    │   quantizer->search(query, nprobe)
    │   └─→ 找到最近的 nprobe 个桶
    │
    └─→ 【第二步】精细搜索
        对每个候选桶:
            ├─→ 预计算 PQ 距离表
            ├─→ 扫描桶内的 PQ 编码
            └─→ 查表计算距离
```

**核心原因**: PQ量化的是**残差向量**，而残差 = 原向量 - 所属桶的质心，所以**必须先知道桶号**才能做PQ编码。

---

## 4. Faiss 量化器家族

### 4.1 量化器继承结构

```
Quantizer (基类) - impl/Quantizer.h:15
    │
    ├── ProductQuantizer (PQ) - impl/ProductQuantizer.h:29
    │   └── 特点: 分段量化, 每段独立
    │
    ├── ScalarQuantizer (SQ) - impl/ScalarQuantizer.h:24
    │   └── 特点: 每个维度独立量化
    │       ├── QT_8bit (每维度1字节)
    │       ├── QT_4bit (每维度0.5字节)
    │       ├── QT_fp16 (半精度浮点)
    │       └── QT_bf16 (bfloat16)
    │
    ├── ResidualQuantizer (RQ) - impl/ResidualQuantizer.h:27
    │   └── 特点: 多层残差, 逐级细化
    │
    ├── LocalSearchQuantizer (LSQ) - impl/LocalSearchQuantizer.h:45
    │   └── 特点: 优化全局重构误差
    │
    └── AdditiveQuantizer (加性量化器) - impl/AdditiveQuantizer.h:26
        ├── ProductResidualQuantizer (PRQ)
        └── ProductLocalSearchQuantizer (PLSQ)
```

### 4.2 各量化器对比

| 量化器 | 压缩率 (128维) | 精度 | 速度 | 适用场景 |
|--------|----------------|------|------|----------|
| **PQ** | M=16,nbits=8 → 16字节 | 中 | 快 | **通用推荐** |
| **SQ-8bit** | 128字节 | 高 | 最快 | 内存充足时 |
| **SQ-4bit** | 64字节 | 中低 | 快 | 极致压缩 |
| **RQ** | 可配置 | 可变 | 慢 | 高精度需求 |

### 4.3 典型组合

```
IndexIVFPQ           ← IVF + PQ (最常用)
IndexIVFScalarQuantizer ← IVF + SQ
IndexIVFResidualQuantizer ← IVF + RQ
```

---

## 5. SPTAG vs Faiss IndexIVFPQ

### 5.1 架构对比

```
┌────────────────────────────────────────────────────────────────────┐
│                        SPTAG                                      │
├────────────────────────────────────────────────────────────────────┤
│  原始向量 x                                                        │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────┐   全部向量统一量化                                │
│  │  ①全局量化  │   通用量化器 (PQ/OPQ/其他)                         │
│  │  PQ/OPQ等   │   对原向量直接量化                                │
│  └─────────────┘                                                   │
│      │                                                             │
│      ▼                                                             │
│  量化码 code                                                       │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────┐   基于某种准则划分到 posting list                 │
│  │  ②划分列表  │   KD-tree / SSG / 其他树结构                       │
│  │  Posting    │   基于量化码或原始特征划分                         │
│  │  List       │                                                    │
│  └─────────────┘                                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     Faiss IndexIVFPQ                              │
├────────────────────────────────────────────────────────────────────┤
│  原始向量 x                                                        │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────┐   先粗分类，每个向量去不同的量化器               │
│  │  ①粗分类    │   Coarse Quantizer (K-means)                      │
│  │  分桶       │   每个向量属于某个桶                              │
│  └─────────────┘                                                   │
│      │                                                             │
│      ▼                                                             │
│  residual = x - centroid[桶号]  ← 每个桶的残差不同               │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────┐   对残差进行PQ量化                               │
│  │  ②PQ量化   │   Product Quantizer                               │
│  │  残差编码   │   量化的是残差，不是原向量                         │
│  └─────────────┘                                                   │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 核心差异

| 维度 | SPTAG | Faiss IndexIVFPQ |
|------|-------|------------------|
| **量化对象** | 原始向量 x | 残差向量 (x - centroid) |
| **量化时机** | 先量化，后划分 | 先划分，后量化 |
| **量化方式** | 统一的全局量化器 | 每个桶共享同一个PQ，但量化的是各自残差 |
| **列表划分** | 树结构/图遍历 | K-means 聚类 |
| **搜索策略** | 遍历树节点找候选 | 直接找 nprobe 个最近桶 |

### 5.3 为什么 Faiss 要量化残差？

**数学角度:**

```
L2距离分解:
||q - x||² = ||q - c||²              + ||c - x||²              - 2⟨q-c, c-x⟩
           = (q到桶中心的距离)²        + (残差的范数)²              - 2×内积项
              ↑                              ↑
           粗距离              精距离（PQ编码）

如果 by_residual=true:
  - 粗距离: ||q - c||² 可以快速计算（只需要算nprobe个）
  - 精距离: PQ编码的是残差 r = x - c
  - 距离 = dis0 + PQ_distance(q - c, PQ_code(x-c))
```

**好处:**

| 优势 | 说明 |
|------|------|
| **降低量化难度** | 残差比原向量分布更集中，更容易用PQ拟合 |
| **提高精度** | 同样的PQ参数，量化残差的误差更小 |
| **加速计算** | dis0 预计算，只需查表算PQ距离 |

### 5.4 SPTAG vs Faiss 总结

| 方面 | SPTAG | Faiss IVFPQ |
|------|-------|-------------|
| **设计哲学** | 先全局量化，再结构划分 | 先空间划分，再局部量化 |
| **量化目标** | 原始向量 | 残差向量 |
| **适用场景** | 需要全局最优子空间 | 需要快速粗筛选+精搜索 |
| **精度来源** | OPQ 全局优化 + 树剪枝 | 粗细两级距离分解 |

---

## 附录：关键文件路径

```
faiss/
├── IndexIVFPQ.h              # IVFPQ 索引头文件
├── IndexIVFPQ.cpp            # IVFPQ 索引实现
├── IndexIVF.h                # IVF 基类
├── IndexIVF.cpp              # IVF 基类实现
├── index_io.h                # 索引入口导出接口
├── invlists/
│   └── InvertedLists.h       # 倒排列表接口
├── impl/
│   ├── Quantizer.h           # 量化器基类
│   ├── ProductQuantizer.h    # PQ 量化器
│   ├── ScalarQuantizer.h     # SQ 量化器
│   ├── ResidualQuantizer.h   # RQ 量化器
│   ├── pq_code_distance.h    # PQ 距离计算
│   ├── index_write.cpp       # 索引入口
│   └── index_read.cpp        # 索引导入
└── utils/
    ├── Heap.h                # 堆操作
    └── distances.h           # 距离计算
```

---

*文档生成时间: 2026-03-27*
*基于 Faiss 源码分析*
