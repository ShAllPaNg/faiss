# Faiss IndexIVFPQ Conversation Record

## Conversation 1: IndexPQ Implementation

**User**: 扫描一下faiss目录下的CPU的indexPQ索引类型的实现，然后向我介绍

**Assistant**: 
## IndexPQ 索引实现介绍

IndexPQ是Faiss中基于**乘积量化**的CPU索引类型。

### 核心结构

```
IndexPQ : IndexFlatCodes
├── ProductQuantizer pq  // 乘积量化器
├── search_type          // 搜索类型
├── polysemous_ht        // 多义汉明阈值
└── codes                // 存储的编码数据
```

### ProductQuantizer原理

- **维度分割**：将d维向量分成M个子空间，每个子空间维度为 d/M
- **子量化器**：每个子空间有2^nbits个质心（通常nbits=8，即256个质心）
- **编码**：向量编码为M个索引，每个索引占nbits位

### 训练过程 (IndexPQ.cpp:50-72)

1. **标准训练**：调用`pq.train(n, x)`，使用k-means聚类
2. **多义训练**：优化质心使汉明距离更有效

### 搜索模式 (IndexPQ.h:64-71)

```cpp
ST_PQ                    // 非对称PQ搜索（默认）
ST_HE                    // 汉明距离
ST_generalized_HE        // 广义汉明距离  
ST_SDC                   // 对称PQ搜索
ST_polysemous            // 汉明过滤+PQ组合
ST_polysemous_generalize // 广义汉明过滤+PQ
```

### 核心搜索流程 (IndexPQ.cpp:164-270)

1. **ST_PQ模式**：
   - 计算查询向量的距离表（M×ksub）
   - 对每个数据库编码，查表累加距离
   - 维护top-k堆

2. **Polysemous搜索**（IndexPQ.cpp:332-436）：
   - 先计算查询编码
   - 用汉明距离快速过滤（阈值过滤）
   - 对通过过滤的向量计算精确PQ距离

### 编解码 (IndexPQ.cpp:440-446)

```cpp
sa_encode: pq.compute_codes(x, bytes, n)  // 向量→编码
sa_decode: pq.decode(bytes, x, n)          // 编码→向量
```

### 性能优化

- **距离表预计算**：避免重复计算子向量到质心的距离
- **SIMD指令**：使用SIMD加速距离计算
- **并行化**：OpenMP多线程处理多个查询
- **多义性**：汉明距离过滤减少精确计算量

### 典型参数

- d=128, M=8, nbits=8 → code_size=8字节
- 压缩率：128×4字节 → 8字节，约64倍压缩

---

## Conversation 2: IndexIVFPQ Introduction

**User**: 找到indexIVFPQ的代码，给我介绍下

**Assistant**: 
## IndexIVFPQ 索引实现介绍

IndexIVFPQ 是 **倒排索引(IVF) + 乘积量化(PQ)** 的组合索引，继承自 `IndexIVF`。

### 核心结构 (IndexIVFPQ.h:34-151)

```
IndexIVFPQ : IndexIVF
├── ProductQuantizer pq           // 乘积量化器
├── by_residual = true            // 默认编码残差向量
├── use_precomputed_table         // 预计算表模式
├── precomputed_table             // 预计算的距离表
├── polysemous_ht                 // 多义汉明阈值
└── scan_table_threshold          // 扫描阈值
```

### 核心思想

**两级量化**：
1. **粗量化**：用量化器(如IndexFlat)找到最近的聚类中心
2. **细量化**：用PQ编码**残差向量**（原始向量 - 聚类中心）

### 添加向量流程 (IndexIVFPQ.cpp:225-327)

```cpp
add_core_o():
  1. quantizer->assign(n, x, idx)           // 找到每个向量的聚类
  2. compute_residuals(x, idx)              // 计算残差: x - centroid
  3. pq.compute_codes(residuals, codes)     // PQ编码残差
  4. invlists->add_entry(key, id, code)     // 存入倒排列表
```

### 搜索流程

**距离计算分解** (IndexIVFPQ.cpp:349-377)：

对于L2距离 `d = ||x - y_C - y_R||²`（x=查询，y_C=粗质心，y_R=细PQ质心）：

```
d = ||x - y_C||²  +  ||y_R||²  +  2*(y_C|y_R)  -  2*(x|y_R)
     term1            term2       term3           term4
```

- **term1**: 粗搜索时已计算
- **term2+term3**: 可预计算（不依赖查询x）→ `precomputed_table`
- **term4**: 查询时计算的距离表

### 预计算表 (IndexIVFPQ.cpp:379-485)

```cpp
use_precomputed_table:
  0: 不使用（默认）
  1: 标准预计算表 (nlist × M × ksub)
  2: 紧凑版本（用于MultiIndexQuantizer）
```

### 扫描器实现 (IndexIVFPQ.cpp:804-1252)

`IVFPQScanner` 模板类实现了三种扫描模式：

1. **scan_list_with_table** (precompute_mode=2)
   - 使用预计算的距离表
   - 4路展开优化

2. **scan_list_with_pointer** (precompute_mode=1)
   - 使用表指针，减少内存访问

3. **scan_on_the_fly_dist** (precompute_mode=0)
   - 实时解码计算，无预计算

### 多义搜索 (IndexIVFPQ.cpp:1040-1157)

```cpp
scan_list_polysemous_hc():
  1. 计算查询编码 q_code
  2. 对每个数据库编码计算汉明距离
  3. 若 hd < polysemous_ht，计算精确PQ距离
  4. 4路展开+SIMD优化
```

### 编解码 (IndexIVFPQ.cpp:97-134)

```cpp
encode(key, x, code):
  residual = x - centroid[key]
  pq.compute_code(residual, code)

decode_multiple(keys, codes, x):
  pq.decode(codes, x)
  x += centroid[keys]  // 加回质心
```

### 性能优化

| 技术 | 作用 |
|------|------|
| 预计算表 | 避免重复计算质心相关项 |
| 4路展开 | 减少分支预测开销 |
| SIMD | 加速距离计算 |
| 多义过滤 | 汉明距离快速过滤 |
| OpenMP | 多线程处理 |

### 典型参数

```
d=128, nlist=4096, M=8, nbits=8
code_size = 8字节
压缩率: 128×4 → 8字节 (64倍)
```

### 与IndexPQ的区别

| 特性 | IndexPQ | IndexIVFPQ |
|------|---------|------------|
| 搜索范围 | 全库 | 仅相关聚类 |
| 编码对象 | 原始向量 | 残差向量 |
| 精度 | 较低 | 较高 |
| 速度 | 较慢 | 较快（nprobe小） |

---

## Conversation 3: by_residual Parameter

**User**: 可以不使用残差吗？

**Assistant**: 可以。设置 `by_residual = false` 即可。

### 代码位置

IndexIVFPQ.cpp:57 默认 `by_residual = true`，但可以手动设为 false。

### 区别对比

| 模式 | 编码对象 | 距离计算 |
|------|----------|----------|
| `by_residual=true` | `x - centroid` | 残差+质心分解 |
| `by_residual=false` | `x` (原始向量) | 直接PQ距离 |

### 关键代码逻辑

```cpp
// 编码 (IndexIVFPQ.cpp:97-104)
void IndexIVFPQ::encode(idx_t key, const float* x, uint8_t* code) const {
    if (by_residual) {
        quantizer->compute_residual(x, residual, key);
        pq.compute_code(residual, code);
    } else {
        pq.compute_code(x, code);  // 直接编码原始向量
    }
}

// L2搜索 (IndexIVFPQ.cpp:577-583)
void init_query_L2() {
    if (!by_residual) {
        pq.compute_distance_table(qi, sim_table);  // 直接用查询向量
    }
}
```

### 何时不用残差

1. **Inner Product 距离**：残差对IP距离无意义
2. **聚类中心不明显**：数据分布均匀时
3. **简化实现**：不需要预计算表

### 注意

`by_residual=false` 时，`use_precomputed_table` 会被禁用（IndexIVFPQ.cpp:396-402），因为预计算表是为残差模式优化的。

---

## Conversation 4: How to Set by_residual

**User**: 如何设置by_residual = false?好像没看到相关接口

**Assistant**: `by_residual` 是 **public 成员变量**，直接访问即可：

### C++ 使用

```cpp
faiss::IndexIVFPQ index(quantizer, d, nlist, M, nbits);
index.by_residual = false;  // 直接设置
```

### Python 使用

```python
import faiss

index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
index.by_residual = False  # 直接设置属性
```

### 注意事项

1. **必须在 `add()` 之前设置**，否则已编码的向量仍使用残差模式
2. 设置 `by_residual=false` 后，`use_precomputed_table` 会自动失效
3. 对于 **Inner Product 度量**，`by_residual=true` 没有意义，建议设为 `false`

---

## Conversation 5: IndexIVFPQ::train Function

**User**: indexIVFPQ的train成员函数干什么用的？给我详细解释一下其流程

**Assistant**: 
## IndexIVFPQ::train 详细流程

`IndexIVFPQ` 继承自 `IndexIVF`，`train()` 定义在父类中。

### 整体流程图

```
IndexIVF::train(n, x)
    │
    ├─► 1. train_q1()           // 训练一级量化器（聚类中心）
    │
    ├─► 2. 子采样（可选）        // 限制训练向量数量
    │
    ├─► 3. 计算残差（如果by_residual=true）
    │
    └─► 4. train_encoder()      // IndexIVFPQ重写，训练PQ
```

---

### 1. train_q1() - 训练一级量化器 (IndexIVF.cpp:56-113)

```cpp
void Level1Quantizer::train_q1(n, x, verbose, metric_type) {
    if (quantizer->is_trained && quantizer->ntotal == nlist) {
        // 已训练，跳过
        return;
    }
    
    // 使用 k-means 聚类训练量化器
    Clustering clus(d, nlist, cp);  // cp 是聚类参数
    clus.train(n, x, *quantizer);   // 在数据上训练
    quantizer->is_trained = true;
}
```

**作用**：生成 `nlist` 个聚类中心（粗质心）

---

### 2. IndexIVF::train() 主流程 (IndexIVF.cpp:1212-1245)

```cpp
void IndexIVF::train(idx_t n, const float* x) {
    // Step 1: 训练一级量化器
    train_q1(n, x, verbose, metric_type);
    
    // Step 2: 子采样（如果训练数据太多）
    idx_t max_nt = train_encoder_num_vectors();  // 子类可重写
    TransformedVectors tv(x, fvecs_maybe_subsample(d, &n, max_nt, x, verbose));
    
    // Step 3: 准备训练数据
    if (by_residual) {
        // 计算每个向量所属的聚类
        std::vector<idx_t> assign(n);
        quantizer->assign(n, tv.x, assign.data());
        
        // 计算残差: residual = x - centroid
        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());
        
        // Step 4: 用残差训练编码器
        train_encoder(n, residuals.data(), assign.data());
    } else {
        // 直接用原始向量训练
        train_encoder(n, tv.x, nullptr);
    }
    
    is_trained = true;
}
```

---

### 3. IndexIVFPQ::train_encoder() (IndexIVFPQ.cpp:69-91)

```cpp
void IndexIVFPQ::train_encoder(idx_t n, const float* x, const idx_t* /*assign*/) {
    // Step 4a: 训练 Product Quantizer
    pq.train(n, x);  // 对残差向量进行 k-means
    
    // Step 4b: 多义训练（可选）
    if (do_polysemous_training) {
        PolysemousTraining pt;
        pt.optimize_pq_for_hamming(pq, n, x);
    }
    
    // Step 4c: 预计算表（如果 by_residual && L2）
    if (by_residual) {
        precompute_table();
    }
}
```

---

### 4. pq.train() - ProductQuantizer训练

```
ProductQuantizer::train(n, x):
    │
    ├─► 将 d 维向量分成 M 个子空间
    │
    └─► 对每个子空间独立做 k-means（得到 ksub=2^nbits 个质心）
```

---

### 训练数据量计算 (IndexIVFPQ.cpp:89-91)

```cpp
idx_t IndexIVFPQ::train_encoder_num_vectors() const {
    return pq.cp.max_points_per_centroid * pq.ksub;
    // 例如: max_points=256, ksub=256, M=8
    // 需要: 256 * 256 = 65536 个向量
}
```

---

### 总结

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| train_q1 | 原始向量 | nlist个聚类中心 | 粗量化器 |
| compute_residual | 原始向量-质心 | 残差向量 | 减少量化误差 |
| pq.train | 残差向量 | M×ksub个PQ质心 | 细量化器 |
| precompute_table | PQ质心+粗质心 | 预计算距离表 | 加速搜索 |

### Python 示例

```python
import faiss

# 创建索引
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist=100, M=8, nbits=8)

# 训练（只需调用一次）
index.train(training_data)  # training_data: (n, d)

# 之后可以 add 和 search
index.add(data)
index.search(query, k)
```

---

## Conversation 6: Purpose of train_q1

**User**: train_q1这个有啥用？

**Assistant**: `train_q1` 是 **训练一级量化器（粗量化器）** 的函数。

### 作用

生成 `nlist` 个聚类中心（粗质心），用于将向量空间划分成 `nlist` 个区域（倒排列表）。

### 核心代码 (IndexIVF.cpp:56-113)

```cpp
void Level1Quantizer::train_q1(size_t n, const float* x, ...) {
    // 如果量化器已训练且质心数正确，跳过
    if (quantizer->is_trained && quantizer->ntotal == nlist) {
        return;
    }
    
    // k-means 聚类
    Clustering clus(d, nlist, cp);
    clus.train(n, x, *quantizer);
    quantizer->is_trained = true;
}
```

### 实际效果

```
训练前: quantizer 中无质心
训练后: quantizer 中有 nlist 个质心 (如 4096 个)

这些质心用于:
1. add() 时: 找到向量最近的质心，决定放入哪个倒排列表
2. search() 时: 找到查询最近的 nprobe 个质心，只搜索这些列表
```

### 为什么要单独一个函数？

| 场景 | 说明 |
|------|------|
| **复用** | 子类（IndexIVFPQ等）都共用这个一级量化训练逻辑 |
| **灵活性** | 支持三种模式 `quantizer_trains_alone` (0/1/2) |
| **外部预训练** | 如果 quantizer 已训练，会自动跳过 |

### 三种训练模式

```cpp
quantizer_trains_alone = 0  // 默认：用 k-means 训练 (Clustering)
quantizer_trains_alone = 1  // 量化器自己训练 (quantizer->train)
quantizer_trains_alone = 2  // 强制用 L2 IndexFlat 训练
```

### 简单理解
```
train_q1 = "把 nlist 个聚类中心找出来"

例如 nlist=100:
- 用 k-means 把训练数据聚成 100 类
- 每类的中心就是一个 "粗质心"
- 每个质心对应一个倒排列表 (inverted list)
```

---

## Conversation 7: Difference Between nlist and PQ ksub

**User**: PQ训练时不也每个子空间也要分成256（nbit=8)个聚类嘛，这个nlist和后面PQ的聚类有什么区别和联系

**Assistant**: 
## nlist vs PQ ksub 的区别与联系

### 核心区别

| 特性 | 一级量化 (nlist) | 二级量化 (PQ ksub) |
|------|------------------|-------------------|
| **空间** | 整个 d 维空间 | M 个独立的 d/M 维子空间 |
| **质心维度** | d 维 | d/M 维 |
| **质心数量** | nlist 个 | M × ksub 个 |
| **聚类方式** | 一次 k-means | M 次独立 k-means |
| **作用** | 划分倒排列表 | 压缩编码向量 |
| **典型值** | 100~10000 | 256 (nbits=8) |

### 图示

```
一级量化 (nlist=4)              二级量化 (M=4, ksub=256)
─────────────────              ─────────────────────────

    d维全空间                    分成4个子空间
   ┌─────────┐                  
   │  ●  ○   │                  子空间0  子空间1  子空间2  子空间3
   │    ×    │  4个d维质心      ├────┤   ├────┤   ├────┤   ├────┤
   │  △  □   │  ─────────►     kmeans   kmeans   kmeans   kmeans
   └─────────┘                  256个    256个    256个    256个
                                (d/4)维  (d/4)维  (d/4)维  (d/4)维
```

### 具体例子
```
d=128, nlist=1000, M=8, nbits=8

一级量化:
  - 1 次 k-means，聚成 1000 类
  - 1000 个 128 维质心
  - 每个质心 → 一个倒排列表

二级量化 (PQ):
  - 8 次 k-means（每个子空间1次）
  - 8 × 256 = 2048 个 16 维质心
  - 每个向量 → 8 个索引（8字节）
```

### 两者关系（by_residual=true 时）
```
原始向量 x (128维)
      │
      ▼
一级量化: 找最近的粗质心 c (128维)
      │
      ▼
计算残差: r = x - c (128维)
      │
      ▼
二级量化: 对残差 r 做 PQ 编码
      - r 分成 8 段 (每段16维)
      - 每段找最近的 PQ 质心
      - 得到 8 个索引 (code)
```

### 为什么两级都要聚类？

| 只有 PQ (IndexPQ) | IVF + PQ (IndexIVFPQ) |
|-------------------|----------------------|
| 搜索时扫描**全库** | 只扫描 **nprobe 个列表** |
| O(n) | O(n × nprobe/nlist) |
| 精度低（残差大） | 精度高（残差小） |

```
只有 PQ:
  向量 → 直接 PQ 编码 → 全库扫描

IVF + PQ:
  向量 → 找粗质心 → 计算残差 → PQ编码残差 → 只搜相关列表
         (一级)      (减小误差)   (二级)      (加速)
```

### 残差的作用
```
by_residual=true:

  x ───┐
       ├──► r = x - c ──► PQ(r) ──► code (8字节)
  c ───┘
  
  残差 r 通常比原始 x "更小更集中"
  → PQ 编码误差更小
  → 搜索精度更高
```

### 总结
```
nlist: "地图分区" —— 把世界分成 nlist 个区域
ksub:  "压缩字典" —— 每个子空间有 256 个"词"

搜索流程:
  1. 看查询属于哪些"区域" (nprobe 个)
  2. 只在这些区域里，用"压缩字典"匹配
```

---

## Conversation 8: nlist vs Training Vectors

**User**: 我们先以by_residual=true为前提进行讨论，一节量化时，如果nlist数量超过训练向量数量还有意义吗？

**Assistant**: **没有意义，而且会出问题。**

### 原因

一级量化使用 **k-means 聚类**，要求：

```
训练向量数 n >= nlist（质心数）
```

如果 `n < nlist`：
1. 无法初始化 nlist 个质心
2. 聚类失败或产生空聚类
3. 大量倒排列表为空

### Faiss 的检查 (Clustering.cpp:275-330)

```cpp
// 1. 强制检查
FAISS_THROW_IF_NOT_FMT(
    nx >= k,
    "Number of training points (%zd) should be at least as large as number of clusters (%zd)",
    nx, k);

// 2. 警告检查
if (nx < k * min_points_per_centroid) {
    fprintf(stderr, "WARNING clustering %zd points to %zd centroids: "
            "please provide at least %zd training points\n", ...);
}

// 3. 边界情况：刚好相等
if (nx == k) {
    // 直接把训练点当作质心，跳过 k-means
    memcpy(centroids, x, sizeof(float) * d * k);
}
```

### 结论

| 情况 | 训练向量数 n | 结果 |
|------|-------------|------|
| `n < nlist` | 100 < 1000 | **报错** |
| `n == nlist` | 1000 == 1000 | 每个训练点变成一个质心（勉强可行） |
| `n < nlist × min_points` | 500 < 1000×1 | 警告，质心不稳定 |
| `n >= nlist × min_points` | 10000 > 1000×1 | 正常 |

### 实际建议
```
训练向量数 >= nlist × min_points_per_centroid (默认=1)
更理想：训练向量数 >= nlist × 256 (max_points_per_centroid)
```

**所以 nlist 超过训练向量数没有意义，直接报错。**

---

## Conversation 9: max_points_per_centroid

**User**: 既然有min_points，是否有max_points呢？如果训练向量数量n>max_points* nlist会怎么样？

**Assistant**: **有的，会自动子采样。**

### 代码证据 (Clustering.cpp:313-321)

```cpp
if (static_cast<size_t>(nx) > k * max_points_per_centroid) {
    // 子采样到 k * max_points_per_centroid 个点
    nx = subsample_training_set(
            *this, nx, x, line_size, weights, &x_new, &weights_new);
    ...
}
```

### 子采样函数 (Clustering.cpp:71-121)

```cpp
idx_t subsample_training_set(...) {
    printf("Sampling a subset of %zd / %" PRId64 " for training\n",
           clus.k * clus.max_points_per_centroid, nx);
    
    // 随机采样 k * max_points_per_centroid 个点
    nx = clus.k * clus.max_points_per_centroid;
    ...
}
```

### 默认值

```cpp
// Clustering.h
int max_points_per_centroid = 256;
int min_points_per_centroid = 1;
```

### 完整逻辑

| 训练向量数 n | 处理方式 |
|-------------|----------|
| `n < nlist` | **报错** |
| `n < nlist × min` | 警告，继续执行 |
| `nlist × min ≤ n ≤ nlist × max` | 正常使用全部数据 |
| `n > nlist × max` | **随机子采样到 nlist × max 个点** |

### 例子
```
nlist = 1000
max_points_per_centroid = 256

训练数据 n = 1,000,000
阈值 = 1000 × 256 = 256,000

结果: 随机采样 256,000 个点用于训练
```

### IndexIVFPQ 的训练向量数限制

```cpp
// IndexIVFPQ.cpp:89-91
idx_t IndexIVFPQ::train_encoder_num_vectors() const {
    return pq.cp.max_points_per_centroid * pq.ksub;
    // 例如: 256 × 256 = 65,536
}
```

这是针对 **PQ 训练**（二级量化）的限制，一级量化用 Clustering 的 max_points。

---

## Conversation 10: First-Level Training Data

**User**: 我现在说的是一级量化训练

**Assistant**: 一级量化训练确实有 `max_points_per_centroid`，默认 **256**。

### 一级量化训练流程 (IndexIVF.cpp:1212-1245)

```cpp
void IndexIVF::train(idx_t n, const float* x) {
    // Step 1: 训练一级量化器
    train_q1(n, x, verbose, metric_type);  // 内部用 Clustering
    
    // Step 2: 子采样（用于二级量化训练）
    idx_t max_nt = train_encoder_num_vectors();  // IndexIVF 默认返回 0
    TransformedVectors tv(x, fvecs_maybe_subsample(d, &n, max_nt, x, verbose));
    
    // Step 3: 训练二级量化器
    ...
}
```

### 一级量化的 max_points (Clustering.cpp:313-321)

```cpp
// train_q1 内部调用 Clustering::train
if (nx > k * max_points_per_centroid) {  // k = nlist
    nx = subsample_training_set(...);  // 子采样到 nlist × 256 个点
}
```

### 总结

| 阶段 | 参数 | 默认值 | 作用 |
|------|------|--------|------|
| 一级量化 | `max_points_per_centroid` | 256 | 限制 k-means 训练数据量 |
| 二级量化 | `pq.cp.max_points_per_centroid` | 256 | 限制 PQ 训练数据量 |

### 实际效果
```
nlist = 1000
max_points_per_centroid = 256

训练数据 n = 1,000,000

一级量化训练: 采样到 1000 × 256 = 256,000 个点做 k-means
```

### 如何修改？

```cpp
// C++
IndexIVFPQ index(...);
index.cp.max_points_per_centroid = 512;  // 修改一级量化
index.pq.cp.max_points_per_centroid = 512;  // 修改二级量化

// Python
index = faiss.IndexIVFPQ(...)
index.cp.max_points_per_centroid = 512
index.pq.cp.max_points_per_centroid = 512
```

---

## Conversation 11: First-Level Uses Original Vectors

**User**: 一级训练用的是原始向量吗？

**Assistant**: **是的，用原始向量。**

### IndexIVF::train 流程 (IndexIVF.cpp:1212-1242)

```cpp
void IndexIVF::train(idx_t n, const float* x) {
    
    // 一级量化：用原始向量 x
    train_q1(n, x, verbose, metric_type);
    
    // ...子采样...
    
    // 二级量化
    if (by_residual) {
        // 用残差训练
        quantizer->compute_residual_n(n, tv.x, residuals, assign);
        train_encoder(n, residuals, assign);
    } else {
        // 用原始向量训练
        train_encoder(n, tv.x, nullptr);
    }
}
```

### 总结

| 阶段 | by_residual=true | by_residual=false |
|------|------------------|-------------------|
| 一级量化 | 原始向量 x | 原始向量 x |
| 二级量化 | 残差向量 (x - centroid) | 原始向量 x |

### 为什么一级用原始向量？
```
一级量化的目的：把整个向量空间划分成 nlist 个区域
→ 需要在原始空间中找到 nlist 个代表点（质心）
→ 所以用原始向量做 k-means

二级量化的目的：编码残差（减少量化误差）
→ 残差 = 原始向量 - 最近粗质心
→ 残差通常更小更集中，PQ 编码误差更小
```

---

## Conversation 12: Adding Vectors to Clusters

**User**: 将原始向量分成nlsit个区域，训练完成后，再往索引中添加向量，也是往nlist个簇中进行添加吗？

**Assistant**: **是的。**

### 添加向量流程 (IndexIVFPQ.cpp:225-327)

```cpp
void IndexIVFPQ::add_core_o(idx_t n, const float* x, ...) {
    // 1. 找到每个向量最近的质心（所属的簇）
    quantizer->assign(n, x, idx);  // idx[i] = 第i个向量属于哪个簇
    
    // 2. 计算残差（如果 by_residual=true）
    if (by_residual) {
        quantizer->compute_residual(x, residual, idx);
    }
    
    // 3. PQ 编码
    pq.compute_codes(residual, code);
    
    // 4. 加入对应的倒排列表
    invlists->add_entry(idx, id, code);  // idx 就是簇编号
}
```

### 图示
```
训练完成后:
  quantizer 中有 nlist 个质心 (如 1000 个)
  对应 1000 个倒排列表 (inverted lists)

添加向量时:
  向量 x ──► quantizer.assign() ──► 找到最近质心 idx=23
                                        │
                                        ▼
                                  加入倒排列表 #23
```

### 总结

| 阶段 | 操作 | 使用的数据结构 |
|------|------|---------------|
| 训练 | k-means 聚类 | quantizer (存储 nlist 个质心) |
| 添加 | assign + add_entry | quantizer + invlists (nlist 个列表) |

**训练决定了 nlist 个簇，添加时向量只能放入这 nlist 个簇中。**

---

## Conversation 13: Vector Addition Flow

**User**: 对于每一个新加入和向量，先找到nlist中的最近的一个作为所属的簇，然后与质心计算残差，得到残差向量，再对残差进行PQ量化，得到PQ码，是这样吗

**Assistant**: **完全正确。**

### 代码验证 (IndexIVFPQ.cpp:225-284)

```cpp
void IndexIVFPQ::add_core_o(idx_t n, const float* x, ...) {
    // Step 1: 找到最近的质心（所属的簇）
    quantizer->assign(n, x, idx);  // idx[i] = 第i个向量的簇编号
    
    // Step 2: 计算残差
    if (by_residual) {
        // residual[i] = x[i] - centroid[idx[i]]
        quantizer->compute_residual_n(n, x, residuals, idx);
    }
    
    // Step 3: 对残差进行 PQ 编码
    pq.compute_codes(to_encode, codes, n);  // to_encode = residuals
    
    // Step 4: 存入对应倒排列表
    invlists->add_entry(key, id, code);
}
```

### 流程图
```
新向量 x (128维)
      │
      ▼
┌─────────────────────┐
│ quantizer.assign(x) │ ──► 找到最近质心 c，得到簇编号 idx
└─────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ residual = x - centroid[idx] │ ──► 残差向量 r (128维)
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ pq.compute_codes(r)     │ ──► PQ码 (8字节)
│   - 分成 M=8 段          │
│   - 每段找最近PQ质心     │
│   - 得到 8 个索引        │
└─────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│ invlists[idx].add(id, code)   │ ──► 存入倒排列表 idx
└───────────────────────────────┘
```

### 存储内容

每个倒排列表存储：
```
倒排列表 #23:
┌──────────────────────────────────┐
│ id_1 | code_1 (8字节)            │
│ id_2 | code_2 (8字节)            │
│ ...                              │
│ id_k | code_k (8字节)            │
└──────────────────────────────────┘

code = PQ编码的残差（不是原始向量）
```

---

## Conversation 14: What's Stored in Inverted Lists

**User**: 那么，在倒排索引中保存的是完整向量，还是向量对应的ID？

**Assistant**: **保存的是：向量 ID + PQ 编码（不是完整向量）。**

### 代码证据 (IndexIVFPQ.cpp:302)

```cpp
invlists->add_entry(key, id, code);

// key  = 簇编号（决定存哪个列表）
// id   = 向量的唯一标识
// code = PQ 编码（8字节，压缩后的残差）
```

### 倒排列表结构
```
倒排列表 #23:
┌────────┬────────────┐
│   ID   │  PQ Code   │
├────────┼────────────┤
│  1042  │  [8字节]   │
│  5681  │  [8字节]   │
│ 12033  │  [8字节]   │
│  ...   │   ...      │
└────────┴────────────┘

每个条目:
  - ID: 8字节 (int64)
  - PQ Code: M字节 (如 M=8)
  - 总计: 16字节/向量 (vs 原始 128×4=512字节)
```

### 对比

| 索引类型 | 倒排列表存储 | 每向量大小 |
|---------|-------------|-----------|
| IndexIVFFlat | ID + 完整向量 | 8 + 512 = 520字节 |
| IndexIVFPQ | ID + PQ编码 | 8 + 8 = 16字节 |

### 为什么不存完整向量？
```
目的: 压缩存储

原始向量: 128维 × 4字节 = 512字节
PQ编码:   8个索引 × 1字节 = 8字节

压缩比: 512 / 8 = 64倍
```

### 如何还原向量？

```cpp
// 从倒排列表读取
id = list_ids[j];
code = list_codes[j];

// 1. 解码 PQ
pq.decode(code, residual);  // 残差 → 128维

// 2. 加回质心（如果 by_residual=true）
centroid = quantizer->reconstruct(list_no);
vector = residual + centroid;  // 近似还原
```

---

## Conversation 15: Disk Storage Format

**User**: 如果将索引保存到磁盘，其存储格式是怎样的？

**Assistant**: 
## IndexIVFPQ 磁盘存储格式

### 整体结构 (index_write.cpp:777-786)

```
┌─────────────────────────────────────────────────────────────┐
│  Magic Number (4字节): "IwPQ"                               │
├─────────────────────────────────────────────────────────────┤
│  Index Header                                               │
│    - d, ntotal, is_trained, metric_type                     │
├─────────────────────────────────────────────────────────────┤
│  IVF Header                                                 │
│    - nlist, nprobe                                          │
│    - Quantizer (IndexFlat): 一级质心                         │
│    - DirectMap                                              │
├─────────────────────────────────────────────────────────────┤
│  IndexIVFPQ 特有                                            │
│    - by_residual, code_size                                 │
│    - ProductQuantizer: 二级质心                              │
│    - InvertedLists: 倒排列表数据                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 详细格式

#### 1. Magic Number (4字节)
```
"IwPQ" - IndexIVFPQ
"IwQR" - IndexIVFPQR (带refine)
```

#### 2. Index Header (index_write.cpp:94-105)
```
d            : int64   (向量维度)
ntotal       : int64   (总向量数)
dummy × 2    : int64   (占位)
is_trained   : bool    (是否已训练)
metric_type  : int     (距离类型)
metric_arg   : float   (如果 metric_type > 1)
```

#### 3. IVF Header (index_write.cpp:444-452)
```
nlist        : size_t  (聚类数量)
nprobe       : size_t  (搜索探测数)
quantizer    : Index   (一级量化器，存储 nlist 个 d 维质心)
direct_map   : DirectMap
```

#### 4. IndexIVFPQ 特有 (index_write.cpp:783-786)
```
by_residual  : bool    (是否使用残差)
code_size    : size_t  (PQ 编码大小 = M)
ProductQuantizer:
    d        : size_t  (向量维度)
    M        : size_t  (子空间数量)
    nbits    : size_t  (每子空间位数)
    centroids: float[] (M × ksub × dsub 个质心)
```

#### 5. InvertedLists (index_write.cpp:297-337)
```
"ilar"       : 4字节
nlist        : size_t
code_size    : size_t
list_type    : "full" 或 "sprs"

对每个非空列表:
    codes[]  : uint8[] (n × code_size)  ← PQ 编码
    ids[]    : int64[] (n)              ← 向量 ID
```

---

### 存储大小估算
```
d=128, nlist=1000, M=8, nbits=8, ntotal=1M

1. 一级质心 (quantizer):
   nlist × d × 4 = 1000 × 128 × 4 = 512 KB

2. 二级质心:
   M × ksub × dsub × 4 = 8 × 256 × 16 × 4 = 128 KB

3. 倒排列表:
   ntotal × (code_size + sizeof(id))
   = 1M × (8 + 8) = 16 MB

总计: ~16.6 MB (vs 原始 512 MB)
压缩比: ~30x
```

---

### Python 保存/加载

```python
import faiss

# 保存
faiss.write_index(index, "index.ivfpq")

# 加载
index = faiss.read_index("index.ivfpq")

# 内存映射加载 (大数据)
index = faiss.read_index("index.ivfpq", faiss.IO_FLAG_MMAP)
```

---

### 存储内容总结

| 组件 | 存储内容 | 大小 |
|------|---------|------|
| quantizer | nlist 个 d 维质心 | nlist × d × 4 |
| pq.centroids | M×ksub 个 d/M 维质心 | M × ksub × dsub × 4 |
| invlists.codes | PQ 编码 | ntotal × M |
| invlists.ids | 向量 ID | ntotal × 8 |

---

## Conversation 16: Are Original Vectors Stored?

**User**: 原始向量在这里不进行保存吗？

**Assistant**: **不保存。**

IndexIVFPQ 只保存：
1. **向量 ID** (8字节)
2. **PQ 编码** (M字节，如8字节)

**不保存原始向量**（128×4=512字节）。

### 为什么？
- **压缩目的**：节省存储空间
- **压缩比**：512字节 → 16字节（ID+PQ码），约32倍压缩

### 如何还原？
```python
# 近似还原（有损）
pq_code = invlists.get_code(list_no, offset)
id = invlists.get_id(list_no, offset)

# 1. 解码 PQ
residual = pq.decode(pq_code)

# 2. 加回质心
centroid = quantizer.reconstruct(list_no)
approx_vector = residual + centroid  # 近似值
```

### 对比其他索引

| 索引类型 | 存储内容 | 每向量大小 |
|---------|---------|-----------|
| IndexFlat | 完整向量 | 512字节 |
| IndexIVFFlat | ID + 完整向量 | 520字节 |
| IndexIVFPQ | ID + PQ编码 | 16字节 |

---

## Conversation 17: IndexIVFPQ Search Process

**User**: 接下来介绍下IndexIVFPQ怎么做检索

**Assistant**: 
## IndexIVFPQ 检索流程

### 整体流程图

```
IndexIVFPQ::search(query, k)
         │
         ▼
┌─────────────────────────────┐
│ Step 1: 粗量化搜索           │
│ quantizer->search(query)    │ ──► 找到最近的 nprobe 个质心
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Step 2: 遍历 nprobe 个列表   │
│ for each list in top-nprobe │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Step 3: 计算距离表           │
│ precompute_list_tables()    │ ──► 准备 PQ 距离查表
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Step 4: 扫描倒排列表         │
│ scan_codes()                │ ──► 计算每个编码的距离
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Step 5: 维护 Top-K 堆        │
│ heap_addn()                 │ ──► 更新最近邻结果
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Step 6: 堆排序输出           │
│ heap_reorder()              │ ──► 返回排序后的结果
└─────────────────────────────┘
```

---

### 详细步骤

#### Step 1: 粗量化搜索 (IndexIVF.cpp:333-339)

```cpp
quantizer->search(sub_n, sub_x, cur_nprobe, coarse_dis, idx);

// 输出:
//   idx[i * nprobe + j] = 第i个查询的第j近的质心编号
//   coarse_dis[i * nprobe + j] = 到该质心的距离
```

#### Step 2-5: 核心搜索循环 (IndexIVF.cpp:601-637)

```cpp
for (idx_t i = 0; i < n; i++) {           // 遍历每个查询
    scanner->set_query(x + i * d);        // 设置查询向量
    
    init_result(simi, idxi);              // 初始化堆
    
    for (idx_t ik = 0; ik < cur_nprobe; ik++) {  // 遍历 nprobe 个列表
        scan_one_list(
            keys[i * cur_nprobe + ik],    // 列表编号
            coarse_dis[i * cur_nprobe + ik],  // 粗距离
            simi, idxi,                   // 结果堆
            cur_max_codes - nscan         // 最大扫描数
        );
    }
    
    reorder_result(simi, idxi);           // 排序输出
}
```

#### scan_one_list 内部 (IndexIVF.cpp:517-595)

```cpp
auto scan_one_list = [&](idx_t key, float coarse_dis_i, ...) {
    scanner->set_list(key, coarse_dis_i);  // 准备该列表的距离表
    
    size_t list_size = invlists->list_size(key);
    const uint8_t* codes = invlists->get_codes(key);  // PQ 编码
    const idx_t* ids = invlists->get_ids(key);        // 向量 ID
    
    scanner->scan_codes(list_size, codes, ids, simi, idxi, k);
};
```

---

### IndexIVFPQ 特有的扫描流程

#### scanner->set_query() (IndexIVFPQ.cpp:562-583)

```cpp
void init_query(const float* qi) {
    if (metric_type == METRIC_INNER_PRODUCT) {
        // IP: 预计算内积表
        pq.compute_inner_prod_table(qi, sim_table);
    } else {
        // L2: 预计算距离表
        if (!by_residual) {
            pq.compute_distance_table(qi, sim_table);
        } else if (use_precomputed_table) {
            pq.compute_inner_prod_table(qi, sim_table_2);
        }
    }
}
```

#### scanner->set_list() (IndexIVFPQ.cpp:599-610)

```cpp
float precompute_list_tables() {
    if (by_residual && metric_type == METRIC_L2) {
        if (use_precomputed_table == 0) {
            // 计算残差 + 距离表
            quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_distance_table(residual_vec, sim_table);
        } else if (use_precomputed_table == 1) {
            // 使用预计算表
            dis0 = coarse_dis;
            fvec_madd(pq.M * pq.ksub, 
                      precomputed_table[key], -2.0, sim_table_2, sim_table);
        }
    }
    return dis0;
}
```

#### scanner->scan_codes() (IndexIVFPQ.cpp:1227-1251)

```cpp
size_t scan_codes(size_t ncode, const uint8_t* codes, const idx_t* ids, ...) {
    if (polysemous_ht > 0) {
        // 多义搜索：汉明过滤 + PQ
        scan_list_polysemous(ncode, codes, res);
    } else if (precompute_mode == 2) {
        // 使用预计算表
        scan_list_with_table(ncode, codes, res);
    } else if (precompute_mode == 1) {
        // 使用表指针
        scan_list_with_pointer(ncode, codes, res);
    } else {
        // 实时计算
        scan_on_the_fly_dist(ncode, codes, res);
    }
}
```

#### scan_list_with_table() (IndexIVFPQ.cpp:861-932)

```cpp
void scan_list_with_table(size_t ncode, const uint8_t* codes, SearchResultType& res) {
    for (size_t j = 0; j < ncode; j++) {
        // 查表计算距离
        float dis = dis0 + PQCodeDist::distance_single_code(
            pq.M, pq.nbits, sim_table, codes + j * code_size);
        
        // 更新堆
        res.add(j, dis);
    }
}
```

---

### 距离计算详解 (by_residual=true, L2)

```
查询向量 q
粗质心 c = quantizer[key]
残差 r = q - c

距离表: sim_table[m][j] = ||r_m - pq_centroid[m][j]||²

对每个编码 code = (c0, c1, ..., cM-1):
    distance = dis0 + Σ sim_table[m][code[m]]
             = ||q - c||² + Σ ||r_m - pq_centroid[m][code[m]]||²
```

---

### 检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| nprobe | 1 | 搜索的聚类数量 |
| max_codes | 0 | 最大扫描编码数 (0=无限制) |
| parallel_mode | 0 | 并行模式 |
| polysemous_ht | 0 | 多义汉明阈值 (0=禁用) |

---

### Python 示例

```python
import faiss

# 创建并训练索引
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist=100, M=8, nbits=8)
index.train(data)
index.add(data)

# 搜索
index.nprobe = 10  # 搜索 10 个聚类
distances, labels = index.search(query, k=5)
```

---

### 性能优化点

| 技术 | 作用 |
|------|------|
| nprobe | 平衡精度与速度 |
| 预计算表 | 避免重复计算 |
| 4路展开 | 减少分支开销 |
| 多义搜索 | 汉明距离快速过滤 |
| OpenMP | 多线程并行 |

---

## Conversation 18: Coarse Search Algorithm

**User**: 检索的第一步是在nlist中找到nprobe个list是吗？这里如何确定nprobe个向量，用的什么算法

**Assistant**: **是的，是在 nlist 个质心中找到最近的 nprobe 个。**

### 代码位置 (IndexIVF.cpp:333-339)

```cpp
quantizer->search(sub_n, sub_x, cur_nprobe, coarse_dis, idx);
```

### quantizer 是什么？

默认是 **IndexFlat**（暴力搜索），存储 nlist 个 d 维质心。

```cpp
// 典型创建方式
quantizer = faiss.IndexFlatL2(d)  // Python
```

### 算法：暴力搜索 + Top-K

```
Step 1: 计算查询 q 到所有 nlist 个质心的距离
        dist[i] = ||q - centroid[i]||²   (i = 0, 1, ..., nlist-1)

Step 2: 排序，取最小的 nprobe 个
        top-nprobe indices → idx[]
        top-nprobe distances → coarse_dis[]
```

### IndexFlat::search 实现

### 算法选择 (ResultHandler.h:734-744)

```cpp
if (k == 1) {
    Top1BlockResultHandler      // 特殊优化：只找最近的 1 个
} else if (k < 100) {
    HeapBlockResultHandler      // 堆算法：维护大小为 k 的最大堆
} else {
    ReservoirBlockResultHandler // 蓄水池算法：适合大 k
}
```

---

### 三种算法详解

#### 1. k = 1：Top1 (单次遍历)
```
遍历 nlist 个质心，记录最小距离的索引
时间复杂度: O(nlist)
```

#### 2. k < 100：堆算法
```
维护一个大小为 k 的最大堆

for each centroid in nlist:
    dis = distance(query, centroid)
    if dis < heap.max:  // 比堆顶小
        heap.replace_top(dis, id)  // 替换堆顶

时间复杂度: O(nlist × log k)
```

#### 3. k >= 100：蓄水池算法
```
维护一个大小为 2k 的缓冲区

for each centroid in nlist:
    add_to_reservoir(dis, id)
    if reservoir full:
        partition around median  // 分区，保留最小的 2k 个

最后从 reservoir 中选出最小的 k 个

时间复杂度: O(nlist) (摊还)
```

---

### 阈值定义 (distances.cpp:590)

```cpp
int distance_compute_min_k_reservoir = 100;
```

---

### 总结

| nprobe | 算法 | 复杂度 | 原因 |
|--------|------|--------|------|
| 1 | Top1 | O(nlist) | 单次遍历足够 |
| 2-99 | 堆 | O(nlist × log nprobe) | 小 k 时堆效率高 |
| ≥100 | 蓄水池 | O(nlist) | 大 k 时蓄水池更优 |

### 实际流程
```
查询 q (d维)
    │
    ▼
计算 q 到所有 nlist 个质心的距离
    │
    ▼
根据 nprobe 选择算法
    │
    ├─ nprobe=1  → Top1: 找最小距离
    ├─ nprobe<100 → 堆: 维护 top-nprobe
    └─ nprobe≥100 → 蓄水池: 采样+分区
    │
    ▼
返回: nprobe 个最近的质心索引 + 距离
```

---

## Conversation 19: Reservoir Algorithm Accuracy

**User**: 蓄水池算法能保证精准的top nrpobe吗

**Assistant**: **是的，能保证精准的 top-nprobe。**

这里的"蓄水池算法"不是传统的随机采样，而是 **ReservoirTopN**（确定性选择算法）。

### ReservoirTopN 工作原理 (ResultHandler.h:394-459)

```cpp
struct ReservoirTopN {
    size_t n;        // 需要的 top-n
    size_t capacity; // 缓冲区大小 (= 2n)
    T* vals;         // 距离数组
    TI* ids;         // ID 数组
    
    bool add_result(T val, TI id) {
        if (C::cmp(threshold, val)) {  // 比阈值更优
            if (i == capacity) {
                shrink_fuzzy();  // 缓冲区满，执行分区
            }
            vals[i] = val;
            ids[i] = id;
            i++;
        }
    }
    
    void shrink_fuzzy() {
        // 使用 partition_fuzzy 精确分区
        threshold = partition_fuzzy<C>(vals, ids, capacity, n, (capacity + n) / 2, &i);
        // 分区后：前 i 个元素就是精确的 top-i（i 在 n 到 capacity/2 之间）
    }
};
```

### partition_fuzzy 算法 (partitioning.cpp:121-217)

```cpp
// 基于 QuickSelect 的变种
// 找到一个阈值 thresh，使得：
//   - 恰好 q_min 到 q_max 个元素 < thresh
//   - 保留这些元素，丢弃其他

T partition_fuzzy_median3(T* vals, TI* ids, size_t n, size_t q_min, size_t q_max) {
    // 二分搜索找阈值
    for (int it = 0; it < 200; it++) {
        count_lt_and_eq(vals, n, thresh, n_lt, n_eq);
        
        if (n_lt <= q_min && n_lt + n_eq >= q_min) {
            break;  // 找到合适的阈值
        }
        
        // 调整阈值
        if (n_lt < q_min) thresh_inf = thresh;
        else thresh_sup = thresh;
        
        thresh = sample_threshold_median3(vals, n, thresh_inf, thresh_sup);
    }
    
    // 压缩数组，保留 < thresh 的元素
    compress_array(vals, ids, n, thresh, n_eq_1);
    
    return thresh;
}
```

---

### 对比三种算法的精度

| 算法 | 精度 | 时间复杂度 |
|------|------|-----------|
| Top1 | 精确 | O(nlist) |
| 堆 | 精确 | O(nlist × log k) |
| ReservoirTopN | **精确** | O(nlist) |

### ReservoirTopN 的优势
```
当 k 较大时（如 k=1000）:
  - 堆算法: 每次插入 O(log 1000) ≈ 10 次比较
  - ReservoirTopN: 分区操作 O(2k)，但摊还后接近 O(1)

所以 k >= 100 时，ReservoirTopN 更快
```

---

### 总结
```
ReservoirTopN ≠ 随机采样

它是:
  1. 确定性算法
  2. 基于 QuickSelect 的精确 top-k
  3. 时间复杂度 O(nlist)，比堆更快（大 k 时）
  4. 结果与堆算法完全一致
```

---

## Conversation 20: Alternative Quantizer Types

**User**: 第一阶段的quantizer可以选用除indexFlat之外的索引类型吗？比如indexHNSW

**Assistant**: **可以。** quantizer 的类型是 `Index*`，可以使用任何 Index 子类。

### 代码证据

```cpp
// Level1Quantizer::quantizer 定义 (IndexIVF.h)
Index* quantizer;  // 可以是任何 Index 子类
```

### 实际使用案例 (distributed_ondisk/make_trained_index.py:95)

```cpp
quantizer = faiss.IndexHNSWFlat(d, 32)
```

### Python 示例

```python
import faiss

# 使用 IndexHNSW 作为 quantizer
quantizer = faiss.IndexHNSWFlat(d, 32)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

# 或使用 IndexIVFFlat
quantizer = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
```

---

### 常用 quantizer 类型对比

| quantizer 类型 | 搜索复杂度 | 优点 | 缺点 |
|----------------|------------|------|------|
| **IndexFlat** | O(nlist) | 简单、精确 | 慢（nlist 大时） |
| **IndexIVF** | O(nprobe) | 快速 | 需要训练二级量化器 |
| **IndexHNSW** | O(log nlist) | 最快 | 内存占用大 |
| **IndexPQ** | O(nlist) | 内存小 | 精度有损失 |

---

### 注意事项

1. **训练时**：quantizer 会自动训练（或使用已训练的）

```cpp
// IndexIVF.cpp:67-89
if (quantizer_trains_alone == 0) {
    Clustering clus(d, nlist, cp);
    clus.train(n, x, *quantizer);  // 用 quantizer 做 k-means
}
```

2. **必须保证 quantizer->ntotal == nlist**

```cpp
// IndexIVF.cpp:74-75
FAISS_THROW_IF_NOT_MSG(
    quantizer->ntotal == nlist,
    "nlist not consistent with quantizer size");
```

---

### 使用 IndexHNSW 的优势
```
nlist = 10000 时:

IndexFlat:
  - 搜索 10000 个质心，O(10000)

IndexHNSW:
  - 搜索 10000 个质心，O(log 10000) ≈ 13

速度提升: 10000 / 13 ≈ 769 倍
```

### 总结

| 问题 | 答案 |
|------|------|
| 可以用 IndexHNSW 吗？ | ✅ 可以 |
| 可以用 IndexIVF 吗？ | ✅ 可以 |
| 可以用 IndexPQ 吗？ | ✅ 可以 |
| 必须满足的条件 | quantizer->ntotal == nlist |

---

## Conversation 21: Getting PQ Codes from Lists

**User**: 找到nprobe之后，如何获取这些list中的哭向量的pq code?

**Assistant**: 
## 获取 nprobe 个列表中的 PQ Code

### 核心代码 (IndexIVF.cpp:553-584)

```cpp
// 遍历 nprobe 个列表
for (idx_t ik = 0; ik < cur_nprobe; ik++) {
    idx_t key = keys[i * cur_nprobe + ik];  // 列表编号
    
    // 1. 获取列表大小
    size_t list_size = invlists->list_size(key);
    
    // 2. 获取 PQ codes
    InvertedLists::ScopedCodes scodes(invlists, key);
    const uint8_t* codes = scodes.get();
    
    // 3. 获取 IDs
    InvertedLists::ScopedIds sids(invlists, key);
    const idx_t* ids = sids.get();
    
    // 4. 扫描计算距离
    scanner->scan_codes(list_size, codes, ids, simi, idxi, k);
}
```

---

### InvertedLists 核心接口 (InvertedLists.h:59-86)

```cpp
// 1. 获取列表大小
virtual size_t list_size(size_t list_no) const = 0;

// 2. 获取 PQ codes（连续存储）
// 返回: list_size * code_size 字节
virtual const uint8_t* get_codes(size_t list_no) const = 0;

// 3. 获取 IDs（连续存储）
// 返回: list_size 个 int64
virtual const idx_t* get_ids(size_t list_no) const = 0;

// 4. 获取单个 code
virtual const uint8_t* get_single_code(size_t list_no, size_t offset) const;

// 5. 释放资源
virtual void release_codes(size_t list_no, const uint8_t* codes) const;
virtual void release_ids(size_t list_no, const idx_t* ids) const;
```

---

### 数据布局

```
倒排列表 #23 (list_size = 3, code_size = 8):

codes (uint8数组):
┌────────┬────────┬────────┐
│ code_0 │ code_1 │ code_2 │  每个 8 字节
│ 8字节  │ 8字节  │ 8字节  │
└────────┴────────┴────────┘
地址: codes + 0*8, codes + 1*8, codes + 2*8

ids (int64数组):
┌───────┬───────┬───────┐
│ id_0  │ id_1  │ id_2  │  每个 8 字节
└───────┴───────┴───────┘
地址: ids + 0, ids + 1, ids + 2
```

---

### 安全访问方式：ScopedCodes / ScopedIds (InvertedLists.h:201-242)

```cpp
// RAII 包装，自动释放资源
struct ScopedCodes {
    const InvertedLists* il;
    const uint8_t* codes;
    size_t list_no;
    
    ScopedCodes(const InvertedLists* il_, size_t list_no_)
        : il(il_), codes(il_->get_codes(list_no_)), list_no(list_no_) {}
    
    const uint8_t* get() { return codes; }
    
    ~ScopedCodes() { il->release_codes(list_no, codes); }  // 自动释放
};
```

---

### 完整流程
```
查询 q
    │
    ▼
quantizer->search(q, nprobe)
    │
    ▼
得到 nprobe 个列表编号: keys[0..nprobe-1]
    │
    ▼
for each key in keys:
    │
    ├─► list_size = invlists->list_size(key)
    │
    ├─► codes = invlists->get_codes(key)
    │      └─► 返回 list_size * code_size 字节
    │
    ├─► ids = invlists->get_ids(key)
    │      └─► 返回 list_size 个 int64
    │
    └─► scanner->scan_codes(list_size, codes, ids, ...)
           │
           └─► for j in 0..list_size:
                   code = codes + j * code_size
                   id = ids[j]
                   dis = compute_distance(code)
                   update_heap(dis, id)
```

---

### Python 访问方式

```python
import faiss

index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
# ... 训练和添加 ...

# 获取列表 #10 的信息
list_no = 10
list_size = index.invlists.list_size(list_no)
ids = faiss.rev_swig_ptr(index.invlists.get_ids(list_no), list_size)
codes = faiss.rev_swig_ptr(index.invlists.get_codes(list_no), list_size * index.code_size)
```

---

### ArrayInvertedLists 实现 (InvertedLists.h:246-278)

```cpp
struct ArrayInvertedLists : InvertedLists {
    std::vector<std::vector<uint8_t>> codes;  // codes[list_no][offset * code_size]
    std::vector<std::vector<idx_t>> ids;      // ids[list_no][offset]
    
    const uint8_t* get_codes(size_t list_no) const override {
        return codes[list_no].data();
    }
    
    const idx_t* get_ids(size_t list_no) const override {
        return ids[list_no].data();
    }
};
```

---

### 总结

| 操作 | API | 返回值 |
|------|-----|--------|
| 获取列表大小 | `invlists->list_size(list_no)` | `size_t` |
| 获取 PQ codes | `invlists->get_codes(list_no)` | `const uint8_t*` |
| 获取 IDs | `invlists->get_ids(list_no)` | `const idx_t*` |
| 安全访问 | `ScopedCodes scodes(invlists, list_no)` | RAII 包装 |

---

## Conversation 22: IDs and PQ Codes Storage Order

**User**: 意思是同一个list的ids和pq code存储在一起，并且ids连续存储，pq code也连续存储吗？那个id的顺序和pq code的顺序也是对应的吗？

**Assistant**: **是的，顺序严格对应。**

### ArrayInvertedLists 存储 (InvertedLists.h:246-248)

```cpp
struct ArrayInvertedLists : InvertedLists {
    std::vector<std::vector<uint8_t>> codes;  // codes[list_no] 是连续数组
    std::vector<std::vector<idx_t>> ids;      // ids[list_no] 是连续数组
};
```

### 布局图
```
ids[0]  ──┐
          │ 对应
codes[0] ─┘

ids[1]  ──┐
          │ 对应
codes[1] ─┘

...

ids[j]  ──┐
          │ 对应
codes[j] ─┘
```

### 具体例子
```
列表 #23 (list_size=3, code_size=8):

ids[23] (vector<idx_t>):
┌───────┬───────┬───────┐
│ 1042  │ 5681  │ 12033 │  ← 连续存储
└───────┴───────┴───────┘
  ids[0]  ids[1]  ids[2]

codes[23] (vector<uint8_t>):
┌────────────────┬────────────────┬────────────────┐
│ [0x3a,0x7f...] │ [0x12,0x45...] │ [0x8c,0x21...] │  ← 连续存储
│    8 字节      │    8 字节      │    8 字节      │
└────────────────┴────────────────┴────────────────┘
   codes[0:8]      codes[8:16]     codes[16:24]
      ↓               ↓               ↓
   对应 id[0]      对应 id[1]      对应 id[2]
```

### 添加时保证顺序 (ArrayInvertedLists.cpp)

```cpp
size_t ArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    
    // 同时追加到两个数组
    for (size_t i = 0; i < n_entry; i++) {
        ids[list_no].push_back(ids_in[i]);
    }
    
    size_t o = codes[list_no].size();
    codes[list_no].resize(o + n_entry * code_size);
    memcpy(&codes[list_no][o], code, n_entry * code_size);
    
    return ids[list_no].size() - n_entry;  // 返回起始偏移
}
```

### 访问时的对应关系

```cpp
// 第 j 个向量
idx_t id = ids[list_no][j];
const uint8_t* code = &codes[list_no][j * code_size];

// 两者对应同一个向量
```

### 总结

| 问题 | 答案 |
|------|------|
| ids 连续存储？ | ✅ 是，`vector<idx_t>` |
| codes 连续存储？ | ✅ 是，`vector<uint8_t>` |
| ids 和 codes 在一起？ | ❌ 分开两个数组 |
| 顺序对应？ | ✅ 是，`ids[j]` ↔ `codes[j*code_size : (j+1)*code_size]` |

---

## Conversation 23: Distance Calculation in Residual Mode

**User**: 在残差模式下，如何计算query向量与库向量的距离？

**Assistant**: 
## 残差模式下距离计算

### 核心公式 (IndexIVFPQ.cpp:349-377)

```
查询向量: x (128维)
库向量:   y = y_C + y_R
          y_C = 粗质心 (128维，由 list_no 决定)
          y_R = 残差 (128维，由 PQ code 解码)

距离: d = ||x - y||² = ||x - y_C - y_R||²

展开:
d = ||x - y_C||² + ||y_R||² + 2*(y_C·y_R) - 2*(x·y_R)
    term1          term2       term3         term4
```

---

### 四个分解项

| 项 | 公式 | 说明 |
|----|------|------|
| term1 | `||x - y_C||²` | 查询到粗质心的距离（粗搜索已计算） |
| term2 | `||y_R||²` | PQ残差的模²（可预计算） |
| term3 | `2 * (y_C · y_R)` | 粗质心与残差的内积（可预计算） |
| term4 | `-2 * (x · y_R)` | 查询与残差的内积（查询时计算） |

---

### 计算策略

**预计算表** = term2 + term3（训练时计算，按 list_no 索引）

```
precomputed_table[list_no][m][j] = ||pq_centroid[m][j]||² + 2 * (coarse_centroid[list_no] · pq_centroid[m][j])
```

**查询表** = term4 相关（查询时计算）

```
query_table[m][j] = x · pq_centroid[m][j]
```

**最终距离表** = 预计算表 - 2 × 查询表

```
sim_table[list_no] = precomputed_table[list_no] - 2 * query_table
```

---

### 搜索时三步走

**Step 1: 粗搜索**
- 用 quantizer 找最近的 nprobe 个粗质心
- 得到 term1（粗距离）和 list_no

**Step 2: 准备距离表**
- 从预计算表取出 term2 + term3（按 list_no 查）
- 计算查询表（term4 相关）
- 合并：sim_table = 预计算表 - 2 × 查询表

**Step 3: 查表计算**
- 对每个 PQ code [c0, c1, ..., c7]
- 查 8 次 sim_table，累加
- 距离 = term1 + 查表结果

---

### 三种表

| 表 | 数量 | 大小 | 内容 |
|----|------|------|------|
| 预计算表 | nlist 张 | M × ksub | term2 + term3（与粗质心相关） |
| 查询表 | 每个查询 1 张 | M × ksub | term4 相关（与查询相关） |
| 最终距离表 | 每个列表 1 张 | M × ksub | 合并后用于查表 |

---

### 为什么高效？

1. **查表代替向量运算**：8 次查表 vs 128 维向量运算
2. **预计算**：term2 + term3 训练时算好
3. **粗搜索**：只扫描 nprobe 个列表

---

### 总结

**残差模式 = 向量分解 + 公式展开 + 预计算 + 查表**

核心优化：用 **8 次查表** 替代 **128 维向量运算**！

---

## Conversation 24: by_residual=false 时的 PQ 中心表

**User**: 以 by_residual 为 false 为前提。这种情况下是否就只有一个 pq 中心表了？

**Assistant**: 
是的。

当 `by_residual = false` 时：
- **只有 1 个 PQ 中心表**（`ProductQuantizer::centroids`），用于编码**原始向量**
- 粗量化中心（nlist 个）仅用于**向量分配到桶**，不参与 PQ 编码
- `precompute_table = false`，查询时不需要预计算 nlist 个表

对比：

| | by_residual=true | by_residual=false |
|---|---|---|
| PQ 编码对象 | 残差 (x - 粗中心) | 原始向量 x |
| PQ 中心表数量 | 1 | 1 |
| precompute_table | true (nlist个表) | false |

---

## Conversation 25: get_codes/get_ids 接口位置

**User**: 后续会话仍以 by_residual=false 为前提。在检索向量，获取到 nprobe 个 list 后，应该需要将 list 中的 ids 和 pq codes 从磁盘读取到内存，这部分操作的代码在哪？

**Assistant**: 
在 `InvertedLists.h:66-73` 定义了接口。具体实现取决于使用的 InvertedLists 类型：

**从磁盘读取**：`OnDiskInvertedLists.cpp:385-393`

```cpp
// OnDiskInvertedLists.cpp:385-400
const uint8_t* OnDiskInvertedLists::get_codes(size_t list_no) const {
    if (lists[list_no].offset == INVALID_OFFSET) {
        return nullptr;
    }
    return ptr + lists[list_no].offset;
}

const idx_t* OnDiskInvertedLists::get_ids(size_t list_no) const {
    if (lists[list_no].offset == INVALID_OFFSET) {
        return nullptr;
    }
    return (const idx_t*)(ptr + lists[list_no].offset + 
                          code_size * lists[list_no].capacity);
}
```

**关键**：`OnDiskInvertedLists` 使用 **mmap** 将文件映射到虚拟内存：

| 操作 | 位置 |
|------|------|
| 接口定义 | `InvertedLists.h:66-73` |
| OnDisk 实现 | `OnDiskInvertedLists.cpp:385-400` |
| mmap 初始化 | `OnDiskInvertedLists.cpp:268-289` |

`get_codes/get_ids` **不会立即读取数据**，只返回 mmap 指针。实际数据由 OS 在访问时按需 **page in**。

---

## Conversation 26: IndexIVFPQ search 调用 get_codes/get_ids 的时机

**User**: IndexIVFPQ 的 search 何时调用到这个接口？

**Assistant**: 
**调用链**：

```
IndexIVF::search (IndexIVF.cpp:304)
    └─► quantizer->search()                    // 粗量化，找 nprobe 个桶
    └─► search_preassigned() (IndexIVF.cpp:344)
            └─► scan_one_list() (IndexIVF.cpp:517)
                    ├─► ScopedCodes(invlists, key) → get_codes()  // 558行
                    ├─► ScopedIds(invlists, key) → get_ids()      // 565行
                    └─► scanner->scan_codes(list_size, codes, ids, ...)  // 583行
```

**关键代码** `IndexIVF.cpp:558-584`：

```cpp
InvertedLists::ScopedCodes scodes(invlists, key);  // → get_codes()
const uint8_t* codes = scodes.get();

sids = std::make_unique<InvertedLists::ScopedIds>(invlists, key);  // → get_ids()
const idx_t* ids = sids->get();

scanner->scan_codes(list_size, codes, ids, simi, idxi, k);
```

`ScopedCodes/ScopedIds` 构造函数内部调用 `get_codes/get_ids`（`InvertedLists.h:207,228`）。

---

## Conversation 27: OnDiskInvertedLists 继承关系

**User**: OnDiskInvertedLists 是 InvertedList 的子类吗？

**Assistant**: 
是的。

`OnDiskInvertedLists` **直接继承自 `InvertedLists`**：

```cpp
// OnDiskInvertedLists.h:60
struct OnDiskInvertedLists : InvertedLists {
```

继承关系：

```
InvertedLists (基类)
    ├─► ReadOnlyInvertedLists
    ├─► ArrayInvertedLists
    ├─► OnDiskInvertedLists      ← 直接继承
    ├─► BlockInvertedLists
    └─► ...其他
```

`OnDiskInvertedLists` 实现了完整的读写接口（`add_entries`, `update_entries`, `resize`），所以它不是只读的。

---

## Conversation 28: 华为 C++ 编程风格偏好设置

**User**: 你知道华为 C++ 的编程风格吗？知道的话，我需要你以后都按华为风格进行 C/C++ 编程。可以将这一指令保存到偏好里。

**Assistant**: 
是的，华为 C++ 编程风格主要包括：

**命名规范**：
- 类名：大驼峰（`MyClass`）
- 函数名：大驼峰（`GetValue`）
- 变量/参数：小驼峰（`totalCount`）
- 成员变量：小驼峰+下划线后缀（`memberVar_`）
- 常量/宏：全大写+下划线（`MAX_SIZE`）

**代码格式**：
- 缩进：4空格
- 大括号：K&R风格（左括号不换行）
- 行宽：不超过120字符

**其他规范**：
- 头文件包含顺序：系统头文件 → 标准库 → 第三方库 → 项目头文件
- 优先使用 `const`、`constexpr`
- 避免裸指针，优先智能指针
- RAII 资源管理

创建了 `/home/yzc/.opencode.md` 文件保存偏好。

---

## Conversation 29: 创建 IndexIVFPQ 测试项目

**User**: 在 `/home/yzc/faiss` 目录下新建一个 `test_ivfpq` 目录，实现 C++ 测试项目：
1. 创建 float 向量函数，维度可配置
2. 创建 IndexIVFPQ 索引，一阶段量化使用 IndexHNSW
3. 测试完整流程：train, add, 保存, 加载, 检索, 重排
4. 使用 INI 配置文件

**Assistant**: 
创建了完整项目：

```
/home/yzc/faiss/test_ivfpq/
├── CMakeLists.txt      # CMake 配置
├── Config.h/cpp        # INI 配置读写
├── VectorGenerator.h/cpp  # 随机向量生成
├── IndexTest.h/cpp     # IndexIVFPQ 测试流程
├── main.cpp            # 入口
└── README.md           # 说明文档
```

**编译运行**：
```bash
cd /home/yzc/faiss/test_ivfpq/build
cmake ..
make -j$(nproc)
LD_LIBRARY_PATH=/home/yzc/faiss/buildDebug/faiss:$LD_LIBRARY_PATH ./test_ivfpq -g   # 生成配置
LD_LIBRARY_PATH=/home/yzc/faiss/buildDebug/faiss:$LD_LIBRARY_PATH ./test_ivfpq -c config.ini  # 运行
```

---

## Conversation 30: 使用 IndexHNSWFlat 做一级量化器

**User**: 能不能改用 IndexHNSWFlat 做一级量化器？

**Assistant**: 
可以。关键修改：

1. 使用 `IndexHNSWFlat` 代替 `IndexFlat`
2. 设置 `quantizer_trains_alone = 2`

```cpp
quantizer_ = new faiss::IndexHNSWFlat(config_.dimension_, config_.hnswM_);
quantizer_->hnsw.efSearch = config_.hnswEfSearch_;

index_ = new faiss::IndexIVFPQ(quantizer_, config_.dimension_, 
                                config_.nlist_, config_.m_, config_.nbits_);
index_->quantizer_trains_alone = 2;  // 关键！
```

**`quantizer_trains_alone` 含义**：

| 值 | 含义 |
|----|------|
| 0 | 用 quantizer 做 k-means 聚类（默认，IndexFlat） |
| 1 | quantizer 自己训练（MultiIndex, ResidualCoarseQuantizer） |
| 2 | 先用 IndexFlatL2 做 k-means，再用质心训练 quantizer（IndexHNSWFlat） |

**训练流程（quantizer_trains_alone=2）**：
```
1. Clustering → 得到 nlist 个质心
2. quantizer->train(nlist, centroids)  // 训练 HNSW
3. quantizer->add(nlist, centroids)    // 添加质心到 HNSW
```

测试通过，QPS: 1785.71

---

## Conversation 31: index_factory_sub 函数解析

**User**: 浏览下当前文件的 index_factory_sub 函数，讲讲作用。

**Assistant**: 
**位置**：`faiss/index_factory.cpp:861-1126`

**作用**：通过解析字符串描述来**递归构建复合索引**，是 `index_factory` 的核心实现。

### 解析流程（从外到内）

| 优先级 | 解析内容 | 示例 |
|--------|----------|------|
| 1 | IDMap/IDMap2 | `IDMap,Flat` 或 `Flat,IDMap` |
| 2 | Refine 重排 | `IVF,PQ,RFlat` 或 `IVF,PQ,Refine(Flat)` |
| 3 | VectorTransform | `PCA128,IVF,PQ`（会改变维度 d） |
| 4 | 括号子索引 | `IVF(PQ8),Flat` → 递归解析括号内 |
| 5 | 基础索引 | `Flat`, `HNSW32`, `NSG32` |
| 6 | IVF 系列索引 | `IVF1024,PQ8` → 粗量化器 + 编码 |

### 递归调用关系

```
index_factory("PCA64,IVF1024_HNSW32,PQ8")
    └─► index_factory_sub("PCA64,IVF1024_HNSW32,PQ8")
            ├─► parse_VectorTransform("PCA64") → PCAMatrix
            └─► index_factory_sub(d=64, "IVF1024_HNSW32,PQ8")
                    ├─► parse_coarse_quantizer("IVF1024_HNSW32")
                    │       └─► new IndexHNSWFlat(32)
                    └─► parse_IndexIVF("PQ8")
                            └─► new IndexIVFPQ(quantizer, ...)
```

### 与 index_factory 的关系

```cpp
Index* index_factory(int d, const char* description, ...) {
    return index_factory_sub(d, description, ...).release();
}
```

---

## Conversation 32: index_factory 创建 IndexIVFPQ 的字符串格式

**User**: 如果想创建一个 IndexIVFPQ 对象，应该传入什么字符串？

**Assistant**: 
**基本格式**：
```
IVF<nlist>,PQ<M>x<nbits>
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `nlist` | 聚类中心数量，支持 k/M 后缀 | `IVF1024`, `IVF10k`, `IVF1M` |
| `M` | PQ 子空间数量 | `PQ8`, `PQ16` |
| `nbits` | 每个子空间的比特数（可选，默认8） | `PQ8x4`, `PQ16x10` |
| `np` | 禁用 polysemous 训练（可选） | `PQ8np` |

### 常用示例

| 字符串 | 说明 |
|--------|------|
| `IVF1024,PQ8` | nlist=1024, M=8, nbits=8（默认） |
| `IVF1024,PQ16` | nlist=1024, M=16, nbits=8 |
| `IVF100k,PQ8x4` | nlist=100000, M=8, nbits=4 |

### 使用不同量化器

| 字符串 | 量化器类型 |
|--------|------------|
| `IVF1024,PQ8` | IndexFlat（默认） |
| `IVF1024_HNSW32,PQ8` | IndexHNSWFlat (M=32) |
| `IVF1024_NSG32,PQ8` | IndexNSGFlat (R=32) |

---

## Conversation 33: PQ8 参数含义

**User**: `IVF1024_HNSW32,PQ8` 这里 pq8 表示 pq 子空间为 8 吗？

**Assistant**: 
是的。

`PQ8` 表示：
- **M = 8**：将向量分成 **8 个子空间**
- **nbits = 8**：每个子空间有 2^8 = 256 个中心点（默认值）

### 完整解析 `IVF1024_HNSW32,PQ8`

| 部分 | 含义 |
|------|------|
| `IVF1024` | nlist = 1024（1024 个聚类桶） |
| `_HNSW32` | 使用 IndexHNSWFlat 作为量化器，HNSW M=32 |
| `PQ8` | M=8 子空间，nbits=8（默认） |

### PQ 编码参数

| 写法 | M | nbits | 每子空间中心数 | code_size |
|------|---|-------|---------------|-----------|
| `PQ8` | 8 | 8 | 256 | 8 字节 |
| `PQ16` | 16 | 8 | 256 | 16 字节 |
| `PQ8x4` | 8 | 4 | 16 | 8 字节 |

**code_size = M × ceil(nbits/8)**

### 示例：128 维向量用 `PQ8`

```
原始向量: 128 维 float (512 字节)
    ↓
分成 8 个子空间，每个 16 维
    ↓
每个子空间用 8 bits (1 字节) 编码
    ↓
PQ code: 8 字节 (压缩比 64:1)
```

---

## Conversation 34: IndexIVFPQ 量化器的磁盘存储机制

**User**: IndexIVFPQ 中的一阶量化器是以指针保存在成员中的，那保存到磁盘时如何处理？以及从磁盘中读取索引时怎么分辨？从磁盘加载时会创建一个一阶量化器的实例吗？

**Assistant**: 

### 写入流程

**`index_write.cpp:444-451`**：
```cpp
static void write_ivf_header(const IndexIVF* ivf, IOWriter* f) {
    write_index_header(ivf, f);
    WRITE1(ivf->nlist);
    WRITE1(ivf->nprobe);
    write_index(ivf->quantizer, f);  // 递归写入量化器
    write_direct_map(&ivf->direct_map, f);
}
```

**`write_index()` 使用 `dynamic_cast` 判断类型**：
```cpp
// index_write.cpp:846-854
uint32_t h = dynamic_cast<const IndexHNSWFlatPanorama*>(idx) ? fourcc("IHfP")
        : dynamic_cast<const IndexHNSWFlat*>(idx)   ? fourcc("IHNf")  // ← IndexHNSWFlat
        : dynamic_cast<const IndexHNSWPQ*>(idx)     ? fourcc("IHNp")
        : dynamic_cast<const IndexFlat*>(idx)       ? fourcc("IxF2")  // ← IndexFlat
        ...
WRITE1(h);  // 写入 4 字节类型标识
// 然后写入索引具体数据
```

### 读取流程

**`index_read.cpp:1045`**：
```cpp
ivf->quantizer = read_index(f);  // 读取量化器
ivf->own_fields = true;          // IVF 拥有量化器所有权
```

**`read_index()` 根据 fourcc 创建实例**：
```cpp
// index_read.cpp:1121-1124
std::unique_ptr<Index> read_index_up(IOReader* f, int io_flags) {
    uint32_t h;
    READ1(h);  // 读取 4 字节类型标识
    
    if (h == fourcc("IxF2")) {
        idxf = std::make_unique<IndexFlatL2>();  // 创建 IndexFlat
    } else if (h == fourcc("IHNf")) {
        idxhnsw = std::make_unique<IndexHNSWFlat>();  // 创建 IndexHNSWFlat
    }
    // ... 读取索引数据
}
```

### 磁盘文件结构

```
┌─────────────────────────────────────┐
│ IndexIVFPQ Header                   │
│   - nlist, nprobe, by_residual...   │
├─────────────────────────────────────┤
│ Quantizer (嵌套索引)                 │
│   ┌─────────────────────────────┐   │
│   │ fourcc ("IHNf" / "IxF2")    │   │ ← 4 字节类型标识
│   │ IndexHNSWFlat / IndexFlat   │   │ ← 具体数据
│   │   - HNSW 图 / codes         │   │
│   └─────────────────────────────┘   │
├─────────────────────────────────────┤
│ ProductQuantizer                    │
│   - M, nbits, centroids...          │
├─────────────────────────────────────┤
│ InvertedLists                       │
│   - codes, ids...                   │
└─────────────────────────────────────┘
```

### 关键点

| 问题 | 答案 |
|------|------|
| 量化器如何保存？ | 递归调用 `write_index()`，写入类型标识 + 数据 |
| 如何分辨类型？ | 使用 **fourcc**（4 字节字符码）如 `"IHNf"` |
| 加载时创建实例？ | **是的**，`read_index()` 会 `new` 对应类型 |
| 谁拥有所有权？ | `own_fields = true`，IVF 拥有量化器 |

### 常见 fourcc 标识

| fourcc | 类型 |
|--------|------|
| `IxF2` | IndexFlatL2 |
| `IxFI` | IndexFlatIP |
| `IHNf` | IndexHNSWFlat |
| `IHNp` | IndexHNSWPQ |
| `IwPQ` | IndexIVFPQ |
| `IwQR` | IndexIVFPQR |

---

## Conversation 35: IndexIVFPQ 检索流程 (by_residual=false)

**User**: 假设 by_residual = false，IndexIVFPQ 在检索阶段怎么调用到 PQ 距离计算的？给我分析下流程和函数调用链。

**Assistant**: 

### 完整调用链

```
IndexIVFPQ::search()
    │
    ├─► quantizer->search()          // 粗量化，找 nprobe 个桶
    │
    └─► search_preassigned()
            │
            └─► scan_one_list()      // 对每个选中的桶
                    │
                    ├─► get_codes() / get_ids()   // 从倒排列表获取 PQ codes
                    │
                    └─► scanner->scan_codes()
                            │
                            └─► scan_list_with_table()
                                    │
                                    └─► PQCodeDist::distance_single_code()
                                            │
                                            └─► 查表累加
```

### 关键代码分析

#### 1. 查询初始化 `init_query_L2()` (IndexIVFPQ.cpp:577-583)

```cpp
void init_query_L2() {
    if (!by_residual) {
        // by_residual=false: 直接对查询向量计算距离表
        pq.compute_distance_table(qi, sim_table);
    } else if (use_precomputed_table) {
        // by_residual=true: 计算内积表，后续再合并预计算表
        pq.compute_inner_prod_table(qi, sim_table_2);
    }
}
```

**`compute_distance_table` 实现** (ProductQuantizer.cpp:440-467)：
```cpp
void ProductQuantizer::compute_distance_table(const float* x, float* dis_table) {
    // 对每个子空间 m，计算 ||x[m] - c||² (c ∈ ksub 个中心)
    for (size_t m = 0; m < M; m++) {
        fvec_L2sqr_ny(
            dis_table + m * ksub,  // 输出: M × ksub 的距离表
            x + m * dsub,          // 查询的第 m 个子向量
            get_centroids(m, 0),   // 第 m 个子空间的 ksub 个中心
            dsub,                  // 子空间维度
            ksub                   // 256
        );
    }
}
```

**sim_table 布局**：
```
┌─────────────────┬─────────────────┬─────────────────┐
│ 子空间 0        │ 子空间 1        │ ...  子空间 M-1 │
│ 256 个距离      │ 256 个距离      │     256 个距离  │
└─────────────────┴─────────────────┴─────────────────┘
```

#### 2. 列表初始化 `precompute_list_tables()` (IndexIVFPQ.cpp:599-611)

```cpp
float precompute_list_tables() {
    float dis0 = 0;
    if (by_residual) {
        // by_residual=true: 需要合并预计算表
        ...
    }
    // by_residual=false: dis0 = 0, sim_table 已经是最终的距离表
    return dis0;
}
```

**关键点**：`by_residual=false` 时，`dis0 = 0`，`sim_table` 在 `init_query_L2()` 中已经计算好了。

#### 3. 扫描码 `scan_codes()` (IndexIVFPQ.cpp:1227-1251)

```cpp
size_t scan_codes(size_t ncode, const uint8_t* codes, const idx_t* ids, ...) {
    if (this->polysemous_ht > 0) {
        this->scan_list_polysemous(ncode, codes, res);
    } else if (precompute_mode == 2) {
        this->scan_list_with_table(ncode, codes, res);  // ← 主要路径
    }
    return res.nup;
}
```

#### 4. 查表计算距离 `scan_list_with_table()` (IndexIVFPQ.cpp:861-932)

```cpp
template <class SearchResultType>
void scan_list_with_table(size_t ncode, const uint8_t* codes, SearchResultType& res) {
    for (size_t j = 0; j < ncode; j++) {
        // 计算 PQ code 对应的距离
        float dis = dis0 + PQCodeDist::distance_single_code(
            pq.M, pq.nbits, sim_table, codes + j * pq.code_size);
        res.add(j, dis);
    }
}
```

#### 5. 单码距离计算 `distance_single_code()` (pq_code_distance.h:63-82)

```cpp
static float distance_single_code(
        size_t M, size_t nbits,
        const float* sim_table,  // M × ksub 的距离表
        const uint8_t* code) {   // M 字节的 PQ code
    
    PQDecoderT decoder(code, nbits);
    const size_t ksub = 1 << nbits;  // 256
    const float* tab = sim_table;
    float result = 0;

    for (size_t m = 0; m < M; m++) {
        uint8_t cid = decoder.decode();   // 解码第 m 个子空间的中心 ID
        result += tab[cid];               // 查表累加
        tab += ksub;                      // 移动到下一个子空间的表
    }
    return result;
}
```

### by_residual=false vs by_residual=true 对比

| 步骤 | by_residual=false | by_residual=true |
|------|-------------------|------------------|
| `init_query_L2()` | `pq.compute_distance_table(qi, sim_table)` | `pq.compute_inner_prod_table(qi, sim_table_2)` |
| `precompute_list_tables()` | **跳过**，dis0=0 | 合并预计算表到 sim_table |
| sim_table 内容 | `‖q[m] - c‖²` (直接距离) | `term2 + term3` (需合并) |
| dis0 | 0 | `‖q - y_C‖²` |
| 最终距离 | `sum(sim_table[m][code[m]])` | `dis0 + sum(sim_table[m][code[m]])` |

### 距离计算示意

**by_residual=false 时**：
```
sim_table[m][c] = ‖q_sub[m] - centroid[m][c]‖²  (M × ksub 个值)

距离 = Σ sim_table[m][code[m]]   // M 次查表累加
```

**示例 (M=8, nbits=8)**：
```cpp
// PQ code: [c0, c1, c2, c3, c4, c5, c6, c7]  (8 字节)
// sim_table: 8 × 256 = 2048 个距离值

float dist = 0;
for (int m = 0; m < 8; m++) {
    dist += sim_table[m * 256 + code[m]];  // 8 次查表
}
```

### 总结

| 组件 | 作用 |
|------|------|
| `init_query_L2()` | 计算查询向量到所有 PQ 中心的距离表 (M × ksub) |
| `precompute_list_tables()` | by_residual=false 时跳过 |
| `scan_list_with_table()` | 遍历列表中的 PQ codes |
| `distance_single_code()` | M 次查表 + 累加得到距离 |

**核心优化**：用 **M 次查表** 替代 **d 维向量距离计算**。
