# NPU Index Implementation for Faiss

本文档描述了基于 NPU (Neural Processing Unit) 的 Faiss 索引实现，包括 NPUIndexPQ 和 NPUIndexIVFPQ。

## 目录结构

```
faiss/NPU/
├── CMakeLists.txt           # 编译配置文件
├── NPUIndexPQ.h            # NPUIndexPQ 头文件
├── NPUIndexPQ.cpp          # NPUIndexPQ 实现
├── NPUIndexIVFPQ.h         # NPUIndexIVFPQ 头文件
├── NPUIndexIVFPQ.cpp       # NPUIndexIVFPQ 实现
├── NPUResources.h          # NPU 资源管理头文件
├── NPUResources.cpp        # NPU 资源管理实现
├── NPUOps.h                # NPU 算子接口头文件
├── NPUOps.cpp              # NPU 算子实现（CPU 模拟版）
└── README_NPU.md           # 本文档
```

## 整体架构

### 1. NPUIndexPQ - 纯 PQ 索引

**继承关系：** `NPUIndexPQ` -> `IndexFlatCodes` -> `Index`

**核心功能：**
- 基于 Product Quantizer 的向量编码和检索
- 支持批量向量添加和搜索
- 支持 L2 和内积距离度量

**主要数据结构：**
```cpp
ProductQuantizer pq;          // 乘积量化器
std::vector<uint8_t> codes;   // 编码后的向量数据
```

**关键流程：**

#### 训练阶段 (train)
1. **输入：** 训练数据 `x` (n × d)
2. **步骤：**
   - 调用 `ProductQuantizer::train()` 训练子量化器
   - 将质心数据复制到 NPU 设备内存
   - 设置 `is_trained = true`
3. **输出：** 训练好的 PQ 量化器

**代码位置：** `NPUIndexPQ.cpp:86-102`

#### 添加向量 (add)
1. **输入：** 待添加向量 `x` (n × d)
2. **步骤：**
   - 计算向量的 PQ 编码
   - 将编码追加到 codes 数组
   - 更新 ntotal
   - 将编码数据复制到 NPU 设备
3. **输出：** 更新后的索引

**代码位置：** `NPUIndexPQ.cpp:104-119`

#### 搜索 (search)
1. **输入：** 查询向量 `x` (n × d)，top-k 参数
2. **步骤：**
   - 计算查询向量的距离表 (distance table)
     - L2 距离：`pq.compute_distance_table()`
     - 内积：`pq.compute_inner_prod_table()`
   - 使用 PQ 编码计算距离
   - 维护 top-k 堆结构
3. **输出：** 距离数组和标签数组

**代码位置：** `NPUIndexPQ.cpp:121-154`

### 2. NPUIndexIVFPQ - 倒排文件 PQ 索引

**继承关系：** `NPUIndexIVFPQ` -> `IndexIVF` -> `Index`

**核心功能：**
- 基于倒排文件结构的 PQ 索引
- 粗量化器 + 细量化器的两级量化
- 支持残差编码

**主要数据结构：**
```cpp
Index* quantizer;            // 粗量化器
ProductQuantizer pq;         // 细量化器（对残差编码）
InvertedLists* invlists;     // 倒排列表
```

**关键流程：**

#### 训练阶段 (train)
1. **输入：** 训练数据 `x` (n × d)
2. **步骤：**
   - 训练粗量化器（quantizer）
   - 使用粗量化器分配向量到聚类
   - 计算残差向量
   - 对残差训练细量化器（PQ）
   - 可选：预计算残差表
3. **输出：** 训练好的两级量化器

**代码位置：** `NPUIndexIVFPQ.cpp:103-132`

#### 添加向量 (add_core)
1. **输入：** 向量 `x` 和 ID
2. **步骤：**
   - 使用粗量化器分配到倒排列表
   - 计算残差向量
   - 使用 PQ 编码残差
   - 将编码和 ID 添加到对应的倒排列表
3. **输出：** 更新后的倒排列表

**代码位置：** `NPUIndexIVFPQ.cpp:151-197`

#### 搜索 (search_preassigned)
1. **输入：** 查询向量，预分配的倒排列表
2. **步骤：**
   - 对每个查询向量：
     - 遍历 nprobe 个最近的倒排列表
     - 计算查询残差
     - 计算距离表
     - 扫描倒排列表中的向量
     - 使用 PQ 编码计算近似距离
     - 维护 top-k 结果
3. **输出：** top-k 距离和标签

**代码位置：** `NPUIndexIVFPQ.cpp:226-311`

### 3. NPU 资源管理

**核心类：** `NPUResources`

**功能：**
- NPU 设备内存管理
- 临时内存分配
- 设备同步

**主要方法：**
```cpp
void* allocMemory(size_t size);       // 分配持久内存
void deallocMemory(void* ptr);        // 释放持久内存
void* allocTempMemory(size_t size);   // 分配临时内存
void syncDevice();                    // 同步设备
```

**代码位置：** `NPUResources.h/cpp`

### 4. NPU 算子

**核心算子：**

#### NPUPQDistanceOp
- `compute_distance_table()` - 计算查询向量的距离表
- `search_pq()` - 使用 PQ 编码进行搜索
- `compute_codes()` - 计算向量的 PQ 编码

**代码位置：** `NPUOps.cpp:15-143`

#### NPUIVFPQSearchOp
- `search_ivfpq()` - IVF-PQ 搜索
- `precompute_residual_table()` - 预计算残差表

**代码位置：** `NPUOps.cpp:145-171`

#### NPUKernels
- `l2_distance()` - L2 距离计算
- `inner_product()` - 内积计算
- `kmeans_assign()` - K-means 分配
- `compute_residual()` - 残差计算

**代码位置：** `NPUOps.cpp:173-254`

## 算子实现说明

### 当前实现（CPU 模拟版）

由于开发环境没有 NPU 硬件，当前实现使用 CPU 代码模拟 NPU 算子：

1. **距离计算**：使用标准 C++ 实现 L2 距离和内积
2. **PQ 编码**：暴力搜索最近的质心
3. **堆维护**：使用插入排序维护 top-k 结果

### NPU 算子优化方向

在实际 NPU 环境中，需要实现以下优化：

1. **距离表计算**
   - 使用 NPU 矩阵乘法加速
   - 批量处理多个查询向量

2. **PQ 搜索**
   - 使用 NPU 并行计算能力
   - 优化的内存访问模式
   - 使用 NPU 专用指令加速

3. **倒排列表扫描**
   - 并行扫描多个倒排列表
   - 向量化距离计算
   - 优化的 top-k 选择算法

4. **自定义算子**
   - 使用 Ascend C 或 TBE 编写自定义算子
   - 针对 NPU 架构优化内存布局
   - 使用 NPU 的高带宽内存（HBM）

## 使用示例

### NPUIndexPQ 使用

```cpp
#include <faiss/NPU/NPUIndexPQ.h>

// 创建索引
int d = 128;      // 向量维度
int M = 16;       // 子量化器数量
int nbits = 8;    // 每个子量化器的比特数

faiss::npu::NPUIndexPQConfig config;
config.device_id = 0;

faiss::npu::NPUIndexPQ index(d, M, nbits, faiss::METRIC_L2, config);

// 训练
index.train(ntrain, train_data);

// 添加向量
index.add(ntotal, database);

// 搜索
index.search(nq, queries, k, distances, labels);
```

### NPUIndexIVFPQ 使用

```cpp
#include <faiss/NPU/NPUIndexIVFPQ.h>
#include <faiss/IndexFlat.h>

// 创建粗量化器
faiss::IndexFlat quantizer(d, nlist, faiss::METRIC_L2);

// 创建索引
int nlist = 1000;
int M = 16;
int nbits = 8;

faiss::npu::NPUIndexIVFPQConfig config;
config.device_id = 0;

faiss::npu::NPUIndexIVFPQ index(
    &quantizer, d, nlist, M, nbits, 
    faiss::METRIC_L2, true, config
);

// 训练
index.train(ntrain, train_data);

// 添加向量
index.add(ntotal, database);

// 搜索
index.search(nq, queries, k, distances, labels);
```

## 性能优化建议

1. **内存管理**
   - 使用 NPU 专用内存分配器
   - 实现内存池减少分配开销
   - 优化数据布局提高缓存命中率

2. **并行化**
   - 批量处理查询向量
   - 并行扫描倒排列表
   - 使用 NPU 多核并行

3. **量化优化**
   - 使用更高效的编码方案
   - 优化质心初始化
   - 实现增量训练

4. **搜索优化**
   - 实现预计算表优化
   - 使用更高效的 top-k 算法
   - 优化内存访问模式

## 与 CPU/GPU 版本的对比

| 特性 | CPU 版本 | GPU 版本 | NPU 版本 |
|------|----------|----------|----------|
| 并行度 | 多线程 | CUDA 核心 | NPU 核心 |
| 内存带宽 | DDR | GDDR/HBM | HBM |
| 功耗 | 高 | 高 | 低 |
| 延迟 | 高 | 低 | 低 |
| 吞吐量 | 中 | 高 | 高 |
| 成本 | 低 | 高 | 中 |

## 后续开发计划

1. **算子优化**
   - 实现真正的 NPU 算子
   - 使用 Ascend C 编写自定义算子
   - 优化内存布局和访问模式

2. **功能扩展**
   - 支持 OPQ 和其他变换
   - 实现混合精度计算
   - 添加增量更新功能

3. **性能优化**
   - 批量处理优化
   - 内存池实现
   - 异步执行支持

4. **测试和验证**
   - 在真实 NPU 环境测试
   - 性能基准测试
   - 精度验证

## 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)
- [Product Quantization 论文](https://hal.inria.fr/inria-00514462v2/document)
- [华为昇腾开发文档](https://www.hiascend.com/document)
- [IndexSDK 参考实现](file:///home/yzc/IndexSDK)

## 编译说明

```bash
cd faiss/NPU
mkdir build && cd build
cmake ..
make -j8
make install
```

## 注意事项

1. 当前实现为 CPU 模拟版本，仅用于演示架构和流程
2. 实际部署需要在有 NPU 硬件的环境中重新实现算子部分
3. 性能优化需要针对具体的 NPU 架构进行调整
4. 内存管理需要使用 NPU 专用的内存分配器

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
