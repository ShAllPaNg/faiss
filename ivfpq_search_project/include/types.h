/**
 * @file types.h
 * @brief IndexIVFPQ向量检索系统的公共数据结构定义
 */

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include <cstdint>

#include <faiss/MetricType.h>

namespace ivfpq_search {

/**
 * @brief 向量生成配置
 */
struct VectorConfig {
    size_t dimension;           ///< 向量维度
    size_t count;               ///< 向量数量
    std::string distribution;   ///< 分布类型: "uniform", "normal", "orthogonal"
    float minVal;               ///< 数值范围最小值 (uniform分布)
    float maxVal;               ///< 数值范围最大值 (uniform分布)
    float mean;                 ///< 均值 (normal分布)
    float std;                  ///< 标准差 (normal分布)

    /**
     * @brief 默认构造函数
     */
    VectorConfig()
        : dimension(128),
          count(10000),
          distribution("uniform"),
          minVal(-1.0f),
          maxVal(1.0f),
          mean(0.0f),
          std(1.0f) {}
};

/**
 * @brief IndexIVFPQ配置
 */
struct IndexConfig {
    size_t nlist;                   ///< IVF桶数量
    size_t M;                       ///< PQ子量化器数量
    size_t nbits;                   ///< 每个子量化器位数
    faiss::MetricType metric;       ///< 距离度量类型
    size_t nprobe;                  ///< 搜索时探测的桶数
    bool by_residual;               ///< 是否使用残差量化
    bool use_precomputed_table;     ///< 是否使用预计算表

    /**
     * @brief 默认构造函数
     */
    IndexConfig()
        : nlist(316),
          M(16),
          nbits(8),
          metric(faiss::METRIC_L2),
          nprobe(16),
          by_residual(true),
          use_precomputed_table(false) {}
};

/**
 * @brief 搜索配置
 */
struct SearchConfig {
    size_t topk;            ///< 初筛返回top-k结果
    size_t finalTopk;       ///< 重排序后最终返回数量
    size_t nthread;         ///< 并发线程数 (0表示自动检测)
    bool enableReRank;      ///< 是否启用重排序

    /**
     * @brief 默认构造函数
     */
    SearchConfig()
        : topk(100),
          finalTopk(10),
          nthread(0),
          enableReRank(true) {}
};

/**
 * @brief 单个检索结果
 */
struct SearchResult {
    int64_t id;              ///< 向量ID (从0开始)
    float distance;          ///< 距离

    /**
     * @brief 默认构造函数
     */
    SearchResult()
        : id(-1),
          distance(0.0f) {}

    /**
     * @brief 构造函数
     * @param id 向量ID
     * @param distance 距离
     */
    SearchResult(int64_t id, float distance)
        : id(id),
          distance(distance) {}
};

/**
 * @brief 向量数据容器
 */
struct VectorData {
    std::vector<float> m_data;    ///< 连续存储的向量数据
    size_t dimension;            ///< 向量维度
    size_t count;                ///< 向量数量

    /**
     * @brief 默认构造函数
     */
    VectorData()
        : dimension(0),
          count(0) {}

    /**
     * @brief 构造函数
     * @param dim 向量维度
     * @param cnt 向量数量
     */
    VectorData(size_t dim, size_t cnt)
        : dimension(dim),
          count(cnt) {
        m_data.resize(dim * cnt);
    }

    /**
     * @brief 获取向量数量
     * @return 向量数量
     */
    size_t Size() const {
        return count;
    }

    /**
     * @brief 获取向量维度
     * @return 向量维度
     */
    size_t Dim() const {
        return dimension;
    }

    /**
     * @brief 获取数据指针
     * @return 数据指针
     */
    float* data() {
        return m_data.data();
    }

    /**
     * @brief 获取数据指针 (const版本)
     * @return 数据指针
     */
    const float* data() const {
        return m_data.data();
    }

    /**
     * @brief 获取指定索引的向量指针
     * @param idx 向量索引
     * @return 向量数据指针
     */
    float* Get(size_t idx) {
        return m_data.data() + idx * dimension;
    }

    /**
     * @brief 获取指定索引的向量指针 (const版本)
     * @param idx 向量索引
     * @return 向量数据指针
     */
    const float* Get(size_t idx) const {
        return m_data.data() + idx * dimension;
    }

    /**
     * @brief 调整大小
     * @param cnt 新的向量数量
     */
    void Resize(size_t cnt) {
        count = cnt;
        m_data.resize(dimension * cnt);
    }

    /**
     * @brief 清空数据
     */
    void Clear() {
        m_data.clear();
        count = 0;
    }
};

/**
 * @brief 全局配置
 */
struct GlobalConfig {
    VectorConfig vectorConfig;
    IndexConfig indexConfig;
    SearchConfig searchConfig;

    std::string trainVectorsPath;
    std::string databaseVectorsPath;
    std::string queryVectorsPath;
    std::string indexPath;
    std::string resultPath;

    int logLevel;

    /**
     * @brief 默认构造函数
     */
    GlobalConfig()
        : logLevel(2) {}
};

} // namespace ivfpq_search

#endif  // TYPES_H
