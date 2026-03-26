/**
 * @file ivfpq_index.h
 * @brief IndexIVFPQ封装类
 */

#ifndef IVFPQ_INDEX_H
#define IVFPQ_INDEX_H

#include <string>
#include <memory>

#include <faiss/IndexIVFPQ.h>

#include "types.h"

namespace ivfpq_search {

/**
 * @brief IndexIVFPQ封装类
 */
class IVFPQIndex {
public:
    /**
     * @brief 构造函数
     * @param config 索引配置
     * @param dimension 向量维度
     */
    IVFPQIndex(const IndexConfig& config, size_t dimension);

    /**
     * @brief 析构函数
     */
    ~IVFPQIndex() = default;

    /**
     * @brief 训练索引
     * @param trainingVectors 训练向量数据
     * @param n 训练向量数量
     * @return 训练是否成功
     */
    bool Train(const float* trainingVectors, size_t n);

    /**
     * @brief 添加向量到索引
     * @param vectors 向量数据
     * @param n 向量数量
     * @param ids 向量ID (可为nullptr，自动分配0开始的ID)
     * @return 添加是否成功
     */
    bool AddVectors(const float* vectors, size_t n, const int64_t* ids = nullptr);

    /**
     * @brief 保存索引到文件
     * @param indexPath 索引文件路径
     * @return 保存是否成功
     */
    bool Save(const std::string& indexPath) const;

    /**
     * @brief 从文件加载索引
     * @param indexPath 索引文件路径
     * @return 加载是否成功
     */
    bool Load(const std::string& indexPath);

    /**
     * @brief 获取IndexIVFPQ指针
     * @return IndexIVFPQ指针
     */
    faiss::IndexIVFPQ* GetIndex() {
        return m_index.get();
    }

    /**
     * @brief 获取IndexIVFPQ指针 (const版本)
     * @return IndexIVFPQ指针
     */
    const faiss::IndexIVFPQ* GetIndex() const {
        return m_index.get();
    }

    /**
     * @brief 获取向量维度
     * @return 向量维度
     */
    size_t GetDimension() const {
        return m_dimension;
    }

    /**
     * @brief 获取索引配置
     * @return 索引配置
     */
    const IndexConfig& GetConfig() const {
        return m_config;
    }

    /**
     * @brief 获取索引中的向量数量
     * @return 向量数量
     */
    size_t GetVectorCount() const {
        return m_index ? m_index->ntotal : 0;
    }

    /**
     * @brief 获取索引统计信息
     * @return 统计信息字符串
     */
    std::string GetStats() const;

    /**
     * @brief 设置nprobe参数
     * @param nprobe 探测的桶数
     */
    void SetNprobe(size_t nprobe);

    /**
     * @brief 重置索引
     */
    void Reset();

private:
    IndexConfig m_config;               ///< 索引配置
    size_t m_dimension;                 ///< 向量维度
    std::unique_ptr<faiss::IndexIVFPQ> m_index;  ///< Faiss IndexIVFPQ指针

    /**
     * @brief 创建IndexIVFPQ
     * @return 创建是否成功
     */
    bool CreateIndex();
};

} // namespace ivfpq_search

#endif  // IVFPQ_INDEX_H
