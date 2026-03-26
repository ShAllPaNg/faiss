/**
 * @file re_ranking.h
 * @brief 重排序模块 - 精确距离计算与重排序
 */

#ifndef RE_RANKING_H
#define RE_RANKING_H

#include <vector>

#include "types.h"

namespace ivfpq_search {

/**
 * @brief 重排序类
 */
class ReRanking {
public:
    /**
     * @brief 构造函数
     * @param databaseVectors 库向量数据
     */
    explicit ReRanking(const VectorData& databaseVectors);

    /**
     * @brief 析构函数
     */
    ~ReRanking() = default;

    /**
     * @brief 单query重排序
     * @param query query向量
     * @param initialResults 初筛结果
     * @param topk 返回top-k结果
     * @return 重排序后的结果
     */
    std::vector<SearchResult> ReRank(
        const float* query,
        const std::vector<SearchResult>& initialResults,
        size_t topk);

    /**
     * @brief 批量重排序
     * @param queries query向量数组
     * @param n query数量
     * @param initialResultsArray 初筛结果数组
     * @param topk 返回top-k结果
     * @return 重排序后的结果数组
     */
    std::vector<std::vector<SearchResult>> BatchReRank(
        const float* queries,
        size_t n,
        const std::vector<std::vector<SearchResult>>& initialResultsArray,
        size_t topk);

private:
    const VectorData& m_database;    ///< 库向量数据引用

    /**
     * @brief 精确计算L2距离
     * @param query query向量
     * @param databaseId 库向量ID
     * @return L2距离平方
     */
    float ComputeExactDistance(const float* query, int64_t databaseId) const;
};

/**
 * @brief 结果排序比较函数 (按距离升序)
 * @param a 结果A
 * @param b 结果B
 * @return a是否排在b前面
 */
inline bool CompareByDistance(const SearchResult& a, const SearchResult& b) {
    return a.distance < b.distance;
}

} // namespace ivfpq_search

#endif  // RE_RANKING_H
