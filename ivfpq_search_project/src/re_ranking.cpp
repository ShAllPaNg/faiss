/**
 * @file re_ranking.cpp
 * @brief 重排序实现
 */

#include "re_ranking.h"

#include <algorithm>
#include <iostream>
#include <cmath>

namespace {

float ComputeL2Distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

} // anonymous namespace

namespace ivfpq_search {

ReRanking::ReRanking(const VectorData& databaseVectors)
    : m_database(databaseVectors) {
}

std::vector<SearchResult> ReRanking::ReRank(
    const float* query,
    const std::vector<SearchResult>& initialResults,
    size_t topk) {
    std::vector<SearchResult> refinedResults;

    if (initialResults.empty()) {
        return refinedResults;
    }

    refinedResults.reserve(initialResults.size());

    // 重新计算精确距离
    for (const auto& result : initialResults) {
        if (result.id >= 0 && static_cast<size_t>(result.id) < m_database.Size()) {
            float exactDistance = ComputeExactDistance(query, result.id);
            refinedResults.emplace_back(result.id, exactDistance);
        }
    }

    // 按距离排序
    std::sort(refinedResults.begin(), refinedResults.end(), CompareByDistance);

    // 取topk
    if (refinedResults.size() > topk) {
        refinedResults.resize(topk);
    }

    return refinedResults;
}

std::vector<std::vector<SearchResult>> ReRanking::BatchReRank(
    const float* queries,
    size_t n,
    const std::vector<std::vector<SearchResult>>& initialResultsArray,
    size_t topk) {
    std::vector<std::vector<SearchResult>> refinedResultsArray(n);

    for (size_t i = 0; i < n; ++i) {
        const float* query = queries + i * m_database.Dim();
        refinedResultsArray[i] = ReRank(query, initialResultsArray[i], topk);
    }

    return refinedResultsArray;
}

float ReRanking::ComputeExactDistance(const float* query, int64_t databaseId) const {
    if (databaseId < 0 || static_cast<size_t>(databaseId) >= m_database.Size()) {
        std::cerr << "[ERROR] Invalid database ID: " << databaseId << std::endl;
        return std::numeric_limits<float>::max();
    }

    const float* databaseVec = m_database.Get(static_cast<size_t>(databaseId));
    return ComputeL2Distance(query, databaseVec, m_database.Dim());
}

} // namespace ivfpq_search
