/**
 * @file concurrent_search.h
 * @brief 并发搜索模块
 */

#ifndef CONCURRENT_SEARCH_H
#define CONCURRENT_SEARCH_H

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

#include <faiss/IndexIVFPQ.h>

#include "types.h"

namespace ivfpq_search {

/**
 * @brief 线程池类
 */
class ThreadPool {
public:
    /**
     * @brief 构造函数
     * @param numThreads 线程数量 (0表示自动检测CPU核心数)
     */
    explicit ThreadPool(size_t numThreads = 0);

    /**
     * @brief 析构函数
     */
    ~ThreadPool();

    /**
     * @brief 提交任务到线程池
     * @param task 任务函数
     */
    void Enqueue(std::function<void()> task);

    /**
     * @brief 等待所有任务完成
     */
    void WaitForAll();

    /**
     * @brief 获取线程数量
     * @return 线程数量
     */
    size_t GetThreadCount() const {
        return m_workers.size();
    }

private:
    std::vector<std::thread> m_workers;    ///< 工作线程
    std::queue<std::function<void()>> m_tasks;  ///< 任务队列
    std::mutex m_queueMutex;               ///< 队列互斥锁
    std::condition_variable m_condition;   ///< 条件变量
    bool m_stop;                           ///< 停止标志
    size_t m_activeTasks;                  ///< 活跃任务数
    std::mutex m_activeMutex;              ///< 活跃任务互斥锁
    std::condition_variable m_activeCondition;  ///< 活跃任务条件变量

    /**
     * @brief 工作线程函数
     */
    void WorkerFunc();
};

/**
 * @brief 并发搜索类
 */
class ConcurrentSearch {
public:
    /**
     * @brief 构造函数
     * @param index IndexIVFPQ指针
     * @param config 搜索配置
     */
    ConcurrentSearch(faiss::IndexIVFPQ* index, const SearchConfig& config);

    /**
     * @brief 析构函数
     */
    ~ConcurrentSearch() = default;

    /**
     * @brief 单query搜索
     * @param query query向量
     * @param results 输出搜索结果
     * @return 搜索是否成功
     */
    bool Search(const float* query, std::vector<SearchResult>& results);

    /**
     * @brief 批量并发搜索
     * @param queries query向量数组
     * @param n query数量
     * @param results 输出搜索结果数组 (每个query对应一个结果向量)
     * @return 搜索是否成功
     */
    bool BatchSearch(const float* queries, size_t n,
                     std::vector<std::vector<SearchResult>>& results);

    /**
     * @brief 获取配置
     * @return 搜索配置
     */
    const SearchConfig& GetConfig() const {
        return m_config;
    }

    /**
     * @brief 设置nprobe参数
     * @param nprobe 探测的桶数
     */
    void SetNprobe(size_t nprobe);

private:
    faiss::IndexIVFPQ* m_index;      ///< IndexIVFPQ指针
    SearchConfig m_config;            ///< 搜索配置
    std::unique_ptr<ThreadPool> m_threadPool;  ///< 线程池

    /**
     * @brief 执行单个query搜索
     * @param query query向量
     * @param k 返回结果数量
     * @param results 输出搜索结果
     */
    void DoSearch(const float* query, size_t k, std::vector<SearchResult>& results);
};

/**
 * @brief L2距离计算函数
 * @param a 向量A
 * @param b 向量B
 * @param dim 向量维度
 * @return L2距离平方
 */
float ComputeL2Distance(const float* a, const float* b, size_t dim);

/**
 * @brief 批量计算L2距离
 * @param query query向量
 * @param vectors 向量数组
 * @param n 向量数量
 * @param dim 向量维度
 * @param distances 输出距离数组
 */
void ComputeL2Distances(const float* query, const float* vectors,
                        size_t n, size_t dim, float* distances);

} // namespace ivfpq_search

#endif  // CONCURRENT_SEARCH_H
