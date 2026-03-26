/**
 * @file concurrent_search.cpp
 * @brief 并发搜索实现
 */

#include "concurrent_search.h"

#include <algorithm>
#include <iostream>
#include <cmath>

namespace ivfpq_search {

// ============================================================================
// ThreadPool 实现
// ============================================================================

ThreadPool::ThreadPool(size_t numThreads)
    : m_stop(false),
      m_activeTasks(0) {
    // 自动检测CPU核心数
    if (numThreads == 0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) {
            numThreads = 4;  // 默认4个线程
        }
    }

    std::cout << "[INFO] Creating thread pool with " << numThreads << " threads" << std::endl;

    // 创建工作线程
    for (size_t i = 0; i < numThreads; ++i) {
        m_workers.emplace_back(&ThreadPool::WorkerFunc, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_stop = true;
    }
    m_condition.notify_all();

    for (auto& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::Enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_tasks.push(task);
        {
            std::unique_lock<std::mutex> activeLock(m_activeMutex);
            ++m_activeTasks;
        }
    }
    m_condition.notify_one();
}

void ThreadPool::WaitForAll() {
    std::unique_lock<std::mutex> lock(m_activeMutex);
    m_activeCondition.wait(lock, [this] {
        return m_activeTasks == 0;
    });
}

void ThreadPool::WorkerFunc() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_condition.wait(lock, [this] {
                return m_stop || !m_tasks.empty();
            });

            if (m_stop && m_tasks.empty()) {
                return;
            }

            if (!m_tasks.empty()) {
                task = std::move(m_tasks.front());
                m_tasks.pop();
            }
        }

        if (task) {
            task();

            {
                std::unique_lock<std::mutex> lock(m_activeMutex);
                --m_activeTasks;
            }
            m_activeCondition.notify_one();
        }
    }
}

// ============================================================================
// ConcurrentSearch 实现
// ============================================================================

ConcurrentSearch::ConcurrentSearch(faiss::IndexIVFPQ* index, const SearchConfig& config)
    : m_index(index),
      m_config(config) {
    // 创建线程池
    m_threadPool = std::make_unique<ThreadPool>(m_config.nthread);

    std::cout << "[INFO] ConcurrentSearch initialized" << std::endl;
    std::cout << "[INFO]   - topk: " << m_config.topk << std::endl;
    std::cout << "[INFO]   - nthread: " << m_threadPool->GetThreadCount() << std::endl;
}

bool ConcurrentSearch::Search(const float* query, std::vector<SearchResult>& results) {
    if (!m_index) {
        std::cerr << "[ERROR] Index is null" << std::endl;
        return false;
    }

    if (!query) {
        std::cerr << "[ERROR] Query is null" << std::endl;
        return false;
    }

    DoSearch(query, m_config.topk, results);
    return true;
}

bool ConcurrentSearch::BatchSearch(const float* queries, size_t n,
                                   std::vector<std::vector<SearchResult>>& results) {
    if (!m_index) {
        std::cerr << "[ERROR] Index is null" << std::endl;
        return false;
    }

    if (!queries || n == 0) {
        std::cerr << "[ERROR] Invalid queries" << std::endl;
        return false;
    }

    results.resize(n);

    std::cout << "[INFO] Batch searching " << n << " queries..." << std::endl;

    // 提交所有搜索任务
    for (size_t i = 0; i < n; ++i) {
        m_threadPool->Enqueue([this, &queries, &results, i]() {
            const float* query = queries + i * m_index->d;
            DoSearch(query, m_config.topk, results[i]);
        });
    }

    // 等待所有任务完成
    m_threadPool->WaitForAll();

    std::cout << "[INFO] Batch search completed" << std::endl;
    return true;
}

void ConcurrentSearch::DoSearch(const float* query, size_t k,
                                 std::vector<SearchResult>& results) {
    results.clear();
    results.reserve(k);

    // Faiss搜索结果
    std::vector<float> distances(k);
    std::vector<int64_t> labels(k);

    try {
        m_index->search(1, query, k, distances.data(), labels.data());

        // 转换为SearchResult
        for (size_t i = 0; i < k; ++i) {
            if (labels[i] >= 0) {  // 有效的ID
                results.emplace_back(labels[i], distances[i]);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during search: " << e.what() << std::endl;
    }
}

void ConcurrentSearch::SetNprobe(size_t nprobe) {
    if (m_index) {
        m_index->nprobe = nprobe;
        std::cout << "[INFO] nprobe set to " << nprobe << std::endl;
    }
}

// ============================================================================
// 距离计算函数
// ============================================================================

float ComputeL2Distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void ComputeL2Distances(const float* query, const float* vectors,
                        size_t n, size_t dim, float* distances) {
    for (size_t i = 0; i < n; ++i) {
        distances[i] = ComputeL2Distance(query, vectors + i * dim, dim);
    }
}

} // namespace ivfpq_search
