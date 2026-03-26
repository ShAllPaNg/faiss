/**
 * @file ivfpq_index.cpp
 * @brief IndexIVFPQ封装实现
 */

#include "ivfpq_index.h"

#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include <iostream>
#include <sstream>

namespace ivfpq_search {

IVFPQIndex::IVFPQIndex(const IndexConfig& config, size_t dimension)
    : m_config(config),
      m_dimension(dimension),
      m_index(nullptr) {
}

bool IVFPQIndex::CreateIndex() {
    // 验证配置
    if (m_dimension == 0) {
        std::cerr << "[ERROR] Dimension cannot be 0" << std::endl;
        return false;
    }

    if (m_dimension % m_config.M != 0) {
        std::cerr << "[ERROR] Dimension " << m_dimension
                  << " must be divisible by M " << m_config.M << std::endl;
        return false;
    }

    // 使用index_factory创建IndexIVFPQ
    // 格式: "IVFPQ[nlist],[M],[nbits]"
    std::stringstream ss;
    ss << "IVF" << m_config.nlist << ",PQ" << m_config.M << "x" << m_config.nbits;

    std::string indexDesc = ss.str();
    std::cout << "[INFO] Creating index: " << indexDesc << std::endl;

    try {
        // 创建索引
        m_index.reset(static_cast<faiss::IndexIVFPQ*>(
            faiss::index_factory(m_dimension, indexDesc.c_str(), m_config.metric)
        ));

        if (!m_index) {
            std::cerr << "[ERROR] Failed to create IndexIVFPQ" << std::endl;
            return false;
        }

        // 设置参数
        m_index->by_residual = m_config.by_residual;
        // m_index->use_precomputed_table = m_config.use_precomputed_table;

        std::cout << "[INFO] IndexIVFPQ created successfully" << std::endl;
        std::cout << "[INFO]   - nlist: " << m_config.nlist << std::endl;
        std::cout << "[INFO]   - M: " << m_config.M << std::endl;
        std::cout << "[INFO]   - nbits: " << m_config.nbits << std::endl;
        std::cout << "[INFO]   - by_residual: " << (m_config.by_residual ? "true" : "false") << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while creating index: " << e.what() << std::endl;
        return false;
    }
}

bool IVFPQIndex::Train(const float* trainingVectors, size_t n) {
    if (!m_index) {
        if (!CreateIndex()) {
            return false;
        }
    }

    if (m_index->is_trained) {
        std::cout << "[WARN] Index is already trained" << std::endl;
        return true;
    }

    if (!trainingVectors || n == 0) {
        std::cerr << "[ERROR] Invalid training vectors" << std::endl;
        return false;
    }

    std::cout << "[INFO] Training index with " << n << " vectors..." << std::endl;

    try {
        m_index->train(n, trainingVectors);

        if (m_index->is_trained) {
            std::cout << "[INFO] Index trained successfully" << std::endl;
            return true;
        } else {
            std::cerr << "[ERROR] Index training failed" << std::endl;
            return false;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while training: " << e.what() << std::endl;
        return false;
    }
}

bool IVFPQIndex::AddVectors(const float* vectors, size_t n, const int64_t* ids) {
    if (!m_index) {
        std::cerr << "[ERROR] Index not created" << std::endl;
        return false;
    }

    if (!m_index->is_trained) {
        std::cerr << "[ERROR] Index not trained" << std::endl;
        return false;
    }

    if (!vectors || n == 0) {
        std::cerr << "[ERROR] Invalid vectors to add" << std::endl;
        return false;
    }

    std::cout << "[INFO] Adding " << n << " vectors to index..." << std::endl;

    try {
        if (ids) {
            m_index->add_with_ids(n, vectors, ids);
        } else {
            m_index->add(n, vectors);
        }

        std::cout << "[INFO] Vectors added successfully, total: "
                  << m_index->ntotal << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while adding vectors: " << e.what() << std::endl;
        return false;
    }
}

bool IVFPQIndex::Save(const std::string& indexPath) const {
    if (!m_index) {
        std::cerr << "[ERROR] No index to save" << std::endl;
        return false;
    }

    std::cout << "[INFO] Saving index to: " << indexPath << std::endl;

    try {
        faiss::write_index(m_index.get(), indexPath.c_str());
        std::cout << "[INFO] Index saved successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while saving index: " << e.what() << std::endl;
        return false;
    }
}

bool IVFPQIndex::Load(const std::string& indexPath) {
    std::cout << "[INFO] Loading index from: " << indexPath << std::endl;

    try {
        std::unique_ptr<faiss::Index> loadedIndex(faiss::read_index(indexPath.c_str()));

        if (!loadedIndex) {
            std::cerr << "[ERROR] Failed to load index" << std::endl;
            return false;
        }

        // 尝试转换为IndexIVFPQ
        faiss::IndexIVFPQ* ivfpqIndex = dynamic_cast<faiss::IndexIVFPQ*>(loadedIndex.get());
        if (!ivfpqIndex) {
            std::cerr << "[ERROR] Loaded index is not IndexIVFPQ" << std::endl;
            return false;
        }

        // 释放所有权
        m_index.reset(ivfpqIndex);
        loadedIndex.release();

        // 更新维度
        m_dimension = m_index->d;

        std::cout << "[INFO] Index loaded successfully" << std::endl;
        std::cout << "[INFO]   - Dimension: " << m_dimension << std::endl;
        std::cout << "[INFO]   - Vectors: " << m_index->ntotal << std::endl;
        std::cout << "[INFO]   - nlist: " << m_index->nlist << std::endl;
        std::cout << "[INFO]   - is_trained: " << (m_index->is_trained ? "true" : "false") << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while loading index: " << e.what() << std::endl;
        return false;
    }
}

std::string IVFPQIndex::GetStats() const {
    if (!m_index) {
        return "Index not created";
    }

    std::stringstream ss;
    ss << "=== IndexIVFPQ Statistics ===\n";
    ss << "Dimension: " << m_dimension << "\n";
    ss << "Vector count: " << m_index->ntotal << "\n";
    ss << "nlist: " << m_index->nlist << "\n";
    ss << "is_trained: " << (m_index->is_trained ? "true" : "false") << "\n";
    ss << "Metric: " << (m_index->metric_type == faiss::METRIC_L2 ? "L2" : "INNER_PRODUCT") << "\n";

    if (m_index->ntotal > 0) {
        ss << "Code size: " << m_index->code_size << " bytes\n";
    }

    return ss.str();
}

void IVFPQIndex::SetNprobe(size_t nprobe) {
    if (m_index) {
        m_index->nprobe = nprobe;
        std::cout << "[INFO] nprobe set to " << nprobe << std::endl;
    }
}

void IVFPQIndex::Reset() {
    if (m_index) {
        m_index->reset();
        std::cout << "[INFO] Index reset" << std::endl;
    }
}

} // namespace ivfpq_search
