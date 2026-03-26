/**
 * @file vector_generator.cpp
 * @brief 向量生成器实现
 */

#include "vector_generator.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

namespace ivfpq_search {

VectorFileHeader::VectorFileHeader()
    : magic(VECTOR_FILE_MAGIC),
      version(VECTOR_FILE_VERSION),
      dimension(0),
      count(0) {
    std::memset(reserved, 0, sizeof(reserved));
}

VectorGenerator::VectorGenerator(const VectorConfig& config)
    : m_config(config) {
    // 使用随机设备初始化随机数生成器
    std::random_device rd;
    m_rng.seed(rd());
}

VectorData VectorGenerator::Generate() {
    return Generate(m_config.count);
}

VectorData VectorGenerator::Generate(size_t count) {
    VectorData data(m_config.dimension, count);

    std::string dist = m_config.distribution;
    std::transform(dist.begin(), dist.end(), dist.begin(), ::tolower);

    if (dist == "uniform") {
        GenerateUniform(data);
    } else if (dist == "normal") {
        GenerateNormal(data);
    } else if (dist == "orthogonal") {
        GenerateOrthogonal(data);
    } else {
        // 默认使用均匀分布
        std::cerr << "[WARN] Unknown distribution: " << m_config.distribution
                  << ", using uniform" << std::endl;
        GenerateUniform(data);
    }

    return data;
}

void VectorGenerator::GenerateUniform(VectorData& data) {
    std::uniform_real_distribution<float> dist(m_config.minVal, m_config.maxVal);

    for (size_t i = 0; i < data.data.size(); ++i) {
        data.data[i] = dist(m_rng);
    }
}

void VectorGenerator::GenerateNormal(VectorData& data) {
    std::normal_distribution<float> dist(m_config.mean, m_config.std);

    for (size_t i = 0; i < data.data.size(); ++i) {
        data.data[i] = dist(m_rng);
    }
}

void VectorGenerator::GenerateOrthogonal(VectorData& data) {
    // 使用Gram-Schmidt正交化生成正交向量
    // 先生成随机向量，然后进行QR分解

    size_t dim = data.dimension;
    size_t count = std::min(data.count, dim);  // 正交向量最多dim个

    // 临时存储
    std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));

    // 生成随机向量
    std::uniform_real_distribution<float> dist(m_config.minVal, m_config.maxVal);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            vectors[i][j] = dist(m_rng);
        }
    }

    // Gram-Schmidt正交化
    for (size_t i = 0; i < count; ++i) {
        // 减去之前所有向量的投影
        for (size_t j = 0; j < i; ++j) {
            // 计算内积
            float dot = 0.0f;
            for (size_t k = 0; k < dim; ++k) {
                dot += vectors[i][k] * vectors[j][k];
            }

            // 计算第j个向量的模平方
            float normSq = 0.0f;
            for (size_t k = 0; k < dim; ++k) {
                normSq += vectors[j][k] * vectors[j][k];
            }

            // 减去投影
            float coeff = dot / normSq;
            for (size_t k = 0; k < dim; ++k) {
                vectors[i][k] -= coeff * vectors[j][k];
            }
        }

        // 归一化
        float norm = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            norm += vectors[i][j] * vectors[i][j];
        }
        norm = std::sqrt(norm);

        if (norm > 1e-10f) {
            for (size_t j = 0; j < dim; ++j) {
                vectors[i][j] /= norm;
            }
        }
    }

    // 复制到输出
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(data.Get(i), vectors[i].data(), dim * sizeof(float));
    }

    // 如果请求的向量数量大于维度，剩余的用均匀分布填充
    for (size_t i = count; i < data.count; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            data.data[i * dim + j] = dist(m_rng);
        }
    }
}

bool VectorGenerator::SaveToFile(const VectorData& data, const std::string& filePath) const {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open file for writing: " << filePath << std::endl;
        return false;
    }

    // 写入文件头
    if (!WriteFileHeader(file, data)) {
        file.close();
        return false;
    }

    // 写入向量数据
    file.write(reinterpret_cast<const char*>(data.data.data()),
               data.data.size() * sizeof(float));

    if (!file) {
        std::cerr << "[ERROR] Failed to write vector data" << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}

bool VectorGenerator::LoadFromFile(const std::string& filePath, VectorData& data) const {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open file for reading: " << filePath << std::endl;
        return false;
    }

    // 读取文件头
    if (!ReadFileHeader(file, data)) {
        file.close();
        return false;
    }

    // 分配内存
    data.Resize(data.count);

    // 读取向量数据
    file.read(reinterpret_cast<char*>(data.data.data()),
              data.data.size() * sizeof(float));

    if (!file) {
        std::cerr << "[ERROR] Failed to read vector data" << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}

bool VectorGenerator::WriteFileHeader(std::ofstream& file, const VectorData& data) const {
    VectorFileHeader header;
    header.dimension = data.dimension;
    header.count = data.count;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    if (!file) {
        std::cerr << "[ERROR] Failed to write file header" << std::endl;
        return false;
    }

    return true;
}

bool VectorGenerator::ReadFileHeader(std::ifstream& file, VectorData& data) const {
    VectorFileHeader header;

    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!file) {
        std::cerr << "[ERROR] Failed to read file header" << std::endl;
        return false;
    }

    // 验证魔数
    if (header.magic != VECTOR_FILE_MAGIC) {
        std::cerr << "[ERROR] Invalid file format: magic number mismatch" << std::endl;
        std::cerr << "[ERROR] Expected: 0x" << std::hex << VECTOR_FILE_MAGIC
                  << ", Got: 0x" << header.magic << std::dec << std::endl;
        return false;
    }

    // 验证版本
    if (header.version != VECTOR_FILE_VERSION) {
        std::cerr << "[WARN] File version mismatch: expected " << VECTOR_FILE_VERSION
                  << ", got " << header.version << std::endl;
    }

    // 填充数据
    data.dimension = static_cast<size_t>(header.dimension);
    data.count = static_cast<size_t>(header.count);

    return true;
}

} // namespace ivfpq_search
