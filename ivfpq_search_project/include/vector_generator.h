/**
 * @file vector_generator.h
 * @brief 向量生成器，支持生成多种分布的随机向量
 */

#ifndef VECTOR_GENERATOR_H
#define VECTOR_GENERATOR_H

#include <string>
#include <random>
#include <memory>

#include "types.h"

namespace ivfpq_search {

/**
 * @brief 向量生成器类
 */
class VectorGenerator {
public:
    /**
     * @brief 构造函数
     * @param config 向量生成配置
     */
    explicit VectorGenerator(const VectorConfig& config);

    /**
     * @brief 析构函数
     */
    ~VectorGenerator() = default;

    /**
     * @brief 生成向量数据
     * @return 生成的向量数据
     */
    VectorData Generate();

    /**
     * @brief 生成指定数量的向量
     * @param count 向量数量
     * @return 生成的向量数据
     */
    VectorData Generate(size_t count);

    /**
     * @brief 保存向量到文件
     * @param data 向量数据
     * @param filePath 文件路径
     * @return 保存是否成功
     */
    bool SaveToFile(const VectorData& data, const std::string& filePath) const;

    /**
     * @brief 从文件加载向量
     * @param filePath 文件路径
     * @param data 输出的向量数据
     * @return 加载是否成功
     */
    bool LoadFromFile(const std::string& filePath, VectorData& data) const;

    /**
     * @brief 获取配置
     * @return 当前配置
     */
    const VectorConfig& GetConfig() const {
        return m_config;
    }

    /**
     * @brief 设置配置
     * @param config 新配置
     */
    void SetConfig(const VectorConfig& config) {
        m_config = config;
    }

private:
    VectorConfig m_config;          ///< 向量生成配置
    std::mt19937 m_rng;             ///< 随机数生成器

    /**
     * @brief 生成均匀分布向量
     * @param data 输出向量数据
     */
    void GenerateUniform(VectorData& data);

    /**
     * @brief 生成正态分布向量
     * @param data 输出向量数据
     */
    void GenerateNormal(VectorData& data);

    /**
     * @brief 生成正交向量
     * @param data 输出向量数据
     */
    void GenerateOrthogonal(VectorData& data);

    /**
     * @brief 写入文件头
     * @param file 输出文件流
     * @param data 向量数据
     * @return 写入是否成功
     */
    bool WriteFileHeader(std::ofstream& file, const VectorData& data) const;

    /**
     * @brief 读取并验证文件头
     * @param file 输入文件流
     * @param data 输出向量数据(仅填充维度和数量)
     * @return 读取是否成功
     */
    bool ReadFileHeader(std::ifstream& file, VectorData& data) const;
};

/**
 * @brief 向量文件魔数
 */
constexpr uint32_t VECTOR_FILE_MAGIC = 0x56454353;  // "VECS" in little endian

/**
 * @brief 向量文件版本
 */
constexpr uint32_t VECTOR_FILE_VERSION = 1;

/**
 * @brief 向量文件头结构
 */
struct VectorFileHeader {
    uint32_t magic;        ///< 魔数
    uint32_t version;      ///< 版本号
    uint64_t dimension;    ///< 向量维度
    uint64_t count;        ///< 向量数量
    uint8_t  reserved[16]; ///< 保留字段

    /**
     * @brief 构造函数，初始化默认值
     */
    VectorFileHeader();
};

} // namespace ivfpq_search

#endif  // VECTOR_GENERATOR_H
