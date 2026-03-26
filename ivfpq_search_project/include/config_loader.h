/**
 * @file config_loader.h
 * @brief INI配置文件加载器
 */

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>

#include "types.h"

namespace ivfpq_search {

/**
 * @brief 配置加载器类
 */
class ConfigLoader {
public:
    /**
     * @brief 构造函数
     */
    ConfigLoader() = default;

    /**
     * @brief 析构函数
     */
    ~ConfigLoader() = default;

    /**
     * @brief 从INI文件加载配置
     * @param configPath 配置文件路径
     * @param config 输出配置结构
     * @return 加载是否成功
     */
    bool LoadFromFile(const std::string& configPath, GlobalConfig& config);

    /**
     * @brief 保存配置到INI文件
     * @param configPath 配置文件路径
     * @param config 输入配置结构
     * @return 保存是否成功
     */
    bool SaveToFile(const std::string& configPath, const GlobalConfig& config) const;

    /**
     * @brief 创建默认配置文件
     * @param configPath 配置文件路径
     * @return 创建是否成功
     */
    static bool CreateDefault(const std::string& configPath);

    /**
     * @brief 验证配置有效性
     * @param config 待验证的配置
     * @return 验证是否通过
     */
    bool Validate(const GlobalConfig& config) const;

private:
    /**
     * @brief 解析字符串为MetricType
     * @param metricStr 度量类型字符串
     * @return MetricType枚举值
     */
    faiss::MetricType ParseMetricType(const std::string& metricStr) const;

    /**
     * @brief 将MetricType转换为字符串
     * @param metric MetricType枚举值
     * @return 度量类型字符串
     */
    std::string MetricTypeToString(faiss::MetricType metric) const;

    /**
     * @brief 去除字符串首尾空白
     * @param str 输入字符串
     * @return 处理后的字符串
     */
    std::string Trim(const std::string& str) const;

    /**
     * @brief 分割字符串
     * @param str 输入字符串
     * @param delimiter 分隔符
     * @return 分割后的字符串列表
     */
    std::vector<std::string> Split(const std::string& str, char delimiter) const;
};

} // namespace ivfpq_search

#endif  // CONFIG_LOADER_H
