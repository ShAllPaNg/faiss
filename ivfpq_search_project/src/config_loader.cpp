/**
 * @file config_loader.cpp
 * @brief INI配置文件加载器实现
 */

#include "config_loader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace ivfpq_search {

bool ConfigLoader::LoadFromFile(const std::string& configPath, GlobalConfig& config) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open config file: " << configPath << std::endl;
        return false;
    }

    std::string line;
    std::string currentSection;

    while (std::getline(file, line)) {
        // 去除首尾空白
        line = Trim(line);

        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // 解析section
        if (line[0] == '[' && line.back() == ']') {
            currentSection = line.substr(1, line.length() - 2);
            continue;
        }

        // 解析key=value
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            continue;
        }

        std::string key = Trim(line.substr(0, equalPos));
        std::string value = Trim(line.substr(equalPos + 1));

        // 根据section和key填充配置
        if (currentSection == "vector") {
            if (key == "dimension") {
                config.vectorConfig.dimension = std::stoul(value);
            } else if (key == "database_count") {
                config.vectorConfig.count = std::stoul(value);
            } else if (key == "train_count") {
                // 单独存储训练集数量，不存入VectorConfig
            } else if (key == "query_count") {
                // query数量单独处理
            } else if (key == "distribution") {
                config.vectorConfig.distribution = value;
            } else if (key == "min_val") {
                config.vectorConfig.minVal = std::stof(value);
            } else if (key == "max_val") {
                config.vectorConfig.maxVal = std::stof(value);
            } else if (key == "mean") {
                config.vectorConfig.mean = std::stof(value);
            } else if (key == "std") {
                config.vectorConfig.std = std::stof(value);
            }
        } else if (currentSection == "index") {
            if (key == "nlist") {
                config.indexConfig.nlist = std::stoul(value);
            } else if (key == "M") {
                config.indexConfig.M = std::stoul(value);
            } else if (key == "nbits") {
                config.indexConfig.nbits = std::stoul(value);
            } else if (key == "metric") {
                config.indexConfig.metric = ParseMetricType(value);
            } else if (key == "nprobe") {
                config.indexConfig.nprobe = std::stoul(value);
            } else if (key == "by_residual") {
                config.indexConfig.by_residual = (value == "true" || value == "1");
            } else if (key == "use_precomputed_table") {
                config.indexConfig.use_precomputed_table = (value == "true" || value == "1");
            }
        } else if (currentSection == "search") {
            if (key == "topk") {
                config.searchConfig.topk = std::stoul(value);
            } else if (key == "final_topk") {
                config.searchConfig.finalTopk = std::stoul(value);
            } else if (key == "nthread") {
                config.searchConfig.nthread = std::stoul(value);
            } else if (key == "enable_re_rank") {
                config.searchConfig.enableReRank = (value == "true" || value == "1");
            }
        } else if (currentSection == "path") {
            if (key == "train_vectors_path") {
                config.trainVectorsPath = value;
            } else if (key == "database_vectors_path") {
                config.databaseVectorsPath = value;
            } else if (key == "query_vectors_path") {
                config.queryVectorsPath = value;
            } else if (key == "index_path") {
                config.indexPath = value;
            } else if (key == "result_path") {
                config.resultPath = value;
            }
        } else if (currentSection == "advanced") {
            if (key == "log_level") {
                config.logLevel = std::stoi(value);
            }
        }
    }

    file.close();

    // 验证配置
    return Validate(config);
}

bool ConfigLoader::SaveToFile(const std::string& configPath, const GlobalConfig& config) const {
    std::ofstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to create config file: " << configPath << std::endl;
        return false;
    }

    file << "# ==============================================\n";
    file << "# IndexIVFPQ Vector Search Configuration File\n";
    file << "# ==============================================\n\n";

    // [vector] section
    file << "[vector]\n";
    file << "dimension = " << config.vectorConfig.dimension << "\n";
    file << "database_count = " << config.vectorConfig.count << "\n";
    file << "distribution = " << config.vectorConfig.distribution << "\n";
    file << "min_val = " << config.vectorConfig.minVal << "\n";
    file << "max_val = " << config.vectorConfig.maxVal << "\n";
    file << "mean = " << config.vectorConfig.mean << "\n";
    file << "std = " << config.vectorConfig.std << "\n\n";

    // [index] section
    file << "[index]\n";
    file << "nlist = " << config.indexConfig.nlist << "\n";
    file << "M = " << config.indexConfig.M << "\n";
    file << "nbits = " << config.indexConfig.nbits << "\n";
    file << "metric = " << MetricTypeToString(config.indexConfig.metric) << "\n";
    file << "nprobe = " << config.indexConfig.nprobe << "\n";
    file << "by_residual = " << (config.indexConfig.by_residual ? "true" : "false") << "\n";
    file << "use_precomputed_table = " << (config.indexConfig.use_precomputed_table ? "true" : "false") << "\n\n";

    // [search] section
    file << "[search]\n";
    file << "topk = " << config.searchConfig.topk << "\n";
    file << "final_topk = " << config.searchConfig.finalTopk << "\n";
    file << "nthread = " << config.searchConfig.nthread << "\n";
    file << "enable_re_rank = " << (config.searchConfig.enableReRank ? "true" : "false") << "\n\n";

    // [path] section
    file << "[path]\n";
    file << "train_vectors_path = " << config.trainVectorsPath << "\n";
    file << "database_vectors_path = " << config.databaseVectorsPath << "\n";
    file << "query_vectors_path = " << config.queryVectorsPath << "\n";
    file << "index_path = " << config.indexPath << "\n";
    file << "result_path = " << config.resultPath << "\n\n";

    // [advanced] section
    file << "[advanced]\n";
    file << "log_level = " << config.logLevel << "\n";

    file.close();
    return true;
}

bool ConfigLoader::CreateDefault(const std::string& configPath) {
    std::ofstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to create default config file: " << configPath << std::endl;
        return false;
    }

    file << "# ==============================================\n";
    file << "# IndexIVFPQ Vector Search Configuration File\n";
    file << "# ==============================================\n";
    file << "# 说明: 此配置文件用于IndexIVFPQ向量检索系统的参数配置\n";
    file << "# 修改配置后需重启程序生效\n\n";

    file << "[vector]\n";
    file << "# 向量维度 (推荐: 64, 128, 256)\n";
    file << "dimension = 128\n\n";
    file << "# 库向量数量 (用于测试)\n";
    file << "database_count = 100000\n\n";
    file << "# query向量数量\n";
    file << "query_count = 100\n\n";
    file << "# 训练集向量数量 (通常为数据库的10%-50%，或至少nlist×M)\n";
    file << "train_count = 50000\n\n";
    file << "# 向量分布类型: uniform, normal, orthogonal\n";
    file << "distribution = uniform\n\n";
    file << "# 数值范围 (uniform分布时使用)\n";
    file << "min_val = -1.0\n";
    file << "max_val = 1.0\n\n";
    file << "# 均值和标准差 (normal分布时使用)\n";
    file << "mean = 0.0\n";
    file << "std = 1.0\n\n";

    file << "[index]\n";
    file << "# IVF倒排列表数量 (建议: sqrt(database_count) 或更大)\n";
    file << "nlist = 316\n\n";
    file << "# PQ子量化器数量 (dimension必须能被M整除)\n";
    file << "M = 16\n\n";
    file << "# 每个子量化器位数 (推荐: 8, 每个子量化器256个质心)\n";
    file << "nbits = 8\n\n";
    file << "# 距离度量类型: L2, INNER_PRODUCT\n";
    file << "metric = L2\n\n";
    file << "# 搜索时探测的桶数 (建议: nlist/10 到 nlist/2)\n";
    file << "nprobe = 16\n\n";
    file << "# 是否使用残差量化 (推荐: true)\n";
    file << "by_residual = true\n\n";
    file << "# 是否使用预计算表 (内存消耗大，速度更快)\n";
    file << "use_precomputed_table = false\n\n";

    file << "[search]\n";
    file << "# 初筛返回top-k结果 (用于重排序)\n";
    file << "topk = 100\n\n";
    file << "# 重排序后最终返回数量\n";
    file << "final_topk = 10\n\n";
    file << "# 并发线程数 (0表示自动检测CPU核心数)\n";
    file << "nthread = 0\n\n";
    file << "# 是否启用重排序\n";
    file << "enable_re_rank = true\n\n";

    file << "[path]\n";
    file << "# 训练集向量路径\n";
    file << "train_vectors_path = data/raw_vectors/train_vectors.fvec\n\n";
    file << "# 库向量路径\n";
    file << "database_vectors_path = data/raw_vectors/database_vectors.fvec\n\n";
    file << "# query向量路径\n";
    file << "query_vectors_path = data/raw_vectors/query_vectors.fvec\n\n";
    file << "# IndexIVFPQ索引保存路径\n";
    file << "index_path = data/index/ivfpq_index.faiss\n\n";
    file << "# 搜索结果输出路径\n";
    file << "result_path = data/results/search_results.txt\n\n";

    file << "[advanced]\n";
    file << "# 输出日志级别: 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG\n";
    file << "log_level = 2\n";

    file.close();
    return true;
}

bool ConfigLoader::Validate(const GlobalConfig& config) const {
    // 验证向量配置
    if (config.vectorConfig.dimension == 0) {
        std::cerr << "[ERROR] Vector dimension cannot be 0" << std::endl;
        return false;
    }
    if (config.vectorConfig.count == 0) {
        std::cerr << "[ERROR] Vector count cannot be 0" << std::endl;
        return false;
    }
    if (config.vectorConfig.dimension % config.indexConfig.M != 0) {
        std::cerr << "[ERROR] Dimension must be divisible by M" << std::endl;
        return false;
    }

    // 验证索引配置
    if (config.indexConfig.nlist == 0) {
        std::cerr << "[ERROR] nlist cannot be 0" << std::endl;
        return false;
    }
    if (config.indexConfig.M == 0) {
        std::cerr << "[ERROR] M cannot be 0" << std::endl;
        return false;
    }
    if (config.indexConfig.nbits == 0 || config.indexConfig.nbits > 16) {
        std::cerr << "[ERROR] nbits must be between 1 and 16" << std::endl;
        return false;
    }
    if (config.indexConfig.nprobe > config.indexConfig.nlist) {
        std::cerr << "[WARN] nprobe is larger than nlist" << std::endl;
    }

    // 验证搜索配置
    if (config.searchConfig.topk == 0) {
        std::cerr << "[ERROR] topk cannot be 0" << std::endl;
        return false;
    }
    if (config.searchConfig.finalTopk > config.searchConfig.topk) {
        std::cerr << "[ERROR] final_topk cannot be larger than topk" << std::endl;
        return false;
    }

    return true;
}

faiss::MetricType ConfigLoader::ParseMetricType(const std::string& metricStr) const {
    std::string lowerStr = metricStr;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    if (lowerStr == "l2") {
        return faiss::METRIC_L2;
    } else if (lowerStr == "inner_product" || lowerStr == "ip") {
        return faiss::METRIC_INNER_PRODUCT;
    }

    return faiss::METRIC_L2;  // 默认L2
}

std::string ConfigLoader::MetricTypeToString(faiss::MetricType metric) const {
    switch (metric) {
        case faiss::METRIC_L2:
            return "L2";
        case faiss::METRIC_INNER_PRODUCT:
            return "INNER_PRODUCT";
        default:
            return "L2";
    }
}

std::string ConfigLoader::Trim(const std::string& str) const {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::vector<std::string> ConfigLoader::Split(const std::string& str, char delimiter) const {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(Trim(token));
    }

    return tokens;
}

} // namespace ivfpq_search
