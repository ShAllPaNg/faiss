#include "Config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

static std::string Trim(const std::string& str)
{
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

Config Config::LoadFromFile(const std::string& filePath)
{
    Config config;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Config file not found: " << filePath 
                  << ", using default config" << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#' || line[0] == '[') {
            continue;
        }

        size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }

        std::string key = Trim(line.substr(0, pos));
        std::string value = Trim(line.substr(pos + 1));

        if (key == "dimension") {
            config.dimension_ = std::stoi(value);
        } else if (key == "num_train_vecs") {
            config.numTrainVecs_ = std::stoi(value);
        } else if (key == "num_db_vecs") {
            config.numDbVecs_ = std::stoi(value);
        } else if (key == "num_query_vecs") {
            config.numQueryVecs_ = std::stoi(value);
        } else if (key == "nlist") {
            config.nlist_ = std::stoi(value);
        } else if (key == "nprobe") {
            config.nprobe_ = std::stoi(value);
        } else if (key == "m") {
            config.m_ = std::stoi(value);
        } else if (key == "nbits") {
            config.nbits_ = std::stoi(value);
        } else if (key == "hnsw_m") {
            config.hnswM_ = std::stoi(value);
        } else if (key == "hnsw_ef_search") {
            config.hnswEfSearch_ = std::stoi(value);
        } else if (key == "top_k") {
            config.topK_ = std::stoi(value);
        } else if (key == "rerank_k") {
            config.rerankK_ = std::stoi(value);
        } else if (key == "by_residual") {
            config.byResidual_ = (value == "true" || value == "1");
        } else if (key == "index_file_path") {
            config.indexFilePath_ = value;
        }
    }

    file.close();
    return config;
}

void Config::SaveToFile(const std::string& filePath) const
{
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to create config file: " << filePath << std::endl;
        return;
    }

    file << "# IndexIVFPQ Test Configuration\n";
    file << "# Generated automatically\n\n";
    
    file << "[vector]\n";
    file << "dimension = " << dimension_ << "\n";
    file << "num_train_vecs = " << numTrainVecs_ << "\n";
    file << "num_db_vecs = " << numDbVecs_ << "\n";
    file << "num_query_vecs = " << numQueryVecs_ << "\n\n";
    
    file << "[ivf]\n";
    file << "nlist = " << nlist_ << "\n";
    file << "nprobe = " << nprobe_ << "\n";
    file << "by_residual = " << (byResidual_ ? "true" : "false") << "\n\n";
    
    file << "[pq]\n";
    file << "m = " << m_ << "\n";
    file << "nbits = " << nbits_ << "\n\n";
    
    file << "[hnsw]\n";
    file << "hnsw_m = " << hnswM_ << "\n";
    file << "hnsw_ef_search = " << hnswEfSearch_ << "\n\n";
    
    file << "[search]\n";
    file << "top_k = " << topK_ << "\n";
    file << "rerank_k = " << rerankK_ << "\n\n";
    
    file << "[io]\n";
    file << "index_file_path = " << indexFilePath_ << "\n";

    file.close();
}

void Config::Print() const
{
    std::cout << "========== Configuration ==========\n";
    std::cout << "dimension: " << dimension_ << "\n";
    std::cout << "num_train_vecs: " << numTrainVecs_ << "\n";
    std::cout << "num_db_vecs: " << numDbVecs_ << "\n";
    std::cout << "num_query_vecs: " << numQueryVecs_ << "\n";
    std::cout << "nlist: " << nlist_ << "\n";
    std::cout << "nprobe: " << nprobe_ << "\n";
    std::cout << "m: " << m_ << "\n";
    std::cout << "nbits: " << nbits_ << "\n";
    std::cout << "hnsw_m: " << hnswM_ << "\n";
    std::cout << "hnsw_ef_search: " << hnswEfSearch_ << "\n";
    std::cout << "top_k: " << topK_ << "\n";
    std::cout << "rerank_k: " << rerankK_ << "\n";
    std::cout << "by_residual: " << (byResidual_ ? "true" : "false") << "\n";
    std::cout << "index_file_path: " << indexFilePath_ << "\n";
    std::cout << "===================================\n";
}
