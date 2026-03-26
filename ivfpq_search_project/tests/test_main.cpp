/**
 * @file test_main.cpp
 * @brief IndexIVFPQ向量检索系统测试程序
 */

#include <iostream>
#include <chrono>

#include "config_loader.h"
#include "vector_generator.h"
#include "ivfpq_index.h"
#include "concurrent_search.h"
#include "re_ranking.h"

using namespace ivfpq_search;

/**
 * @brief 打印搜索结果
 * @param results 搜索结果
 * @param queryIndex query索引
 */
void PrintSearchResults(const std::vector<SearchResult>& results, size_t queryIndex) {
    std::cout << "\n--- Query " << queryIndex << " Results ---" << std::endl;
    std::cout << "Rank\tID\t\tDistance" << std::endl;
    std::cout << "----\t--\t\t--------" << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << i << "\t" << results[i].id << "\t\t"
                  << std::sqrt(results[i].distance) << std::endl;
    }
}

/**
 * @brief 主测试函数
 */
int RunTest(const std::string& configPath) {
    std::cout << "========================================" << std::endl;
    std::cout << "  IndexIVFPQ Vector Search System" << std::endl;
    std::cout << "========================================" << std::endl;

    // ========== 1. 加载配置 ==========
    std::cout << "\n[Step 1] Loading configuration..." << std::endl;
    ConfigLoader loader;
    GlobalConfig config;

    if (!loader.LoadFromFile(configPath, config)) {
        std::cerr << "[ERROR] Failed to load config file: " << configPath << std::endl;
        return -1;
    }

    std::cout << "[INFO] Configuration loaded successfully" << std::endl;

    // ========== 2. 生成向量 ==========
    std::cout << "\n[Step 2] Generating vectors..." << std::endl;
    VectorGenerator generator(config.vectorConfig);

    // 生成训练集向量
    std::cout << "[INFO] Generating training vectors..." << std::endl;
    VectorData trainVectors(config.vectorConfig.dimension, 50000);  // 默认5万训练向量
    trainVectors = generator.Generate(trainVectors.Size());

    if (!generator.SaveToFile(trainVectors, config.trainVectorsPath)) {
        std::cerr << "[ERROR] Failed to save training vectors" << std::endl;
        return -1;
    }

    // 生成库向量
    std::cout << "[INFO] Generating database vectors..." << std::endl;
    VectorData databaseVectors = generator.Generate();
    if (!generator.SaveToFile(databaseVectors, config.databaseVectorsPath)) {
        std::cerr << "[ERROR] Failed to save database vectors" << std::endl;
        return -1;
    }

    // 生成query向量
    std::cout << "[INFO] Generating query vectors..." << std::endl;
    VectorConfig queryConfig = config.vectorConfig;
    queryConfig.count = 100;  // 100个query
    generator.SetConfig(queryConfig);
    VectorData queryVectors = generator.Generate();
    if (!generator.SaveToFile(queryVectors, config.queryVectorsPath)) {
        std::cerr << "[ERROR] Failed to save query vectors" << std::endl;
        return -1;
    }

    std::cout << "[INFO] Vectors generated and saved successfully" << std::endl;

    // ========== 3. 构建索引 ==========
    std::cout << "\n[Step 3] Building IndexIVFPQ..." << std::endl;
    IVFPQIndex index(config.indexConfig, config.vectorConfig.dimension);

    auto trainStart = std::chrono::high_resolution_clock::now();
    if (!index.Train(trainVectors.data(), trainVectors.Size())) {
        std::cerr << "[ERROR] Failed to train index" << std::endl;
        return -1;
    }
    auto trainEnd = std::chrono::high_resolution_clock::now();
    double trainTime = std::chrono::duration<double>(trainEnd - trainStart).count();
    std::cout << "[INFO] Training time: " << trainTime << " seconds" << std::endl;

    auto addStart = std::chrono::high_resolution_clock::now();
    if (!index.AddVectors(databaseVectors.data(), databaseVectors.Size())) {
        std::cerr << "[ERROR] Failed to add vectors" << std::endl;
        return -1;
    }
    auto addEnd = std::chrono::high_resolution_clock::now();
    double addTime = std::chrono::duration<double>(addEnd - addStart).count();
    std::cout << "[INFO] Add vectors time: " << addTime << " seconds" << std::endl;

    // 保存索引
    if (!index.Save(config.indexPath)) {
        std::cerr << "[ERROR] Failed to save index" << std::endl;
        return -1;
    }

    std::cout << "[INFO] Index statistics:" << std::endl;
    std::cout << index.GetStats() << std::endl;

    // ========== 4. 搜索 ==========
    std::cout << "\n[Step 4] Searching..." << std::endl;
    ConcurrentSearch searcher(index.GetIndex(), config.searchConfig);
    searcher.SetNprobe(config.indexConfig.nprobe);

    auto searchStart = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<SearchResult>> initialResults;
    if (!searcher.BatchSearch(queryVectors.data(), queryVectors.Size(), initialResults)) {
        std::cerr << "[ERROR] Failed to search" << std::endl;
        return -1;
    }
    auto searchEnd = std::chrono::high_resolution_clock::now();
    double searchTime = std::chrono::duration<double>(searchEnd - searchStart).count();

    std::cout << "[INFO] Search completed" << std::endl;
    std::cout << "[INFO] Search time: " << searchTime << " seconds" << std::endl;
    std::cout << "[INFO] Average time per query: "
              << (searchTime / queryVectors.Size() * 1000) << " ms" << std::endl;

    // ========== 5. 重排序 (如果启用) ==========
    std::vector<std::vector<SearchResult>> finalResults;
    if (config.searchConfig.enableReRank) {
        std::cout << "\n[Step 5] Re-ranking..." << std::endl;

        ReRanking reRanking(databaseVectors);

        auto reRankStart = std::chrono::high_resolution_clock::now();
        finalResults = reRanking.BatchReRank(
            queryVectors.data(),
            queryVectors.Size(),
            initialResults,
            config.searchConfig.finalTopk);
        auto reRankEnd = std::chrono::high_resolution_clock::now();
        double reRankTime = std::chrono::duration<double>(reRankEnd - reRankStart).count();

        std::cout << "[INFO] Re-ranking completed" << std::endl;
        std::cout << "[INFO] Re-ranking time: " << reRankTime << " seconds" << std::endl;
    } else {
        finalResults = initialResults;
    }

    // ========== 6. 输出结果 ==========
    std::cout << "\n[Step 6] Output results..." << std::endl;

    // 打印前3个query的结果
    size_t numToPrint = std::min(size_t(3), finalResults.size());
    for (size_t i = 0; i < numToPrint; ++i) {
        PrintSearchResults(finalResults[i], i);
    }

    // ========== 完成 ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    std::string configPath = "config/default.ini";

    if (argc > 1) {
        configPath = argv[1];
    }

    return RunTest(configPath);
}
