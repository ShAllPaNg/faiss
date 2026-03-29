#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <cstdint>

struct Config {
    int32_t dimension_ = 128;
    int32_t numTrainVecs_ = 100000;
    int32_t numDbVecs_ = 1000000;
    int32_t numQueryVecs_ = 1000;
    int32_t nlist_ = 1000;
    int32_t nprobe_ = 32;
    int32_t m_ = 8;
    int32_t nbits_ = 8;
    int32_t hnswM_ = 32;
    int32_t hnswEfSearch_ = 64;
    int32_t topK_ = 100;
    int32_t rerankK_ = 200;
    bool byResidual_ = true;
    std::string indexFilePath_ = "ivfpq_index.bin";
    std::string configFilePath_ = "config.ini";

    static Config LoadFromFile(const std::string& filePath);
    void SaveToFile(const std::string& filePath) const;
    void Print() const;
};

#endif
