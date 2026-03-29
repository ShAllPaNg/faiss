#ifndef INDEX_TEST_H
#define INDEX_TEST_H

#include "Config.h"
#include "VectorGenerator.h"
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <vector>
#include <memory>

struct SearchResult {
    std::vector<faiss::idx_t> ids_;
    std::vector<float> distances_;
};

class IndexTest {
public:
    explicit IndexTest(const Config& config);
    ~IndexTest();
    
    void Run();
    
private:
    void GenerateData();
    void BuildIndex();
    void TrainIndex();
    void AddVectors();
    void SaveIndex();
    void LoadIndex();
    void Search();
    void Rerank();
    void Evaluate() const;
    
    float ComputeDistance(const float* a, const float* b, int32_t dim) const;
    
    Config config_;
    std::unique_ptr<VectorGenerator> vecGen_;
    
    std::vector<float> trainVecs_;
    std::vector<float> dbVecs_;
    std::vector<float> queryVecs_;
    
    faiss::IndexIVFPQ* index_ = nullptr;
    faiss::IndexHNSWFlat* quantizer_ = nullptr;
    
    SearchResult pqResult_;
    SearchResult rerankResult_;
    
    std::vector<std::vector<float>> dbVecsStorage_;
};

#endif
