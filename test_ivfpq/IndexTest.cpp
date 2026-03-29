#include "IndexTest.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <faiss/impl/AuxIndexStructures.h>

IndexTest::IndexTest(const Config& config)
    : config_(config)
{
    vecGen_ = std::make_unique<VectorGenerator>(config_.dimension_);
}

IndexTest::~IndexTest()
{
    delete index_;
    delete quantizer_;
}

void IndexTest::GenerateData()
{
    std::cout << "\n[1] Generating data..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    trainVecs_ = vecGen_->GenerateVectors(config_.numTrainVecs_);
    std::cout << "  Train vectors: " << config_.numTrainVecs_ 
              << " x " << config_.dimension_ << std::endl;
    
    dbVecs_ = vecGen_->GenerateVectors(config_.numDbVecs_);
    std::cout << "  Database vectors: " << config_.numDbVecs_ 
              << " x " << config_.dimension_ << std::endl;
    
    queryVecs_ = vecGen_->GenerateVectors(config_.numQueryVecs_);
    std::cout << "  Query vectors: " << config_.numQueryVecs_ 
              << " x " << config_.dimension_ << std::endl;
    
    dbVecsStorage_.resize(config_.numDbVecs_);
    for (int32_t i = 0; i < config_.numDbVecs_; ++i) {
        dbVecsStorage_[i].assign(
            dbVecs_.data() + static_cast<size_t>(i) * config_.dimension_,
            dbVecs_.data() + static_cast<size_t>(i + 1) * config_.dimension_
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
}

void IndexTest::BuildIndex()
{
    std::cout << "\n[2] Building index structure..." << std::endl;
    
    quantizer_ = new faiss::IndexHNSWFlat(config_.dimension_, config_.hnswM_);
    quantizer_->hnsw.efSearch = config_.hnswEfSearch_;
    
    std::cout << "  Quantizer: IndexHNSWFlat (M=" << config_.hnswM_ 
              << ", efSearch=" << config_.hnswEfSearch_ << ")" << std::endl;
    
    index_ = new faiss::IndexIVFPQ(quantizer_, config_.dimension_, 
                                    config_.nlist_, config_.m_, config_.nbits_);
    index_->by_residual = config_.byResidual_;
    index_->nprobe = config_.nprobe_;
    index_->quantizer_trains_alone = 2;
    
    std::cout << "  Index: IndexIVFPQ (nlist=" << config_.nlist_ 
              << ", M=" << config_.m_ << ", nbits=" << config_.nbits_ 
              << ", by_residual=" << (config_.byResidual_ ? "true" : "false") << ")" << std::endl;
}

void IndexTest::TrainIndex()
{
    std::cout << "\n[3] Training index..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (config_.numTrainVecs_ < config_.nlist_) {
        std::cerr << "  Warning: numTrainVecs < nlist, training may fail!" << std::endl;
    }
    
    index_->train(config_.numTrainVecs_, trainVecs_.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Training completed, time: " << duration.count() << " ms" << std::endl;
    std::cout << "  nlist = " << index_->nlist << std::endl;
}

void IndexTest::AddVectors()
{
    std::cout << "\n[4] Adding vectors..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    index_->add(config_.numDbVecs_, dbVecs_.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Added " << index_->ntotal << " vectors" << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
}

void IndexTest::SaveIndex()
{
    std::cout << "\n[5] Saving index to disk..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    faiss::write_index(index_, config_.indexFilePath_.c_str());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Saved to: " << config_.indexFilePath_ << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
}

void IndexTest::LoadIndex()
{
    std::cout << "\n[6] Loading index from disk..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    delete index_;
    index_ = nullptr;
    
    faiss::Index* loadedIndex = faiss::read_index(config_.indexFilePath_.c_str());
    index_ = dynamic_cast<faiss::IndexIVFPQ*>(loadedIndex);
    
    if (!index_) {
        std::cerr << "  Error: Failed to cast loaded index to IndexIVFPQ!" << std::endl;
        return;
    }
    
    index_->nprobe = config_.nprobe_;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Loaded from: " << config_.indexFilePath_ << std::endl;
    std::cout << "  ntotal = " << index_->ntotal << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
}

void IndexTest::Search()
{
    std::cout << "\n[7] Searching (PQ)..." << std::endl;
    
    pqResult_.ids_.resize(static_cast<size_t>(config_.numQueryVecs_) * config_.topK_);
    pqResult_.distances_.resize(static_cast<size_t>(config_.numQueryVecs_) * config_.topK_);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    index_->search(config_.numQueryVecs_, queryVecs_.data(), config_.topK_,
                   pqResult_.distances_.data(), pqResult_.ids_.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double qps = config_.numQueryVecs_ * 1000.0 / duration.count();
    std::cout << "  Search completed, time: " << duration.count() << " ms" << std::endl;
    std::cout << "  QPS: " << qps << std::endl;
}

void IndexTest::Rerank()
{
    std::cout << "\n[8] Reranking..." << std::endl;
    
    rerankResult_.ids_.resize(static_cast<size_t>(config_.numQueryVecs_) * config_.topK_);
    rerankResult_.distances_.resize(static_cast<size_t>(config_.numQueryVecs_) * config_.topK_);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int32_t rerankK = std::min(config_.rerankK_, config_.topK_);
    
    #pragma omp parallel for
    for (int32_t q = 0; q < config_.numQueryVecs_; ++q) {
        const float* query = queryVecs_.data() + static_cast<size_t>(q) * config_.dimension_;
        
        std::vector<std::pair<float, faiss::idx_t>> candidates;
        candidates.reserve(config_.topK_);
        
        for (int32_t k = 0; k < config_.topK_; ++k) {
            faiss::idx_t id = pqResult_.ids_[static_cast<size_t>(q) * config_.topK_ + k];
            if (id < 0 || id >= config_.numDbVecs_) {
                continue;
            }
            
            float dist = ComputeDistance(query, dbVecsStorage_[id].data(), config_.dimension_);
            candidates.emplace_back(dist, id);
        }
        
        std::sort(candidates.begin(), candidates.end());
        
        for (int32_t k = 0; k < config_.topK_ && k < static_cast<int32_t>(candidates.size()); ++k) {
            size_t idx = static_cast<size_t>(q) * config_.topK_ + k;
            rerankResult_.ids_[idx] = candidates[k].second;
            rerankResult_.distances_[idx] = candidates[k].first;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Reranking completed, time: " << duration.count() << " ms" << std::endl;
}

float IndexTest::ComputeDistance(const float* a, const float* b, int32_t dim) const
{
    float dist = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

void IndexTest::Evaluate() const
{
    std::cout << "\n[9] Sample results (first 3 queries):" << std::endl;
    
    for (int32_t q = 0; q < std::min(3, config_.numQueryVecs_); ++q) {
        std::cout << "\n  Query " << q << ":\n";
        std::cout << "    PQ top-5:  ";
        for (int32_t k = 0; k < std::min(5, config_.topK_); ++k) {
            size_t idx = static_cast<size_t>(q) * config_.topK_ + k;
            std::cout << "[" << pqResult_.ids_[idx] << "," 
                      << pqResult_.distances_[idx] << "] ";
        }
        std::cout << "\n";
        
        std::cout << "    Rerank top-5: ";
        for (int32_t k = 0; k < std::min(5, config_.topK_); ++k) {
            size_t idx = static_cast<size_t>(q) * config_.topK_ + k;
            std::cout << "[" << rerankResult_.ids_[idx] << "," 
                      << rerankResult_.distances_[idx] << "] ";
        }
        std::cout << "\n";
    }
}

void IndexTest::Run()
{
    std::cout << "========================================" << std::endl;
    std::cout << "    IndexIVFPQ Test with HNSW Quantizer" << std::endl;
    std::cout << "========================================" << std::endl;
    
    config_.Print();
    
    GenerateData();
    BuildIndex();
    TrainIndex();
    AddVectors();
    SaveIndex();
    LoadIndex();
    Search();
    Rerank();
    Evaluate();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "    Test completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
}
