#include <faiss/NPU/NPUIndexPQ.h>
#include <faiss/NPU/NPUIndexIVFPQ.h>
#include <faiss/IndexFlat.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

void test_npu_index_pq() {
    std::cout << "Testing NPUIndexPQ..." << std::endl;
    
    const int d = 128;
    const int M = 16;
    const int nbits = 8;
    const size_t ntrain = 10000;
    const size_t ntotal = 100000;
    const size_t nq = 100;
    const int k = 10;
    
    std::vector<float> train_data(ntrain * d);
    std::vector<float> database(ntotal * d);
    std::vector<float> queries(nq * d);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(-1.0, 1.0);
    
    for (size_t i = 0; i < ntrain * d; i++) {
        train_data[i] = distrib(rng);
    }
    for (size_t i = 0; i < ntotal * d; i++) {
        database[i] = distrib(rng);
    }
    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = distrib(rng);
    }
    
    faiss::npu::NPUIndexPQConfig config;
    config.device_id = 0;
    
    faiss::npu::NPUIndexPQ index(d, M, nbits, faiss::METRIC_L2, config);
    
    std::cout << "Training index..." << std::endl;
    index.train(ntrain, train_data.data());
    
    std::cout << "Adding vectors..." << std::endl;
    index.add(ntotal, database.data());
    
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    std::cout << "Searching..." << std::endl;
    index.search(nq, queries.data(), k, distances.data(), labels.data());
    
    std::cout << "Search completed. Sample results:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Query " << i << ": label[0]=" << labels[i * k] 
                  << ", dist[0]=" << distances[i * k] << std::endl;
    }
    
    std::cout << "NPUIndexPQ test passed!" << std::endl << std::endl;
}

void test_npu_index_ivfpq() {
    std::cout << "Testing NPUIndexIVFPQ..." << std::endl;
    
    const int d = 128;
    const size_t nlist = 100;
    const int M = 16;
    const int nbits = 8;
    const size_t ntrain = 10000;
    const size_t ntotal = 100000;
    const size_t nq = 100;
    const int k = 10;
    
    std::vector<float> train_data(ntrain * d);
    std::vector<float> database(ntotal * d);
    std::vector<float> queries(nq * d);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(-1.0, 1.0);
    
    for (size_t i = 0; i < ntrain * d; i++) {
        train_data[i] = distrib(rng);
    }
    for (size_t i = 0; i < ntotal * d; i++) {
        database[i] = distrib(rng);
    }
    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = distrib(rng);
    }
    
    faiss::IndexFlat quantizer(d, nlist, faiss::METRIC_L2);
    
    faiss::npu::NPUIndexIVFPQConfig config;
    config.device_id = 0;
    config.use_precomputed_table = true;
    
    faiss::npu::NPUIndexIVFPQ index(
        &quantizer, d, nlist, M, nbits, 
        faiss::METRIC_L2, false, config
    );
    
    std::cout << "Training index..." << std::endl;
    index.train(ntrain, train_data.data());
    
    std::cout << "Adding vectors..." << std::endl;
    index.add(ntotal, database.data());
    
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    std::cout << "Searching..." << std::endl;
    index.search(nq, queries.data(), k, distances.data(), labels.data());
    
    std::cout << "Search completed. Sample results:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Query " << i << ": label[0]=" << labels[i * k] 
                  << ", dist[0]=" << distances[i * k] << std::endl;
    }
    
    std::cout << "NPUIndexIVFPQ test passed!" << std::endl << std::endl;
}

int main() {
    std::cout << "=== NPU Index Tests ===" << std::endl << std::endl;
    
    try {
        test_npu_index_pq();
        test_npu_index_ivfpq();
        
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
