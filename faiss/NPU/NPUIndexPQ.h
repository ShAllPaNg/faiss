/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_NPU_INDEX_PQ_H
#define FAISS_NPU_INDEX_PQ_H

#include <vector>
#include <memory>

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

namespace npu {

struct NPUResources;

struct NPUIndexPQConfig {
    int device_id = 0;
    size_t temp_memory = 1 << 20;
    bool use_float16 = false;
    
    NPUIndexPQConfig() {}
};

class NPUIndexPQ : public IndexFlatCodes {
public:
    ProductQuantizer pq;
    
    NPUIndexPQ(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            NPUIndexPQConfig config = NPUIndexPQConfig());
    
    NPUIndexPQ();
    
    ~NPUIndexPQ() override;
    
    void train(idx_t n, const float* x) override;
    
    void add(idx_t n, const float* x) override;
    
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    
    void reset() override;
    
    size_t reclaimMemory();
    
    int getDevice() const;
    
    NPUResources* getResources();
    
    void copyFrom(const IndexPQ* index);
    
    void copyTo(IndexPQ* index) const;
    
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
    
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

protected:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    NPUIndexPQConfig config_;
    
    void verifyPQParams_() const;
};

} 

} 

#endif
