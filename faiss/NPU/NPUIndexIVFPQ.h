/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_NPU_INDEX_IVFPQ_H
#define FAISS_NPU_INDEX_IVFPQ_H

#include <vector>
#include <memory>

#include <faiss/IndexIVF.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct IndexIVFPQ;

namespace npu {

struct NPUResources;
class NPUIndexFlat;

struct NPUIndexIVFPQConfig {
    int device_id = 0;
    size_t temp_memory = 1 << 20;
    bool use_float16 = false;
    bool use_precomputed_table = false;
    bool interleaved_layout = false;
    
    NPUIndexIVFPQConfig() {}
};

class NPUIndexIVFPQ : public IndexIVF {
public:
    ProductQuantizer pq;
    
    NPUIndexIVFPQ(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            MetricType metric = METRIC_L2,
            bool own_invlists = true,
            NPUIndexIVFPQConfig config = NPUIndexIVFPQConfig());
    
    NPUIndexIVFPQ();
    
    ~NPUIndexIVFPQ() override;
    
    void train(idx_t n, const float* x) override;
    
    void add(idx_t n, const float* x) override;
    
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    
    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;
    
    void reset() override;
    
    void reserveMemory(size_t numVecs);
    
    size_t reclaimMemory();
    
    int getDevice() const;
    
    NPUResources* getResources();
    
    void copyFrom(const IndexIVFPQ* index);
    
    void copyTo(IndexIVFPQ* index) const;
    
    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;
    
    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* listnos,
            float* x) const override;
    
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
    
    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;
    
    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;
    
    idx_t train_encoder_num_vectors() const override;
    
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;
    
    void precompute_table();
    
    void setPrecomputedCodes(bool enable);
    
    bool getPrecomputedCodes() const;
    
    int getNumSubQuantizers() const;
    
    int getBitsPerCode() const;
    
    int getCentroidsPerSubQuantizer() const;
    
    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

protected:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    NPUIndexIVFPQConfig config_;
    
    int use_precomputed_table_;
    AlignedTable<float> precomputed_table_;
    
    void verifyPQParams_() const;
    
    void updateQuantizer() override;
};

} 

} 

#endif
