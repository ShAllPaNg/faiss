/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/NPU/NPUIndexPQ.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexPQ.h>

#include <cstring>
#include <algorithm>

namespace faiss {
namespace npu {

struct NPUIndexPQ::Impl {
    int device_id;
    size_t temp_memory;
    bool use_float16;
    
    std::vector<uint8_t> codes_device;
    std::vector<float> centroids_device;
    
    Impl(int device, size_t temp_mem, bool fp16)
        : device_id(device), temp_memory(temp_mem), use_float16(fp16) {
    }
    
    void* getDeviceCodes() {
        return codes_device.data();
    }
    
    void* getDeviceCentroids() {
        return centroids_device.data();
    }
    
    size_t getCodesSize() const {
        return codes_device.size();
    }
    
    void allocateCodes(size_t size) {
        codes_device.resize(size);
    }
    
    void copyCodesToDevice(const uint8_t* host_codes, size_t size) {
        codes_device.assign(host_codes, host_codes + size);
    }
    
    void copyCodesFromDevice(uint8_t* host_codes, size_t size) const {
        std::copy(codes_device.begin(), codes_device.end(), host_codes);
    }
};

NPUIndexPQ::NPUIndexPQ(
        int d,
        size_t M,
        size_t nbits,
        MetricType metric,
        NPUIndexPQConfig config)
        : IndexFlatCodes(0, d, metric),
          pq(d, M, nbits),
          impl_(new Impl(config.device_id, config.temp_memory, config.use_float16)),
          config_(config) {
    verifyPQParams_();
    code_size = pq.code_size;
    is_trained = false;
}

NPUIndexPQ::NPUIndexPQ() : impl_(new Impl(0, 1 << 20, false)) {
    is_trained = false;
}

NPUIndexPQ::~NPUIndexPQ() = default;

void NPUIndexPQ::verifyPQParams_() const {
    FAISS_THROW_IF_NOT_MSG(pq.M > 0, "M must be > 0");
    FAISS_THROW_IF_NOT_MSG(pq.nbits > 0, "nbits must be > 0");
    FAISS_THROW_IF_NOT_MSG(d % pq.M == 0, "d must be a multiple of M");
    FAISS_THROW_IF_NOT_MSG(pq.nbits <= 16, "nbits must be <= 16");
}

void NPUIndexPQ::train(idx_t n, const float* x) {
    if (is_trained) {
        return;
    }
    
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    
    pq.train(n, x);
    
    impl_->centroids_device.assign(
            pq.centroids, 
            pq.centroids + pq.M * pq.ksub * pq.dsub);
    
    is_trained = true;
}

void NPUIndexPQ::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    
    idx_t prev_ntotal = ntotal;
    
    std::vector<uint8_t> new_codes(n * code_size);
    pq.compute_codes(x, new_codes.data(), n);
    
    codes.insert(codes.end(), new_codes.begin(), new_codes.end());
    ntotal += n;
    
    impl_->copyCodesToDevice(codes.data(), codes.size());
}

void NPUIndexPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(distances != nullptr);
    FAISS_THROW_IF_NOT(labels != nullptr);
    
    std::vector<float> distance_table(n * pq.M * pq.ksub);
    
    if (metric_type == METRIC_L2) {
        for (idx_t i = 0; i < n; i++) {
            pq.compute_distance_table(x + i * d, distance_table.data() + i * pq.M * pq.ksub);
        }
    } else {
        for (idx_t i = 0; i < n; i++) {
            pq.compute_inner_prod_table(x + i * d, distance_table.data() + i * pq.M * pq.ksub);
        }
    }
    
    if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {(size_t)n, (size_t)k, labels, distances};
        pq.search(x, n, codes.data(), ntotal, &res, true);
    } else {
        float_minheap_array_t res = {(size_t)n, (size_t)k, labels, distances};
        pq.search_ip(x, n, codes.data(), ntotal, &res, true);
    }
}

void NPUIndexPQ::reset() {
    codes.clear();
    ntotal = 0;
    impl_->codes_device.clear();
}

size_t NPUIndexPQ::reclaimMemory() {
    codes.shrink_to_fit();
    impl_->codes_device.shrink_to_fit();
    return 0;
}

int NPUIndexPQ::getDevice() const {
    return config_.device_id;
}

NPUResources* NPUIndexPQ::getResources() {
    return nullptr;
}

void NPUIndexPQ::copyFrom(const IndexPQ* index) {
    d = index->d;
    metric_type = index->metric_type;
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    pq = index->pq;
    code_size = index->code_size;
    codes = index->codes;
    
    impl_->copyCodesToDevice(codes.data(), codes.size());
    
    if (is_trained) {
        impl_->centroids_device.assign(
                pq.centroids,
                pq.centroids + pq.M * pq.ksub * pq.dsub);
    }
}

void NPUIndexPQ::copyTo(IndexPQ* index) const {
    index->d = d;
    index->metric_type = metric_type;
    index->ntotal = ntotal;
    index->is_trained = is_trained;
    index->pq = pq;
    index->code_size = code_size;
    index->codes = codes;
}

void NPUIndexPQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    pq.compute_codes(x, bytes, n);
}

void NPUIndexPQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    pq.decode(bytes, x, n);
}

FlatCodesDistanceComputer* NPUIndexPQ::get_FlatCodesDistanceComputer() const {
    return nullptr;
}

} 
}
