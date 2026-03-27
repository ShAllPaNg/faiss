/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/NPU/NPUIndexIVFPQ.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/DirectMap.h>

#include <cstring>
#include <algorithm>
#include <memory>

namespace faiss {
namespace npu {

struct NPUIndexIVFPQ::Impl {
    int device_id;
    size_t temp_memory;
    bool use_float16;
    
    std::vector<std::vector<uint8_t>> device_invlists_codes;
    std::vector<std::vector<idx_t>> device_invlists_ids;
    std::vector<float> centroids_device;
    std::vector<float> pq_centroids_device;
    
    Impl(int device, size_t temp_mem, bool fp16)
        : device_id(device), temp_memory(temp_mem), use_float16(fp16) {
    }
    
    void allocateInvertedLists(size_t nlist) {
        device_invlists_codes.resize(nlist);
        device_invlists_ids.resize(nlist);
    }
    
    void copyInvertedListToDevice(
            size_t list_no,
            const uint8_t* codes,
            const idx_t* ids,
            size_t size,
            size_t code_size) {
        if (list_no >= device_invlists_codes.size()) {
            return;
        }
        device_invlists_codes[list_no].assign(codes, codes + size * code_size);
        device_invlists_ids[list_no].assign(ids, ids + size);
    }
};

NPUIndexIVFPQ::NPUIndexIVFPQ(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        MetricType metric,
        bool own_invlists,
        NPUIndexIVFPQConfig config)
        : IndexIVF(quantizer, d, nlist, metric, own_invlists),
          pq(d, M, nbits_per_idx),
          impl_(new Impl(config.device_id, config.temp_memory, config.use_float16)),
          config_(config),
          use_precomputed_table_(0) {
    verifyPQParams_();
    code_size = pq.code_size;
    is_trained = false;
    
    impl_->allocateInvertedLists(nlist);
}

NPUIndexIVFPQ::NPUIndexIVFPQ() 
    : IndexIVF(),
      impl_(new Impl(0, 1 << 20, false)),
      use_precomputed_table_(0) {
    is_trained = false;
}

NPUIndexIVFPQ::~NPUIndexIVFPQ() = default;

void NPUIndexIVFPQ::verifyPQParams_() const {
    FAISS_THROW_IF_NOT_MSG(pq.M > 0, "M must be > 0");
    FAISS_THROW_IF_NOT_MSG(pq.nbits > 0, "nbits must be > 0");
    FAISS_THROW_IF_NOT_MSG(d % pq.M == 0, "d must be a multiple of M");
    FAISS_THROW_IF_NOT_MSG(pq.nbits <= 16, "nbits must be <= 16");
}

void NPUIndexIVFPQ::train(idx_t n, const float* x) {
    if (is_trained) {
        return;
    }
    
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    
    quantizer->train(n, x);
    quantizer->is_trained = true;
    
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());
    
    std::vector<float> residuals(n * d);
    for (idx_t i = 0; i < n; i++) {
        quantizer->compute_residual(x + i * d, residuals.data() + i * d, assign[i]);
    }
    
    pq.train(n, residuals.data());
    
    impl_->pq_centroids_device.assign(
            pq.centroids,
            pq.centroids + pq.M * pq.ksub * pq.dsub);
    
    if (config_.use_precomputed_table && metric_type == METRIC_L2 && by_residual) {
        precompute_table();
    }
    
    is_trained = true;
}

void NPUIndexIVFPQ::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void NPUIndexIVFPQ::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    
    std::vector<idx_t> ids(n);
    if (xids) {
        std::copy(xids, xids + n, ids.begin());
    } else {
        for (idx_t i = 0; i < n; i++) {
            ids[i] = ntotal + i;
        }
    }
    
    add_core(n, x, ids.data(), nullptr, nullptr);
}

void NPUIndexIVFPQ::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    
    std::vector<idx_t> assign(n);
    if (precomputed_idx) {
        std::copy(precomputed_idx, precomputed_idx + n, assign.begin());
    } else {
        quantizer->assign(n, x, assign.data());
    }
    
    std::vector<float> residuals(n * d);
    for (idx_t i = 0; i < n; i++) {
        if (assign[i] >= 0) {
            quantizer->compute_residual(x + i * d, residuals.data() + i * d, assign[i]);
        }
    }
    
    std::vector<uint8_t> codes(n * code_size);
    pq.compute_codes(residuals.data(), codes.data(), n);
    
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = assign[i];
        if (list_no >= 0 && list_no < (idx_t)nlist) {
            invlists->add_entry(
                    list_no,
                    xids ? xids[i] : ntotal + i,
                    codes.data() + i * code_size);
        }
    }
    
    ntotal += n;
    
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = assign[i];
        if (list_no >= 0 && list_no < (idx_t)nlist) {
            impl_->copyInvertedListToDevice(
                    list_no,
                    codes.data() + i * code_size,
                    xids ? (xids + i) : &ids[i],
                    1,
                    code_size);
        }
    }
}

void NPUIndexIVFPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    
    std::vector<idx_t> assign(n * nprobe);
    std::vector<float> centroid_dis(n * nprobe);
    
    quantizer->search(n, x, nprobe, centroid_dis.data(), assign.data());
    
    search_preassigned(
            n,
            x,
            k,
            assign.data(),
            centroid_dis.data(),
            distances,
            labels,
            false,
            nullptr,
            nullptr);
}

void NPUIndexIVFPQ::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    
    std::fill(labels, labels + n * k, -1);
    std::fill(distances, distances + n * k, 
              metric_type == METRIC_L2 ? INFINITY : -INFINITY);
    
    for (idx_t i = 0; i < n; i++) {
        std::vector<float> distance_table(pq.M * pq.ksub);
        std::vector<float> residual(d);
        
        for (idx_t j = 0; j < nprobe; j++) {
            idx_t list_no = assign[i * nprobe + j];
            if (list_no < 0 || list_no >= (idx_t)nlist) {
                continue;
            }
            
            quantizer->compute_residual(x + i * d, residual.data(), list_no);
            
            if (metric_type == METRIC_L2) {
                pq.compute_distance_table(residual.data(), distance_table.data());
            } else {
                pq.compute_inner_prod_table(residual.data(), distance_table.data());
            }
            
            size_t list_size = invlists->list_size(list_no);
            InvertedLists::ScopedCodes scoped_codes(invlists, list_no);
            InvertedLists::ScopedIds scoped_ids(invlists, list_no);
            
            std::vector<float> list_distances(list_size);
            for (size_t l = 0; l < list_size; l++) {
                const uint8_t* code = scoped_codes.get() + l * code_size;
                
                float dis = 0;
                for (size_t m = 0; m < pq.M; m++) {
                    uint8_t ci = code[m];
                    dis += distance_table[m * pq.ksub + ci];
                }
                
                if (metric_type == METRIC_L2) {
                    list_distances[l] = dis;
                } else {
                    list_distances[l] = -dis;
                }
            }
            
            idx_t base_idx = i * k;
            for (size_t l = 0; l < list_size && l < (size_t)k; l++) {
                float dis = list_distances[l];
                idx_t id = scoped_ids.get()[l];
                
                if (metric_type == METRIC_L2) {
                    if (dis < distances[base_idx + k - 1] || distances[base_idx + k - 1] == INFINITY) {
                        distances[base_idx + k - 1] = dis;
                        labels[base_idx + k - 1] = store_pairs ? (list_no << 32 | l) : id;
                        
                        for (idx_t m = k - 1; m > 0; m--) {
                            if (distances[base_idx + m] < distances[base_idx + m - 1]) {
                                std::swap(distances[base_idx + m], distances[base_idx + m - 1]);
                                std::swap(labels[base_idx + m], labels[base_idx + m - 1]);
                            }
                        }
                    }
                } else {
                    if (dis > distances[base_idx + k - 1] || distances[base_idx + k - 1] == -INFINITY) {
                        distances[base_idx + k - 1] = dis;
                        labels[base_idx + k - 1] = store_pairs ? (list_no << 32 | l) : id;
                        
                        for (idx_t m = k - 1; m > 0; m--) {
                            if (distances[base_idx + m] > distances[base_idx + m - 1]) {
                                std::swap(distances[base_idx + m], distances[base_idx + m - 1]);
                                std::swap(labels[base_idx + m], labels[base_idx + m - 1]);
                            }
                        }
                    }
                }
            }
        }
    }
}

void NPUIndexIVFPQ::reset() {
    invlists->reset();
    ntotal = 0;
    direct_map.clear();
    
    impl_->device_invlists_codes.clear();
    impl_->device_invlists_ids.clear();
    impl_->allocateInvertedLists(nlist);
}

void NPUIndexIVFPQ::reserveMemory(size_t numVecs) {
}

size_t NPUIndexIVFPQ::reclaimMemory() {
    return 0;
}

int NPUIndexIVFPQ::getDevice() const {
    return config_.device_id;
}

NPUResources* NPUIndexIVFPQ::getResources() {
    return nullptr;
}

void NPUIndexIVFPQ::copyFrom(const IndexIVFPQ* index) {
    d = index->d;
    metric_type = index->metric_type;
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    nlist = index->nlist;
    nprobe = index->nprobe;
    pq = index->pq;
    code_size = index->code_size;
    by_residual = index->by_residual;
    
    quantizer = index->quantizer;
    own_fields = false;
    
    use_precomputed_table_ = index->use_precomputed_table;
    if (use_precomputed_table_ > 0) {
        precomputed_table_ = index->precomputed_table;
    }
    
    invlists = index->invlists;
    
    impl_->allocateInvertedLists(nlist);
}

void NPUIndexIVFPQ::copyTo(IndexIVFPQ* index) const {
    index->d = d;
    index->metric_type = metric_type;
    index->ntotal = ntotal;
    index->is_trained = is_trained;
    index->nlist = nlist;
    index->nprobe = nprobe;
    index->pq = pq;
    index->code_size = code_size;
    index->by_residual = by_residual;
    index->use_precomputed_table = use_precomputed_table_;
    
    if (use_precomputed_table_ > 0) {
        index->precomputed_table = precomputed_table_;
    }
}

void NPUIndexIVFPQ::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(is_trained);
    
    std::vector<float> residuals(n * d);
    for (idx_t i = 0; i < n; i++) {
        if (list_nos[i] >= 0) {
            quantizer->compute_residual(x + i * d, residuals.data() + i * d, list_nos[i]);
        }
    }
    
    pq.compute_codes(residuals.data(), codes, n);
}

void NPUIndexIVFPQ::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* listnos,
        float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    
    std::vector<float> residuals(n * d);
    pq.decode(codes, residuals.data(), n);
    
    for (idx_t i = 0; i < n; i++) {
        if (listnos[i] >= 0) {
            quantizer->add_to_listno(listnos[i], residuals.data() + i * d, x + i * d);
        }
    }
}

void NPUIndexIVFPQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    
    size_t coarse_size = 8;
    size_t pq_size = pq.code_size;
    
    std::vector<uint8_t> pq_codes(n * pq_size);
    std::vector<idx_t> list_nos(n);
    
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* b = bytes + i * (coarse_size + pq_size);
        list_nos[i] = *(idx_t*)b;
        std::copy(b + coarse_size, b + coarse_size + pq_size, pq_codes.data() + i * pq_size);
    }
    
    decode_vectors(n, pq_codes.data(), list_nos.data(), x);
}

void NPUIndexIVFPQ::train_encoder(idx_t n, const float* x, const idx_t* assign) {
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(assign != nullptr);
    
    std::vector<float> residuals(n * d);
    for (idx_t i = 0; i < n; i++) {
        if (assign[i] >= 0) {
            quantizer->compute_residual(x + i * d, residuals.data() + i * d, assign[i]);
        }
    }
    
    pq.train(n, residuals.data());
}

idx_t NPUIndexIVFPQ::train_encoder_num_vectors() const {
    return pq.cp.niter * pq.ksub;
}

void NPUIndexIVFPQ::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < (int64_t)nlist);
    FAISS_THROW_IF_NOT(offset >= 0);
    
    InvertedLists::ScopedCodes scoped_codes(invlists, list_no);
    const uint8_t* code = scoped_codes.get() + offset * code_size;
    
    std::vector<float> residual(d);
    pq.decode(code, residual.data(), 1);
    
    quantizer->add_to_listno(list_no, residual.data(), recons);
}

void NPUIndexIVFPQ::precompute_table() {
    if (!by_residual || metric_type != METRIC_L2) {
        return;
    }
    
    precomputed_table_.resize(nlist * pq.M * pq.ksub);
    
    for (size_t i = 0; i < nlist; i++) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(i, centroid.data());
        
        for (size_t m = 0; m < pq.M; m++) {
            for (size_t j = 0; j < pq.ksub; j++) {
                float dis = 0;
                for (size_t k = 0; k < pq.dsub; k++) {
                    float diff = centroid[m * pq.dsub + k] - 
                                pq.centroids[m * pq.ksub * pq.dsub + j * pq.dsub + k];
                    dis += diff * diff;
                }
                precomputed_table_[i * pq.M * pq.ksub + m * pq.ksub + j] = dis;
            }
        }
    }
    
    use_precomputed_table_ = 1;
}

void NPUIndexIVFPQ::setPrecomputedCodes(bool enable) {
    if (enable && by_residual && metric_type == METRIC_L2) {
        precompute_table();
    } else {
        use_precomputed_table_ = 0;
        precomputed_table_.clear();
    }
}

bool NPUIndexIVFPQ::getPrecomputedCodes() const {
    return use_precomputed_table_ > 0;
}

int NPUIndexIVFPQ::getNumSubQuantizers() const {
    return pq.M;
}

int NPUIndexIVFPQ::getBitsPerCode() const {
    return pq.nbits;
}

int NPUIndexIVFPQ::getCentroidsPerSubQuantizer() const {
    return pq.ksub;
}

InvertedListScanner* NPUIndexIVFPQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* params) const {
    return nullptr;
}

void NPUIndexIVFPQ::updateQuantizer() {
    if (use_precomputed_table_ > 0) {
        precompute_table();
    }
}

}
}
