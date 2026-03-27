/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_NPU_NPU_OPS_H
#define FAISS_NPU_NPU_OPS_H

#include <vector>

namespace faiss {
namespace npu {

struct NPUPQDistanceOp {
    static void compute_distance_table(
            int M,
            int ksub,
            int dsub,
            const float* query,
            const float* centroids,
            float* distance_table,
            bool is_l2_metric);
    
    static void search_pq(
            int n,
            int k,
            int M,
            int ksub,
            int ntotal,
            const float* distance_table,
            const uint8_t* codes,
            float* distances,
            int* labels,
            bool is_l2_metric);
    
    static void compute_codes(
            int n,
            int M,
            int ksub,
            int dsub,
            const float* x,
            const float* centroids,
            uint8_t* codes);
};

struct NPUIVFPQSearchOp {
    static void search_ivfpq(
            int n,
            int k,
            int nprobe,
            int M,
            int ksub,
            const float* queries,
            const float* coarse_centroids,
            const float* pq_centroids,
            const int* list_offsets,
            const uint8_t* list_codes,
            const int* list_ids,
            float* distances,
            int* labels,
            bool is_l2_metric);
    
    static void precompute_residual_table(
            int nlist,
            int M,
            int ksub,
            int dsub,
            const float* coarse_centroids,
            const float* pq_centroids,
            float* precomputed_table);
};

struct NPUKernels {
    static void l2_distance(
            int n,
            int d,
            const float* x,
            const float* y,
            float* distances);
    
    static void inner_product(
            int n,
            int d,
            const float* x,
            const float* y,
            float* distances);
    
    static void kmeans_assign(
            int n,
            int d,
            int k,
            const float* x,
            const float* centroids,
            int* assign);
    
    static void compute_residual(
            int n,
            int d,
            const float* x,
            const float* centroid,
            float* residuals);
};

}
}

#endif
