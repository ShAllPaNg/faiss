/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/NPU/NPUOps.h>

#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

namespace faiss {
namespace npu {

void NPUPQDistanceOp::compute_distance_table(
        int M,
        int ksub,
        int dsub,
        const float* query,
        const float* centroids,
        float* distance_table,
        bool is_l2_metric) {
    for (int m = 0; m < M; m++) {
        const float* sub_query = query + m * dsub;
        float* sub_table = distance_table + m * ksub;
        
        for (int j = 0; j < ksub; j++) {
            const float* centroid = centroids + m * ksub * dsub + j * dsub;
            float dist = 0;
            
            if (is_l2_metric) {
                for (int d = 0; d < dsub; d++) {
                    float diff = sub_query[d] - centroid[d];
                    dist += diff * diff;
                }
            } else {
                for (int d = 0; d < dsub; d++) {
                    dist += sub_query[d] * centroid[d];
                }
            }
            
            sub_table[j] = dist;
        }
    }
}

void NPUPQDistanceOp::search_pq(
        int n,
        int k,
        int M,
        int ksub,
        int ntotal,
        const float* distance_table,
        const uint8_t* codes,
        float* distances,
        int* labels,
        bool is_l2_metric) {
    for (int i = 0; i < n; i++) {
        const float* query_table = distance_table + i * M * ksub;
        float* query_distances = distances + i * k;
        int* query_labels = labels + i * k;
        
        for (int j = 0; j < k; j++) {
            query_distances[j] = is_l2_metric ? 
                std::numeric_limits<float>::max() : 
                -std::numeric_limits<float>::max();
            query_labels[j] = -1;
        }
        
        for (int j = 0; j < ntotal; j++) {
            const uint8_t* code = codes + j * M;
            float dist = 0;
            
            for (int m = 0; m < M; m++) {
                dist += query_table[m * ksub + code[m]];
            }
            
            if (is_l2_metric) {
                if (dist < query_distances[k - 1]) {
                    query_distances[k - 1] = dist;
                    query_labels[k - 1] = j;
                    
                    for (int l = k - 1; l > 0; l--) {
                        if (query_distances[l] < query_distances[l - 1]) {
                            std::swap(query_distances[l], query_distances[l - 1]);
                            std::swap(query_labels[l], query_labels[l - 1]);
                        }
                    }
                }
            } else {
                if (dist > query_distances[k - 1]) {
                    query_distances[k - 1] = dist;
                    query_labels[k - 1] = j;
                    
                    for (int l = k - 1; l > 0; l--) {
                        if (query_distances[l] > query_distances[l - 1]) {
                            std::swap(query_distances[l], query_distances[l - 1]);
                            std::swap(query_labels[l], query_labels[l - 1]);
                        }
                    }
                }
            }
        }
    }
}

void NPUPQDistanceOp::compute_codes(
            int n,
            int M,
            int ksub,
            int dsub,
            const float* x,
            const float* centroids,
            uint8_t* codes) {
    for (int i = 0; i < n; i++) {
        for (int m = 0; m < M; m++) {
            const float* sub_x = x + i * M * dsub + m * dsub;
            const float* sub_centroids = centroids + m * ksub * dsub;
            
            float min_dist = std::numeric_limits<float>::max();
            int best_idx = 0;
            
            for (int j = 0; j < ksub; j++) {
                const float* centroid = sub_centroids + j * dsub;
                float dist = 0;
                
                for (int d = 0; d < dsub; d++) {
                    float diff = sub_x[d] - centroid[d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = j;
                }
            }
            
            codes[i * M + m] = (uint8_t)best_idx;
        }
    }
}

void NPUIVFPQSearchOp::search_ivfpq(
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
            bool is_l2_metric) {
}

void NPUIVFPQSearchOp::precompute_residual_table(
            int nlist,
            int M,
            int ksub,
            int dsub,
            const float* coarse_centroids,
            const float* pq_centroids,
            float* precomputed_table) {
}

void NPUKernels::l2_distance(
            int n,
            int d,
            const float* x,
            const float* y,
            float* distances) {
    for (int i = 0; i < n; i++) {
        float dist = 0;
        for (int j = 0; j < d; j++) {
            float diff = x[j] - y[j];
            dist += diff * diff;
        }
        distances[i] = dist;
    }
}

void NPUKernels::inner_product(
            int n,
            int d,
            const float* x,
            const float* y,
            float* distances) {
    for (int i = 0; i < n; i++) {
        float prod = 0;
        for (int j = 0; j < d; j++) {
            prod += x[j] * y[j];
        }
        distances[i] = prod;
    }
}

void NPUKernels::kmeans_assign(
            int n,
            int d,
            int k,
            const float* x,
            const float* centroids,
            int* assign) {
    for (int i = 0; i < n; i++) {
        const float* xi = x + i * d;
        float min_dist = std::numeric_limits<float>::max();
        int best_idx = 0;
        
        for (int j = 0; j < k; j++) {
            const float* centroid = centroids + j * d;
            float dist = 0;
            
            for (int l = 0; l < d; l++) {
                float diff = xi[l] - centroid[l];
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        
        assign[i] = best_idx;
    }
}

void NPUKernels::compute_residual(
            int n,
            int d,
            const float* x,
            const float* centroid,
            float* residuals) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            residuals[i * d + j] = x[i * d + j] - centroid[j];
        }
    }
}

}
}
