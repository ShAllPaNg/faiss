# Faiss IndexIVFPQ Technical Notes

## 1. IndexPQ Overview

### Core Structure
```
IndexPQ : IndexFlatCodes
├── ProductQuantizer pq  // Product quantizer
├── search_type          // Search type
├── polysemous_ht        // Polysemous Hamming threshold
└── codes                // Stored codes
```

### ProductQuantizer Principle
- **Dimension Split**: Divide d-dimensional vector into M subspaces, each with d/M dimensions
- **Sub-quantizers**: Each subspace has 2^nbits centroids (typically nbits=8, i.e., 256 centroids)
- **Encoding**: Vector encoded as M indices, each nbits

### Training Process (IndexPQ.cpp:50-72)
1. **Standard Training**: Call `pq.train(n, x)` using k-means clustering
2. **Polysemous Training**: Optimize centroids for better Hamming distance effectiveness

### Search Modes (IndexPQ.h:64-71)
- `ST_PQ`: Asymmetric PQ search (default)
- `ST_HE`: Hamming distance
- `ST_generalized_HE`: Generalized Hamming distance
- `ST_SDC`: Symmetric PQ search
- `ST_polysemous`: Hamming filter + PQ combination
- `ST_polysemous_generalize`: Generalized Hamming filter + PQ

### Core Search Flow (IndexPQ.cpp:164-270)
1. **ST_PQ mode**:
   - Compute query vector's distance table (M×ksub)
   - For each database code, look up and accumulate distances
   - Maintain top-k heap

2. **Polysemous search** (IndexPQ.cpp:332-436):
   - First compute query code
   - Use Hamming distance for fast filtering (threshold filtering)
   - Compute exact PQ distance for vectors passing filter

### Encoding/Decoding (IndexPQ.cpp:440-446)
```cpp
sa_encode: pq.compute_codes(x, bytes, n)  // Vector→Code
sa_decode: pq.decode(bytes, x, n)          // Code→Vector
```

### Performance Optimizations
- **Distance Table Pre-computation**: Avoid repeated sub-vector to centroid distance calculations
- **SIMD Instructions**: Accelerate distance calculations
- **Parallelization**: OpenMP multi-threading for multiple queries
- **Polysemous**: Hamming distance filtering reduces exact computation

### Typical Parameters
- d=128, M=8, nbits=8 → code_size=8 bytes
- Compression ratio: 128×4 bytes → 8 bytes, ~64x compression

---

## 2. IndexIVFPQ Overview

### Core Structure (IndexIVFPQ.h:34-151)
```
IndexIVFPQ : IndexIVF
├── ProductQuantizer pq           // Product quantizer
├── by_residual = true            // Encode residual vectors by default
├── use_precomputed_table         // Precomputed table mode
├── precomputed_table             // Precomputed distance table
├── polysemous_ht                 // Polysemous Hamming threshold
└── scan_table_threshold          // Scan threshold
```

### Core Concept
**Two-level Quantization**:
1. **Coarse Quantization**: Use quantizer (e.g., IndexFlat) to find nearest cluster center
2. **Fine Quantization**: Use PQ to encode **residual vectors** (original vector - cluster center)

### Adding Vectors Flow (IndexIVFPQ.cpp:225-327)
```cpp
add_core_o():
  1. quantizer->assign(n, x, idx)           // Find cluster for each vector
  2. compute_residuals(x, idx)              // Compute residual: x - centroid
  3. pq.compute_codes(residuals, codes)     // PQ encode residuals
  4. invlists->add_entry(key, id, code)     // Store in inverted list
```

### Search Flow

**Distance Calculation Decomposition** (IndexIVFPQ.cpp:349-377):

For L2 distance `d = ||x - y_C - y_R||²` (x=query, y_C=coarse centroid, y_R=fine PQ centroid):

```
d = ||x - y_C||²  +  ||y_R||²  +  2*(y_C|y_R)  -  2*(x|y_R)
     term1            term2       term3           term4
```

- **term1**: Already computed during coarse search
- **term2+term3**: Can be precomputed (independent of query x) → `precomputed_table`
- **term4**: Computed at query time (distance table)

### Precomputed Table (IndexIVFPQ.cpp:379-485)
```cpp
use_precomputed_table:
  0: Not used (default)
  1: Standard precomputed table (nlist × M × ksub)
  2: Compact version (for MultiIndexQuantizer)
```

### Scanner Implementations (IndexIVFPQ.cpp:804-1252)
`IVFPQScanner` template class implements three scanning modes:

1. **scan_list_with_table** (precompute_mode=2)
   - Use precomputed distance table
   - 4-way unrolling optimization

2. **scan_list_with_pointer** (precompute_mode=1)
   - Use table pointers to reduce memory access

3. **scan_on_the_fly_dist** (precompute_mode=0)
   - Real-time decoding and computation, no precomputation

### Polysemous Search (IndexIVFPQ.cpp:1040-1157)
```cpp
scan_list_polysemous_hc():
  1. Compute query code q_code
  2. For each database code, compute Hamming distance
  3. If hd < polysemous_ht, compute exact PQ distance
  4. 4-way unrolling + SIMD optimization
```

### Encoding/Decoding (IndexIVFPQ.cpp:97-134)
```cpp
encode(key, x, code):
  residual = x - centroid[key]
  pq.compute_code(residual, code)

decode_multiple(keys, codes, x):
  pq.decode(codes, x)
  x += centroid[keys]  // Add back centroid
```

### Performance Optimizations

| Technique | Purpose |
|-----------|---------|
| Precomputed Table | Avoid repeated centroid-related computation |
| 4-way Unrolling | Reduce branch prediction overhead |
| SIMD | Accelerate distance calculation |
| Polysemous Filtering | Fast Hamming distance filtering |
| OpenMP | Multi-threading |

### Typical Parameters
```
d=128, nlist=4096, M=8, nbits=8
code_size = 8 bytes
Compression ratio: 128×4 → 8 bytes (64x)
```

### Comparison with IndexPQ
| Feature | IndexPQ | IndexIVFPQ |
|---------|---------|------------|
| Search scope | Full database | Only relevant clusters |
| Encoded object | Original vector | Residual vector |
| Accuracy | Lower | Higher |
| Speed | Slower | Faster (with small nprobe) |

---

## 3. by_residual Parameter

### Setting
```cpp
by_residual = true   (default): Encode residuals
by_residual = false: Encode original vectors
```

### Code Location
IndexIVFPQ.cpp:57 defaults `by_residual = true`, can be manually set to false.

### Differences
| Mode | Encoded Object | Distance Calculation |
|------|----------------|---------------------|
| `by_residual=true` | `x - centroid` | Residual + centroid decomposition |
| `by_residual=false` | `x` (original vector) | Direct PQ distance |

### Key Code Logic
```cpp
// Encoding (IndexIVFPQ.cpp:97-104)
void IndexIVFPQ::encode(idx_t key, const float* x, uint8_t* code) const {
    if (by_residual) {
        quantizer->compute_residual(x, residual, key);
        pq.compute_code(residual, code);
    } else {
        pq.compute_code(x, code);  // Direct encode original vector
    }
}

// L2 Search (IndexIVFPQ.cpp:577-583)
void init_query_L2() {
    if (!by_residual) {
        pq.compute_distance_table(qi, sim_table);  // Use query vector directly
    }
}
```

### When to Not Use Residual
1. **Inner Product Distance**: Residuals don't make sense for IP distance
2. **Uniform Data Distribution**: When cluster centers are not obvious
3. **Simplified Implementation**: When precomputed table is not needed

### Note
When `by_residual=false`, `use_precomputed_table` is disabled (IndexIVFPQ.cpp:396-402), as precomputed table is optimized for residual mode.

