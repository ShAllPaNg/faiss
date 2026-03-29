# IndexIVFPQ Test Project

A C++ test project for Faiss IndexIVFPQ with HNSW quantizer.

## Features

- Configurable vector generation (train, database, query vectors)
- IndexIVFPQ with IndexHNSW as first-stage quantizer
- Complete workflow: train → add → save → load → search → rerank
- INI configuration file support

## Build

```bash
mkdir -p build && cd build
cmake .. -Dfaiss_DIR=/path/to/faiss/cmake
make -j
```

## Usage

```bash
# Generate default config file
./test_ivfpq -g

# Edit config.ini as needed
vim config.ini

# Run test
./test_ivfpq -c config.ini
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-c, --config <file>` | Config file path (default: config.ini) |
| `-g, --generate` | Generate default config file and exit |
| `-h, --help` | Show help message |

## Configuration

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| [vector] | dimension | 128 | Vector dimension |
| [vector] | num_train_vecs | 100000 | Number of training vectors |
| [vector] | num_db_vecs | 1000000 | Number of database vectors |
| [vector] | num_query_vecs | 1000 | Number of query vectors |
| [ivf] | nlist | 1000 | Number of IVF clusters |
| [ivf] | nprobe | 32 | Number of probes for search |
| [ivf] | by_residual | true | Use residual encoding |
| [pq] | m | 8 | Number of PQ sub-quantizers |
| [pq] | nbits | 8 | Bits per sub-quantizer |
| [hnsw] | hnsw_m | 32 | HNSW M parameter |
| [hnsw] | hnsw_ef_search | 64 | HNSW efSearch parameter |
| [search] | top_k | 100 | Number of results to return |
| [search] | rerank_k | 200 | Number of candidates for reranking |
| [io] | index_file_path | ivfpq_index.bin | Index file path |
