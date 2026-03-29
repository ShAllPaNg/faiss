#include "VectorGenerator.h"
#include <random>
#include <cstring>

VectorGenerator::VectorGenerator(int32_t dimension, uint32_t seed)
    : dimension_(dimension), seed_(seed)
{
}

std::vector<float> VectorGenerator::GenerateVectors(int32_t numVectors)
{
    std::vector<float> vectors(static_cast<size_t>(numVectors) * dimension_);
    
    std::mt19937 rng(seed_);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        vectors[i] = dist(rng);
    }
    
    seed_ += numVectors;
    return vectors;
}

float* VectorGenerator::GenerateVectorsRaw(int32_t numVectors, int32_t dimension, uint32_t seed)
{
    float* vectors = new float[static_cast<size_t>(numVectors) * dimension];
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    size_t totalSize = static_cast<size_t>(numVectors) * dimension;
    for (size_t i = 0; i < totalSize; ++i) {
        vectors[i] = dist(rng);
    }
    
    return vectors;
}
