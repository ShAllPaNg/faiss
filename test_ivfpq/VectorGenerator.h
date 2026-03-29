#ifndef VECTOR_GENERATOR_H
#define VECTOR_GENERATOR_H

#include <vector>
#include <cstdint>

class VectorGenerator {
public:
    VectorGenerator(int32_t dimension, uint32_t seed = 42);
    
    std::vector<float> GenerateVectors(int32_t numVectors);
    
    static float* GenerateVectorsRaw(int32_t numVectors, int32_t dimension, uint32_t seed = 42);
    
    int32_t GetDimension() const { return dimension_; }

private:
    int32_t dimension_;
    uint32_t seed_;
};

#endif
