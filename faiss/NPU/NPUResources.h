/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_NPU_NPU_RESOURCES_H
#define FAISS_NPU_NPU_RESOURCES_H

#include <memory>
#include <vector>
#include <unordered_map>

namespace faiss {
namespace npu {

class NPUResources {
public:
    NPUResources(int device_id = 0);
    ~NPUResources();
    
    int getDeviceId() const { return device_id_; }
    
    void* allocMemory(size_t size);
    void deallocMemory(void* ptr);
    
    void* allocTempMemory(size_t size);
    void deallocTempMemory(void* ptr);
    
    void syncDevice();
    
    size_t getMemoryUsed() const { return memory_used_; }
    size_t getTempMemoryUsed() const { return temp_memory_used_; }
    
    static int getNumDevices();
    static void setDevice(int device_id);
    static int getCurrentDevice();

private:
    int device_id_;
    size_t memory_used_;
    size_t temp_memory_used_;
    size_t temp_memory_limit_;
    
    std::unordered_map<void*, size_t> memory_allocations_;
    std::unordered_map<void*, size_t> temp_memory_allocations_;
};

class NPUResourcesProvider {
public:
    virtual ~NPUResourcesProvider() = default;
    virtual NPUResources* getResources() = 0;
};

class StandardNPUResources : public NPUResourcesProvider {
public:
    StandardNPUResources(int device_id = 0);
    ~StandardNPUResources() override = default;
    
    NPUResources* getResources() override { return &resources_; }

private:
    NPUResources resources_;
};

} 
} 

#endif
