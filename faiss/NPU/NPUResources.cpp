/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/NPU/NPUResources.h>

#include <cstdlib>
#include <cstring>

namespace faiss {
namespace npu {

NPUResources::NPUResources(int device_id)
        : device_id_(device_id),
          memory_used_(0),
          temp_memory_used_(0),
          temp_memory_limit_(1 << 20) {
}

NPUResources::~NPUResources() {
    for (auto& pair : memory_allocations_) {
        std::free(pair.first);
    }
    for (auto& pair : temp_memory_allocations_) {
        std::free(pair.first);
    }
}

void* NPUResources::allocMemory(size_t size) {
    void* ptr = std::malloc(size);
    if (ptr) {
        memory_allocations_[ptr] = size;
        memory_used_ += size;
    }
    return ptr;
}

void NPUResources::deallocMemory(void* ptr) {
    auto it = memory_allocations_.find(ptr);
    if (it != memory_allocations_.end()) {
        memory_used_ -= it->second;
        memory_allocations_.erase(it);
        std::free(ptr);
    }
}

void* NPUResources::allocTempMemory(size_t size) {
    if (temp_memory_used_ + size > temp_memory_limit_) {
        return nullptr;
    }
    void* ptr = std::malloc(size);
    if (ptr) {
        temp_memory_allocations_[ptr] = size;
        temp_memory_used_ += size;
    }
    return ptr;
}

void NPUResources::deallocTempMemory(void* ptr) {
    auto it = temp_memory_allocations_.find(ptr);
    if (it != temp_memory_allocations_.end()) {
        temp_memory_used_ -= it->second;
        temp_memory_allocations_.erase(it);
        std::free(ptr);
    }
}

void NPUResources::syncDevice() {
}

int NPUResources::getNumDevices() {
    return 1;
}

void NPUResources::setDevice(int device_id) {
}

int NPUResources::getCurrentDevice() {
    return 0;
}

StandardNPUResources::StandardNPUResources(int device_id)
        : resources_(device_id) {
}

}
}
