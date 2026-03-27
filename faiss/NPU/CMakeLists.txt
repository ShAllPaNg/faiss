cmake_minimum_required(VERSION 3.10)
project(faiss_npu)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(FAISS_NPU_SOURCES
    NPUIndexPQ.cpp
    NPUIndexIVFPQ.cpp
    NPUResources.cpp
    NPUOps.cpp
)

add_library(faiss_npu STATIC ${FAISS_NPU_SOURCES})

target_include_directories(faiss_npu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(faiss_npu
    OpenMP::OpenMP_CXX
)

target_compile_options(faiss_npu PRIVATE
    -Wall
    -Wextra
    -O3
    -fPIC
)

install(TARGETS faiss_npu
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    NPUIndexPQ.h
    NPUIndexIVFPQ.h
    NPUResources.h
    NPUOps.h
    DESTINATION include/faiss/NPU
)
