cmake_minimum_required(VERSION 3.10)
project(IVFPQSearch VERSION 1.0 LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Huawei编码风格
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")

# 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Faiss路径
set(FAISS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)
set(FAISS_INCLUDE_DIR ${FAISS_PATH})
set(FAISS_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/faiss)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FAISS_INCLUDE_DIR}
)

# 链接目录
link_directories(
    ${FAISS_LIB_DIR}
)

# 源文件
set(SOURCES
    src/config_loader.cpp
    src/vector_generator.cpp
    src/ivfpq_index.cpp
    src/concurrent_search.cpp
    src/re_ranking.cpp
)

# 头文件
set(HEADERS
    include/types.h
    include/config_loader.h
    include/vector_generator.h
    include/ivfpq_index.h
    include/concurrent_search.h
    include/re_ranking.h
)

# 构建静态库
add_library(ivfpq_search STATIC ${SOURCES} ${HEADERS})

# 链接Faiss库
target_link_libraries(ivfpq_search
    faiss
    pthread
    openblas
    gomp
)

# 构建测试程序
add_executable(ivfpq_search_test
    tests/test_main.cpp
)

target_link_libraries(ivfpq_search_test
    ivfpq_search
)

# 安装规则
install(TARGETS ivfpq_search ivfpq_search_test
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/ivfpq_search
    FILES_MATCHING PATTERN "*.h"
)

# 打印配置信息
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  IVFPQ Search Configuration")
message(STATUS "========================================")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Faiss path:       ${FAISS_PATH}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")
