cmake_minimum_required(VERSION 3.14)
project(test_ivfpq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

set(FAISS_INCLUDE_DIR "/home/yzc/faiss")
set(FAISS_LIBRARY "/home/yzc/faiss/buildDebug/faiss/libfaiss.so")

add_executable(test_ivfpq
    main.cpp
    Config.cpp
    VectorGenerator.cpp
    IndexTest.cpp
)

target_include_directories(test_ivfpq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FAISS_INCLUDE_DIR}
)

target_link_libraries(test_ivfpq PRIVATE
    ${FAISS_LIBRARY}
)
