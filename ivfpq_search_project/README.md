# IndexIVFPQ Vector Search System

基于 Faiss IndexIVFPQ 的向量检索系统，支持向量生成、量化、并发搜索和重排序。

## 功能特性

1. **向量生成**：支持生成多种分布（uniform、normal、orthogonal）的随机向量
2. **索引构建**：使用 IndexIVFPQ 进行向量量化和索引构建
3. **并发检索**：支持多个 query 并发检索
4. **重排序**：基于精确 L2 距离的结果重排序

## 项目结构

```
ivfpq_search_project/
├── CMakeLists.txt          # 构建配置
├── README.md               # 项目说明
├── include/                # 头文件
│   ├── types.h             # 公共数据结构
│   ├── config_loader.h     # 配置文件加载
│   ├── vector_generator.h  # 向量生成器
│   ├── ivfpq_index.h       # IndexIVFPQ封装
│   ├── concurrent_search.h # 并发搜索
│   └── re_ranking.h        # 重排序
├── src/                    # 源文件
│   ├── config_loader.cpp
│   ├── vector_generator.cpp
│   ├── ivfpq_index.cpp
│   ├── concurrent_search.cpp
│   └── re_ranking.cpp
├── config/                 # 配置文件
│   └── default.ini         # 默认配置
├── tests/                  # 测试程序
│   └── test_main.cpp
└── data/                   # 数据目录
    ├── raw_vectors/        # 原始向量
    ├── pq_codes/           # PQ代码
    ├── index/              # 索引文件
    └── results/            # 结果文件
```

## 编译

```bash
cd ivfpq_search_project
mkdir build && cd build
cmake ..
make -j4
```

## 使用方法

```bash
# 使用默认配置文件运行测试
./bin/ivfpq_search_test

# 使用自定义配置文件
./bin/ivfpq_search_test /path/to/config.ini
```

## 配置文件说明

配置文件采用 INI 格式，主要包含以下部分：

- **[vector]**: 向量生成配置（维度、数量、分布等）
- **[index]**: IndexIVFPQ 配置（nlist、M、nbits 等）
- **[search]**: 搜索配置（topk、线程数等）
- **[path]**: 文件路径配置
- **[advanced]**: 高级配置（日志级别等）

详见 `config/default.ini`。

## 编码风格

本项目采用华为编码风格：
- 类名使用大驼峰
- 函数名使用大驼峰
- 变量名使用小驼峰
- 成员变量使用 `m_` 前缀
- 全局变量使用 `g_` 前缀
- 常量使用全大写 + 下划线
- 4空格缩进
- 行宽不超过120字符
