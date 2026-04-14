// src/nodes/registry/node_registry.cpp
#include "node_factory.h"

namespace ai_stream {
namespace core {

// 显式实例化单例，确保静态成员在程序中只有一份
// 实际上 NodeFactory 的实现都在头文件中，此文件主要为了满足某些链接器要求
// 或者可以留空，将单例定义为 inline static 成员 (C++17)

} // namespace core
} // namespace ai_stream