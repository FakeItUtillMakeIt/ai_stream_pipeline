// src/nodes/infer/cuda_stub.cpp
// 无 CUDA 平台（纯 CPU / RKNN-only / Ascend-only）的 CUDA Runtime 链接桩。
//
// 仅在 WITH_CUDA 关闭时编入（见 src/nodes/CMakeLists.txt）。此时检测引擎
// 工厂无可用后端，DetectionInferNode 走 mock 路径，理论上不会触达这些
// 符号；桩全部为 no-op（cudaMalloc 返回失败以便调用方干净退出），
// 保证节点可注册、流水线可构建。
#include "detection_infer.h"

cudaError_t cudaMalloc(void** ptr, size_t size) {
    (void)size;
    *ptr = nullptr;
    return 1;  // 非 cudaSuccess，让调用方走失败分支
}

cudaError_t cudaFree(void* /*ptr*/) {
    return cudaSuccess;
}

cudaError_t cudaMallocHostRaw(void** ptr, size_t size) {
    (void)size;
    *ptr = nullptr;
    return 1;
}

cudaError_t cudaFreeHost(void* /*ptr*/) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* /*dst*/, const void* /*src*/, size_t /*count*/,
                            cudaMemcpyKind /*kind*/, cudaStream_t /*stream*/) {
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int /*flags*/) {
    *stream = nullptr;
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t /*stream*/) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t /*stream*/) {
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t /*stream*/, int /*mode*/) {
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t /*stream*/, cudaGraph_t* graph) {
    if (graph) *graph = nullptr;
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t /*graph*/,
                                 void* /*error_node*/, void* /*log_buffer*/, size_t /*log_size*/) {
    if (exec) *exec = nullptr;
    return 1;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t /*exec*/, cudaStream_t /*stream*/) {
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t /*exec*/) {
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t /*graph*/) {
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t /*error*/) {
    return "CUDA stub (no CUDA support in this build)";
}

namespace ai_stream {
namespace nodes {

// 以下成员在 detection_infer.cpp 中整体位于 WITH_CUDA 守卫内，
// 无 CUDA 构建在此提供定义：initEngine 直接进入 mock 模式
// （engine_ 保持为空，processBatch 生成假检测框）。
bool DetectionInferNode::initEngine(const std::string& engine_path) {
    (void)engine_path;
    LOG_WARN("[DetectionInfer] No inference backend available on this "
             "platform, running in mock mode");
    return true;
}

bool DetectionInferNode::allocatePinnedMemory() { return false; }
void DetectionInferNode::freePinnedMemory() {}
bool DetectionInferNode::captureCudaGraph(int batch_size) { (void)batch_size; return false; }
bool DetectionInferNode::executeCudaGraph() { return false; }
void DetectionInferNode::destroyCudaGraph() {}

} // namespace nodes
} // namespace ai_stream
