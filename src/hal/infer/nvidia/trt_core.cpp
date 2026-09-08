// src/hal/tensorrt/trt_core.cpp
#include "trt_core.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <unordered_map>

namespace ai_stream {
namespace hal {

struct TrtCore::Impl {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    void* stream = nullptr;
    bool loaded = false;

    // kUSER_MANAGED 策略下由本类预分配的激活内存 workspace
    void* workspace = nullptr;
    size_t workspace_size = 0;

    struct Buffer {
        void* ptr = nullptr;
        size_t bytes = 0;
    };
    std::unordered_map<std::string, Buffer> buffers;

    ~Impl() {
        for (auto& [name, buf] : buffers) {
            if (buf.ptr) cudaFree(buf.ptr);
        }
        if (workspace) cudaFree(workspace);
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
};

TrtCore::TrtCore() : impl_(new Impl()) {}
TrtCore::~TrtCore() = default;

bool TrtCore::loadEngine(const std::string& engine_path, const char* log_tag) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        LOG_ERROR_FMT("[{}] Cannot open engine: {}", log_tag, engine_path);
        return false;
    }

    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), static_cast<std::streamsize>(size))) {
        LOG_ERROR_FMT("[{}] Cannot read engine: {}", log_tag, engine_path);
        return false;
    }

    // 释放旧资源（支持重加载）
    if (impl_->context) { delete impl_->context; impl_->context = nullptr; }
    if (impl_->engine) { delete impl_->engine; impl_->engine = nullptr; }
    if (impl_->runtime) { delete impl_->runtime; impl_->runtime = nullptr; }
    impl_->loaded = false;

    impl_->runtime = nvinfer1::createInferRuntime(g_logger);
    if (!impl_->runtime) {
        LOG_ERROR_FMT("[{}] createInferRuntime failed", log_tag);
        return false;
    }

    impl_->engine = impl_->runtime->deserializeCudaEngine(buffer.data(), size);
    if (!impl_->engine) {
        LOG_ERROR_FMT("[{}] deserializeCudaEngine failed", log_tag);
        return false;
    }

    // kUSER_MANAGED：激活内存由用户管理（CUDA Graph 捕获期间禁止 cudaMalloc，
    // 必须预分配并绑定；updateDeviceMemorySizeForShapes/setDeviceMemoryV2 也
    // 仅在该策略下可用）。默认策略(kSTATIC)会预分配最大尺寸，但无法配合
    // CUDA Graph 使用，且 enqueue 期间可能触发内部 cudaMalloc。
#if NV_TENSORRT_MAJOR >= 10
    impl_->context = impl_->engine->createExecutionContext(
        nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
#else
    impl_->context = impl_->engine->createExecutionContext();
#endif
    if (!impl_->context) {
        LOG_ERROR_FMT("[{}] createExecutionContext failed", log_tag);
        return false;
    }

    // 预分配 engine 静态最大 workspace 并绑定（覆盖全部 shape/profile），
    // 保证任意 enqueue（含 CUDA Graph 捕获）都不需要运行期 cudaMalloc。
    const size_t ws_size = static_cast<size_t>(impl_->engine->getDeviceMemorySize());
    if (ws_size > 0) {
        cudaError_t err = cudaMalloc(&impl_->workspace, ws_size);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[{}] cudaMalloc workspace ({}KB) failed: {}",
                          log_tag, ws_size / 1024, cudaGetErrorString(err));
            return false;
        }
        impl_->workspace_size = ws_size;
        impl_->context->setDeviceMemoryV2(impl_->workspace, static_cast<int64_t>(ws_size));
        LOG_INFO_FMT("[{}] Workspace allocated: {}KB (user-managed)", log_tag, ws_size / 1024);
    }

    impl_->loaded = true;

    int nb_io = impl_->engine->getNbIOTensors();
    LOG_INFO_FMT("[{}] Engine loaded: {} ({} I/O tensors)", log_tag, engine_path, nb_io);
    return true;
}

bool TrtCore::isLoaded() const { return impl_->loaded; }

std::vector<TrtTensorMeta> TrtCore::tensors() const {
    std::vector<TrtTensorMeta> result;
    if (!impl_->engine) return result;

    int nb_io = impl_->engine->getNbIOTensors();
    result.reserve(static_cast<size_t>(nb_io));
    for (int i = 0; i < nb_io; ++i) {
        const char* name = impl_->engine->getIOTensorName(i);
        TrtTensorMeta meta;
        meta.name = name;
        meta.is_input = impl_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        nvinfer1::Dims dims = impl_->engine->getTensorShape(name);
        meta.nb_dims = dims.nbDims;
        for (int j = 0; j < dims.nbDims && j < 8; ++j) {
            meta.d[j] = dims.d[j];
        }
        result.push_back(std::move(meta));
    }
    return result;
}

bool TrtCore::setInputShape(const std::string& name, const int64_t* dims, int nb_dims) {
    if (!impl_->context || !dims || nb_dims <= 0 || nb_dims > nvinfer1::Dims::MAX_DIMS) return false;
    nvinfer1::Dims shape{};
    shape.nbDims = nb_dims;
    for (int i = 0; i < nb_dims && i < nvinfer1::Dims::MAX_DIMS; ++i) {
        shape.d[i] = static_cast<int64_t>(dims[i]);
    }
    return impl_->context->setInputShape(name.c_str(), shape);
}

bool TrtCore::setAddress(const std::string& name, void* device_ptr) {
    if (!impl_->context || !device_ptr) return false;
    return impl_->context->setTensorAddress(name.c_str(), device_ptr);
}

void* TrtCore::allocBuffer(const std::string& tensor_name, size_t bytes) {
    if (bytes == 0) return nullptr;

    auto& buf = impl_->buffers[tensor_name];
    if (!buf.ptr || buf.bytes < bytes) {
        if (buf.ptr) cudaFree(buf.ptr);
        cudaError_t err = cudaMalloc(&buf.ptr, bytes);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[TrtCore] cudaMalloc '{}' ({} bytes) failed: {}",
                          tensor_name, bytes, cudaGetErrorString(err));
            buf.ptr = nullptr;
            buf.bytes = 0;
            return nullptr;
        }
        buf.bytes = bytes;
    }
    setAddress(tensor_name, buf.ptr);
    return buf.ptr;
}

void* TrtCore::buffer(const std::string& tensor_name) const {
    auto it = impl_->buffers.find(tensor_name);
    return it != impl_->buffers.end() ? it->second.ptr : nullptr;
}

size_t TrtCore::bufferSize(const std::string& tensor_name) const {
    auto it = impl_->buffers.find(tensor_name);
    return it != impl_->buffers.end() ? it->second.bytes : 0;
}

void TrtCore::setStream(void* stream) { impl_->stream = stream; }
void* TrtCore::stream() const { return impl_->stream; }

bool TrtCore::enqueue() {
    if (!impl_->context) return false;
    // kUSER_MANAGED 下每次执行前确保绑定的是本类自管 workspace。
    // （CUDA Graph 路径可能用更大的临时 workspace 替换过绑定，回退普通
    //   enqueue 时必须恢复，否则可能指向已释放的内存。）
    if (impl_->workspace) {
        impl_->context->setDeviceMemoryV2(impl_->workspace,
                                          static_cast<int64_t>(impl_->workspace_size));
    }
    return impl_->context->enqueueV3(static_cast<cudaStream_t>(impl_->stream));
}

bool TrtCore::synchronize() {
    if (!impl_->stream) return true;
    return cudaStreamSynchronize(static_cast<cudaStream_t>(impl_->stream)) == cudaSuccess;
}

size_t TrtCore::getDeviceMemorySize() const {
    if (!impl_->engine) return 0;
    return static_cast<size_t>(impl_->engine->getDeviceMemorySize());
}

size_t TrtCore::updateDeviceMemorySizeForShapes() {
    if (!impl_->context) return 0;
    return impl_->context->updateDeviceMemorySizeForShapes();
}

bool TrtCore::setDeviceMemory(void* ptr) {
    if (!impl_->context) return false;
    impl_->context->setDeviceMemory(ptr);
    return true;
}

bool TrtCore::setDeviceMemoryV2(void* ptr, int64_t size) {
    if (!impl_->context) return false;
    impl_->context->setDeviceMemoryV2(ptr, size);
    return true;
}

void* TrtCore::workspace() const { return impl_->workspace; }

size_t TrtCore::workspaceSize() const { return impl_->workspace_size; }

void* TrtCore::context() const { return impl_->context; }
void* TrtCore::engine() const { return impl_->engine; }

} // namespace hal
} // namespace ai_stream
