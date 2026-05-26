// src/nodes/preprocess/gpu_resize_normalize.cu
#include "gpu_resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"

#include <cuda_runtime.h>
#include <npp.h>
#include <device_launch_parameters.h>

namespace ai_stream {
namespace nodes {

namespace {

// 简单的 CUDA 内核用于归一化
__global__ void normalizeKernel(float* dst, const unsigned char* src,
                                int width, int height, int channels,
                                float mean_b, float mean_g, float mean_r,
                                float std_b, float std_g, float std_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        dst[idx + 0] = (src[idx + 0] / 255.0f - mean_b) / std_b; // B
        dst[idx + 1] = (src[idx + 1] / 255.0f - mean_g) / std_g; // G
        dst[idx + 2] = (src[idx + 2] / 255.0f - mean_r) / std_r; // R
    }
}

} // namespace

GPUResizeNormalizeNode::GPUResizeNormalizeNode()
    : IGpuPreprocessNode("GPUResizeNormalize"),
      device_id_(0),
      async_processing_(false),
      stream_(nullptr),
      input_gpu_ptr_(nullptr),
      output_gpu_ptr_(nullptr),
      gpu_buffer_size_(0),
      total_processed_(0),
      total_latency_ms_(0.0f) {

    // 初始化 CUDA 设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GPUResizeNormalize] Node created with GPU device %d", device_id_);
    } else {
        LOG_WARN_FMT("[GPUResizeNormalize] No GPU device found, will fallback to CPU");
    }

    // 创建 CUDA 事件用于计时
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

GPUResizeNormalizeNode::~GPUResizeNormalizeNode() {
    stop();

    // 清理 GPU 资源
    if (input_gpu_ptr_) {
        cudaFree(input_gpu_ptr_);
    }
    if (output_gpu_ptr_) {
        cudaFree(output_gpu_ptr_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);

    LOG_INFO_FMT("[GPUResizeNormalize] Node destroyed");
}

bool GPUResizeNormalizeNode::start() {
    if (!IGpuPreprocessNode::start()) {
        return false;
    }

    // 创建 CUDA 流（如果未设置）
    if (!stream_) {
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_));
    }

    running_ = true;
    LOG_INFO_FMT("[GPUResizeNormalize] Node started");
    return true;
}

void GPUResizeNormalizeNode::stop() {
    if (!running_) return;

    running_ = false;

    // 同步所有 CUDA 操作
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }

    LOG_INFO_FMT("[GPUResizeNormalize] Node stopped");
}

void GPUResizeNormalizeNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[GpuResizeNormalize] Received stream end");
        stop();
        broadcast(packet);
        return;
    }
    if (!running_) {
        broadcast(packet);
        return;
    }

    if (packet->type != core::PacketType::DECODED_FRAME) {
        broadcast(packet);
        return;
    }

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame->mat || frame->mat->empty()) {
        LOG_WARN_FMT("[GPUResizeNormalize] Received empty frame");
        return;
    }

    // 记录处理开始时间
    cudaEventRecord(start_event_, stream_);

    // 1. 准备 GPU 缓冲区
    int src_width = frame->mat->cols;
    int src_height = frame->mat->rows;
    int src_channels = frame->mat->channels();
    size_t src_size = src_width * src_height * src_channels;

    // 分配 GPU 内存
    if (gpu_buffer_size_ < src_size) {
        // 释放旧缓冲区
        if (input_gpu_ptr_) cudaFree(input_gpu_ptr_);
        if (output_gpu_ptr_) cudaFree(output_gpu_ptr_);

        // 分配新缓冲区
        CUDA_CHECK(cudaMalloc(&input_gpu_ptr_, src_size));
        CUDA_CHECK(cudaMalloc(&output_gpu_ptr_, src_size * sizeof(float))); // float 格式

        gpu_buffer_size_ = src_size;
        LOG_INFO_FMT("[GPUResizeNormalize] Allocated GPU buffer: %zu bytes", src_size * (1 + sizeof(float)));
    }

    // 2. 将数据从主机内存复制到 GPU
    CUDA_CHECK(cudaMemcpyAsync(static_cast<void*>(input_gpu_ptr_), frame->mat->data, src_size,
                               cudaMemcpyHostToDevice, stream_));

    // 3. 计算目标尺寸（考虑宽高比）
    int dst_width = target_width_;
    int dst_height = target_height_;

    if (keep_aspect_ratio_) {
        float src_aspect = static_cast<float>(src_width) / src_height;
        float dst_aspect = static_cast<float>(target_width_) / target_height_;

        if (src_aspect > dst_aspect) {
            dst_width = target_width_;
            dst_height = static_cast<int>(dst_width / src_aspect);
        } else {
            dst_height = target_height_;
            dst_width = static_cast<int>(dst_height * src_aspect);
        }
    }

    // 4. 使用 NPP 进行 GPU 加速的缩放和颜色空间转换
    NppiSize src_size_npp = {src_width, src_height};
    NppiRect src_roi = {0, 0, src_width, src_height};
    NppiSize dst_size_npp = {dst_width, dst_height};
    NppiRect dst_roi = {0, 0, dst_width, dst_height};

    // 使用 NPP 进行 BGR 到 RGB 的转换和缩放
    // 注意：实际应用中需要根据 NPP 的 API 进行正确的格式转换
    NppStatus status = nppiResize_8u_C3R(
        static_cast<const Npp8u*>(input_gpu_ptr_), src_size_npp.width * 3,
        src_size_npp, src_roi,
        static_cast<Npp8u*>(input_gpu_ptr_), dst_size_npp.width * 3,
        dst_size_npp, dst_roi,
        NPPI_INTER_LINEAR
    );

    if (status != NPP_SUCCESS) {
        LOG_ERROR_FMT("[GPUResizeNormalize] NPP resize failed: %d", int(status));
        return;
    }

    // 5. 归一化操作
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);

    normalizeKernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<float*>(output_gpu_ptr_),
        static_cast<const unsigned char*>(input_gpu_ptr_),
        dst_width, dst_height, 3,
        mean_[0], mean_[1], mean_[2],
        std_[0], std_[1], std_[2]
    );

    // 6. 将结果复制回主机内存
    size_t dst_size = dst_width * dst_height * 3 * sizeof(float);
    auto processed_mat = std::make_shared<cv::Mat>(dst_height, dst_width, CV_32FC3);

    CUDA_CHECK(cudaMemcpyAsync(processed_mat->data, static_cast<const void*>(output_gpu_ptr_), dst_size,
                               cudaMemcpyDeviceToHost, stream_));

    // 记录处理结束时间
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);

    // 计算处理延迟
    float latency_ms;
    cudaEventElapsedTime(&latency_ms, start_event_, stop_event_);
    total_latency_ms_+=static_cast<int64_t>(latency_ms);
    total_processed_++;

    // 7. 构造输出包
    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->source_id = frame->source_id;
    new_packet->mat = processed_mat;
    new_packet->width = dst_width;
    new_packet->height = dst_height;
    new_packet->channels = 3;

    broadcast(new_packet);
}

void GPUResizeNormalizeNode::setGpuDeviceId(int device_id) {
    if (device_id_ != device_id) {
        device_id_ = device_id;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GPUResizeNormalize] Set GPU device ID: %d", device_id_);
    }
}

int GPUResizeNormalizeNode::getGpuDeviceId() const {
    return device_id_;
}

void GPUResizeNormalizeNode::setAsyncProcessing(bool async) {
    async_processing_ = async;
    LOG_INFO_FMT("[GPUResizeNormalize] Set async processing: %d", async);
}

void GPUResizeNormalizeNode::setCudaStream(cudaStream_t stream) {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    stream_ = stream;
    LOG_INFO_FMT("[GPUResizeNormalize] Set external CUDA stream");
}

float GPUResizeNormalizeNode::getAverageLatencyMs() const {
    return total_processed_ > 0 ? total_latency_ms_ / total_processed_ : 0.0f;
}

void GPUResizeNormalizeNode::setTensorRTPreprocessEnabled(bool enable) {
    tensorrt_preprocess_enabled_ = enable;
    LOG_INFO_FMT("[GPUResizeNormalize] Set TensorRT preprocess: %d", enable);
}

// 注册节点
REGISTER_NODE("gpu_resize_normalize", GPUResizeNormalizeNode)

} // namespace nodes
} // namespace ai_stream