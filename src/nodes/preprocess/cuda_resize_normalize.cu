// src/nodes/preprocess/cuda_resize_normalize.cu
// 预处理节点——使用 ImageAcceleratorFactory，支持多后端
#include "cuda_resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace ai_stream {
namespace nodes {

CudaResizeNormalizeNode::CudaResizeNormalizeNode()
    : core::QueuedNode<IGpuPreprocessNode>("CudaResizeNormalize"),
      device_id_(0),
      async_processing_(false),
      stream_(nullptr),
      input_gpu_ptr_(nullptr),
      output_gpu_ptr_(nullptr),
      input_buffer_size_(0),
      output_buffer_size_(0),
      mean_{0.0f, 0.0f, 0.0f},
      std_{1.0f, 1.0f, 1.0f},
      total_processed_(0),
      total_latency_ms_(0.0f) {

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[CudaResizeNormalize] Node created with GPU device {}", device_id_);
    } else {
        LOG_WARN_FMT("[CudaResizeNormalize] No GPU device found");
    }

    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

CudaResizeNormalizeNode::~CudaResizeNormalizeNode() {
    stop();

    if (input_gpu_ptr_) cudaFree(input_gpu_ptr_);
    if (output_gpu_ptr_) cudaFree(output_gpu_ptr_);
    if (stream_ && owns_stream_) cudaStreamDestroy(stream_);
    stream_ = nullptr;
    owns_stream_ = false;
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);

    LOG_INFO_FMT("[CudaResizeNormalize] Node destroyed");
}

void CudaResizeNormalizeNode::setMean(const std::vector<float>& mean) {
    mean_ = mean;
    LOG_INFO_FMT("[CudaResizeNormalize] Mean set to: {},{},{}", mean_[0], mean_[1], mean_[2]);
}

void CudaResizeNormalizeNode::setStd(const std::vector<float>& std) {
    std_ = std;
    LOG_INFO_FMT("[CudaResizeNormalize] Std set to: {},{},{}", std_[0], std_[1], std_[2]);
}

bool CudaResizeNormalizeNode::onStartup() {
    if (!stream_) {
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_));
        owns_stream_ = true;
    }

    // 通过 HAL 工厂创建图像加速器
    accelerator_ = hal::ImageAcceleratorFactory::instance().create(backend_type_);
    if (!accelerator_) {
        LOG_ERROR("[CudaResizeNormalize] Failed to create image accelerator");
        return false;
    }

    LOG_INFO_FMT("[CudaResizeNormalize] Node started (backend: {})", accelerator_->getName());
    return true;
}

void CudaResizeNormalizeNode::onShutdown() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }

    LOG_INFO_FMT("[CudaResizeNormalize] Node stopped");
}

void CudaResizeNormalizeNode::ensureGpuBuffers(int src_w, int src_h, int dst_w, int dst_h) {
    size_t needed_input  = static_cast<size_t>(src_w) * src_h * 3;
    size_t needed_output = static_cast<size_t>(dst_w) * dst_h * 3 * sizeof(float);

    if (input_buffer_size_ < needed_input) {
        if (input_gpu_ptr_) cudaFree(input_gpu_ptr_);
        CUDA_CHECK(cudaMalloc(&input_gpu_ptr_, needed_input));
        input_buffer_size_ = needed_input;
    }

    if (output_buffer_size_ < needed_output) {
        if (output_gpu_ptr_) cudaFree(output_gpu_ptr_);
        CUDA_CHECK(cudaMalloc(&output_gpu_ptr_, needed_output));
        output_buffer_size_ = needed_output;
    }
}

void CudaResizeNormalizeNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO_FMT("[CudaResizeNormalize] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 workerLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }

    if (packet->type != core::PacketType::DECODED_FRAME) {
        LOG_ERROR_FMT("[CudaResizeNormalize] Invalid packet type");
        broadcast(packet);
        return;
    }

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);

    if (mean_.size() < 3 || std_.size() < 3) {
        LOG_ERROR_FMT("[CudaResizeNormalize] Mean or std not initialized");
        return;
    }

    int src_width, src_height;
    int dst_width = target_width_;
    int dst_height = target_height_;

    if (frame->is_gpu && frame->d_ptr) {
        src_width = frame->d_width;
        src_height = frame->d_height;
    } else {
        if (!frame->mat || frame->mat->empty()) {
            LOG_WARN_FMT("[CudaResizeNormalize] Received empty CPU frame");
            return;
        }
        src_width = frame->mat->cols;
        src_height = frame->mat->rows;
        if (frame->mat->channels() != 3) {
            LOG_ERROR_FMT("[CudaResizeNormalize] Unsupported channels: {}", frame->mat->channels());
            return;
        }
    }

    ensureGpuBuffers(src_width, src_height, dst_width, dst_height);

    CUDA_CHECK(cudaEventRecord(start_event_, stream_));

    const unsigned char* src_gpu_ptr = nullptr;
    size_t src_pitch = 0;

    if (frame->is_gpu && frame->d_ptr) {
        LOG_DEBUG_FMT("[CudaResizeNormalize] hardware decode path");
        src_gpu_ptr = static_cast<const unsigned char*>(frame->d_ptr);
        src_pitch = frame->d_pitch;
    } else {
        LOG_DEBUG_FMT("[CudaResizeNormalize] software decode path");
        src_pitch = static_cast<size_t>(src_width) * 3;
        if (frame->mat->isContinuous()) {
            CUDA_CHECK(cudaMemcpyAsync(input_gpu_ptr_, frame->mat->data,
                                       src_height * src_pitch,
                                       cudaMemcpyHostToDevice, stream_));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                input_gpu_ptr_, src_pitch,
                frame->mat->data, frame->mat->step,
                src_width * 3, src_height,
                cudaMemcpyHostToDevice, stream_));
        }
        src_gpu_ptr = static_cast<const unsigned char*>(input_gpu_ptr_);
    }

    // 使用 HAL 加速器执行 resize+normalize
    hal::ResizeNormalizeParams params;
    params.src_width = src_width;
    params.src_height = src_height;
    params.src_pitch = src_pitch;
    params.dst_width = dst_width;
    params.dst_height = dst_height;
    params.keep_aspect_ratio = keep_aspect_ratio_;
    params.mean = mean_;
    params.std = std_;
    params.stream = stream_;

    hal::LetterboxResult letter;
    bool success = accelerator_->resizeNormalize(
        src_gpu_ptr, params, static_cast<float*>(output_gpu_ptr_),
        keep_aspect_ratio_ ? &letter : nullptr);

    if (!success) {
        LOG_ERROR("[CudaResizeNormalize] Accelerator resizeNormalize failed");
        return;
    }

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        LOG_ERROR_FMT("[CudaResizeNormalize] Kernel launch failed: {}", cudaGetErrorString(kernel_err));
        return;
    }

    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float latency_ms = 0.0f;
    cudaEventElapsedTime(&latency_ms, start_event_, stop_event_);
    total_latency_ms_ += static_cast<int64_t>(latency_ms);
    total_processed_++;

    LOG_DEBUG_FMT("[CudaResizeNormalize] Processed {}x{} -> {}x{}{}, latency={:.2f}ms",
                  src_width, src_height, dst_width, dst_height,
                  keep_aspect_ratio_ ? " (letterbox)" : "",
                  latency_ms);

    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->source_id = frame->source_id;
    new_packet->width = target_width_;
    new_packet->height = target_height_;
    new_packet->channels = 3;
    new_packet->frame_id = frame->frame_id;

    new_packet->is_gpu = true;
    new_packet->d_ptr = output_gpu_ptr_;
    new_packet->d_pitch = target_width_ * sizeof(float);
    new_packet->d_width = target_width_;
    new_packet->d_height = target_height_;

    new_packet->d_bgr_ptr = frame->d_bgr_ptr;
    new_packet->d_bgr_pitch = frame->d_bgr_pitch;
    new_packet->d_bgr_width = frame->d_bgr_width;
    new_packet->d_bgr_height = frame->d_bgr_height;

    // Letterbox 参数
    new_packet->letterbox_used = keep_aspect_ratio_;
    if (keep_aspect_ratio_) {
        new_packet->letter_scale = letter.scale;
        new_packet->letter_pad_x = letter.pad_x;
        new_packet->letter_pad_y = letter.pad_y;
    } else {
        new_packet->letter_scale = 1.0f;
        new_packet->letter_pad_x = 0;
        new_packet->letter_pad_y = 0;
    }

    new_packet->source_mat = frame->source_mat;
    new_packet->mat = frame->mat;

    broadcast(new_packet);
}

void CudaResizeNormalizeNode::setGpuDeviceId(int device_id) {
    if (device_id_ != device_id) {
        device_id_ = device_id;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[CudaResizeNormalize] Set GPU device ID: {}", device_id_);
    }
}

int CudaResizeNormalizeNode::getGpuDeviceId() const {
    return device_id_;
}

void CudaResizeNormalizeNode::setAsyncProcessing(bool async) {
    async_processing_ = async;
    LOG_INFO_FMT("[CudaResizeNormalize] Set async processing: {}", async);
}

void CudaResizeNormalizeNode::setCudaStream(void* stream) {
    if (stream_ && owns_stream_) {
        cudaStreamDestroy(stream_);
    }
    stream_ = static_cast<cudaStream_t>(stream);
    owns_stream_ = false;
    LOG_INFO_FMT("[CudaResizeNormalize] Set external CUDA stream");
}

float CudaResizeNormalizeNode::getAverageLatencyMs() const {
    return total_processed_ > 0 ? static_cast<float>(total_latency_ms_) / total_processed_.load() : 0.0f;
}

void CudaResizeNormalizeNode::setTensorRTPreprocessEnabled(bool enable) {
    tensorrt_preprocess_enabled_ = enable;
    LOG_INFO_FMT("[CudaResizeNormalize] Set TensorRT preprocess: {}", enable);
}

REGISTER_NODE("cuda_resize_normalize", CudaResizeNormalizeNode)

} // namespace nodes
} // namespace ai_stream
