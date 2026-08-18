// src/nodes/preprocess/cuda_resize_normalize.cu
#include "cuda_resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace ai_stream {
namespace nodes {

namespace {

// ============================================================
// Mode 1: Direct resize (stretch to fill target)
// Used when keep_aspect_ratio_ == false
// ============================================================
__global__ void resizeNormalizeKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x0p1 = x0 + 1;
    int y0p1 = y0 + 1;
    int x1_i = (x0p1 < src_w) ? x0p1 : (src_w - 1);
    int y1_i = (y0p1 < src_h) ? y0p1 : (src_h - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    const unsigned char* row0 = src + y0 * src_pitch;
    const unsigned char* row1 = src + y1_i * src_pitch;

    float b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1_i * 3 + 0]
            + w10 * row1[x0 * 3 + 0] + w11 * row1[x1_i * 3 + 0];
    float g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1_i * 3 + 1]
            + w10 * row1[x0 * 3 + 1] + w11 * row1[x1_i * 3 + 1];
    float r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1_i * 3 + 2]
            + w10 * row1[x0 * 3 + 2] + w11 * row1[x1_i * 3 + 2];

    int hw = dst_w * dst_h;
    int idx = dst_y * dst_w + dst_x;
    dst[0 * hw + idx] = (r / 255.0f - mean_r) / std_r;
    dst[1 * hw + idx] = (g / 255.0f - mean_g) / std_g;
    dst[2 * hw + idx] = (b / 255.0f - mean_b) / std_b;
}

// ============================================================
// Mode 2: Letterbox resize (keep aspect ratio, pad with gray)
// Used when keep_aspect_ratio_ == true
// ============================================================
__global__ void letterboxResizeNormalizeKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    int letter_w, int letter_h,
    int pad_x, int pad_y,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    int hw = dst_w * dst_h;
    int idx = dst_y * dst_w + dst_x;

    // Check if this pixel is within the letterboxed image region
    bool in_letter_region = (dst_x >= pad_x && dst_x < (pad_x + letter_w)) &&
                            (dst_y >= pad_y && dst_y < (pad_y + letter_h));

    if (!in_letter_region) {
        // Pad region: fill with gray (114/255.0)
        float gray_val = (114.0f / 255.0f);
        dst[0 * hw + idx] = (gray_val - mean_r) / std_r;
        dst[1 * hw + idx] = (gray_val - mean_g) / std_g;
        dst[2 * hw + idx] = (gray_val - mean_b) / std_b;
        return;
    }

    // Map destination pixel to source coordinates (bilinear interpolation)
    float src_x = (dst_x - pad_x + 0.5f) * static_cast<float>(src_w) / letter_w - 0.5f;
    float src_y = (dst_y - pad_y + 0.5f) * static_cast<float>(src_h) / letter_h - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x0p1 = x0 + 1;
    int y0p1 = y0 + 1;
    int x1_i = (x0p1 < src_w) ? x0p1 : (src_w - 1);
    int y1_i = (y0p1 < src_h) ? y0p1 : (src_h - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    const unsigned char* row0 = src + y0 * src_pitch;
    const unsigned char* row1 = src + y1_i * src_pitch;

    float b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1_i * 3 + 0]
            + w10 * row1[x0 * 3 + 0] + w11 * row1[x1_i * 3 + 0];
    float g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1_i * 3 + 1]
            + w10 * row1[x0 * 3 + 1] + w11 * row1[x1_i * 3 + 1];
    float r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1_i * 3 + 2]
            + w10 * row1[x0 * 3 + 2] + w11 * row1[x1_i * 3 + 2];

    dst[0 * hw + idx] = (r / 255.0f - mean_r) / std_r;
    dst[1 * hw + idx] = (g / 255.0f - mean_g) / std_g;
    dst[2 * hw + idx] = (b / 255.0f - mean_b) / std_b;
}

} // anonymous namespace

CudaResizeNormalizeNode::CudaResizeNormalizeNode()
    : core::QueuedNode<IGpuPreprocessNode>("CudaResizeNormalize"),
      device_id_(0),
      async_processing_(false),
      stream_(nullptr),
      input_gpu_ptr_(nullptr),
      output_gpu_ptr_(nullptr),
      input_buffer_size_(0),
      output_buffer_size_(0),
      mean_{0.0f, 0.0f, 0.0f},   // Normalization for YOLO pre-trained models
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
    if (stream_) cudaStreamDestroy(stream_);
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
    }

    LOG_INFO_FMT("[CudaResizeNormalize] Node started");
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
        stop();
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
    // Output always targets the configured size
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

    // Letterbox parameters (only meaningful if keep_aspect_ratio_ is true)
    int letter_w = dst_width;   // actual letterboxed image width (within padding)
    int letter_h = dst_height;  // actual letterboxed image height (within padding)
    int pad_x = 0;              // horizontal padding offset
    int pad_y = 0;              // vertical padding offset
    float scale = 1.0f;         // letterbox scale factor

    if (keep_aspect_ratio_) {
        scale = std::min(
            static_cast<float>(target_width_) / src_width,
            static_cast<float>(target_height_) / src_height);

        letter_w = std::max(1, static_cast<int>(std::round(src_width * scale)));
        letter_h = std::max(1, static_cast<int>(std::round(src_height * scale)));

        pad_x = (target_width_ - letter_w) / 2;
        pad_y = (target_height_ - letter_h) / 2;

        LOG_DEBUG_FMT("[CudaResizeNormalize] Letterbox: src={}x{}, letter={}x{}, pad=({},{}), scale={:.4f}",
                       src_width, src_height, letter_w, letter_h, pad_x, pad_y, scale);
    }

    ensureGpuBuffers(src_width, src_height, dst_width, dst_height);

    CUDA_CHECK(cudaEventRecord(start_event_, stream_));

    const unsigned char* src_gpu_ptr = nullptr;
    size_t src_pitch = 0;

    if (frame->is_gpu && frame->d_ptr) {
        LOG_INFO_FMT("[HWDecode] hardware decode path");
        src_gpu_ptr = static_cast<const unsigned char*>(frame->d_ptr);
        src_pitch = frame->d_pitch;
    } else {
        LOG_INFO_FMT("[HWDecode] software decode path");
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

    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + 15) / 16, (dst_height + 15) / 16);

    if (keep_aspect_ratio_) {
        // Letterbox path: keep aspect ratio, pad with gray (114)
        letterboxResizeNormalizeKernel<<<grid_size, block_size, 0, stream_>>>(
            src_gpu_ptr,
            src_width, src_height, src_pitch,
            static_cast<float*>(output_gpu_ptr_),
            dst_width, dst_height,
            letter_w, letter_h,
            pad_x, pad_y,
            mean_[0], mean_[1], mean_[2],
            std_[0], std_[1], std_[2]
        );
        LOG_DEBUG_FMT("[CudaResizeNormalize] Mode=letterbox, grid=({},{}), block=({},{})",
                       grid_size.x, grid_size.y, block_size.x, block_size.y);
    } else {
        // Direct path: stretch to fill
        resizeNormalizeKernel<<<grid_size, block_size, 0, stream_>>>(
            src_gpu_ptr,
            src_width, src_height, src_pitch,
            static_cast<float*>(output_gpu_ptr_),
            dst_width, dst_height,
            mean_[0], mean_[1], mean_[2],
            std_[0], std_[1], std_[2]
        );
        LOG_DEBUG_FMT("[CudaResizeNormalize] Mode=resize, grid=({},{})",
                       grid_size.x, grid_size.y);
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

    LOG_INFO_FMT("[CudaResizeNormalize] Processed {}x{} -> {}x{}{}, latency={:.2f}ms",
                  src_width, src_height, dst_width, dst_height,
                  keep_aspect_ratio_ ? " (letterbox)" : "",
                  latency_ms);

    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->source_id = frame->source_id;
    // Always report the configured target size to downstream nodes
    new_packet->width = target_width_;
    new_packet->height = target_height_;
    new_packet->channels = 3;
    new_packet->frame_id = frame->frame_id;

    new_packet->is_gpu = true;
    new_packet->d_ptr = output_gpu_ptr_;// NCHW
    // NCHW format: d_pitch represents bytes per row of a single channel
    new_packet->d_pitch = target_width_ * sizeof(float);
    new_packet->d_width = target_width_;
    new_packet->d_height = target_height_;

    new_packet->d_bgr_ptr = frame->d_bgr_ptr;
    new_packet->d_bgr_pitch = frame->d_bgr_pitch;
    new_packet->d_bgr_width = frame->d_bgr_width;
    new_packet->d_bgr_height = frame->d_bgr_height;

    // Letterbox 参数：让下游节点知道如何反变换坐标
    new_packet->letterbox_used = keep_aspect_ratio_;
    if (keep_aspect_ratio_) {
        new_packet->letter_scale = scale;  // scale 已在上方计算
        new_packet->letter_pad_x = pad_x;
        new_packet->letter_pad_y = pad_y;
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

void CudaResizeNormalizeNode::setCudaStream(cudaStream_t stream) {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    stream_ = stream;
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
