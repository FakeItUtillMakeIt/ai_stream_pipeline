// src/nodes/preprocess/cuda_resize_normalize.cu
#include "cuda_resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace ai_stream {
namespace nodes {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG_ERROR_FMT("[GPUResizeNormalize] CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

#define CUDA_CHECK_BOOL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG_ERROR_FMT("[GPUResizeNormalize] CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

namespace {

// 融合 Kernel：双线性插值 Resize + BGR→RGB + Normalize
__global__ void resizeNormalizeKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    int dst_w, int dst_h, size_t dst_pitch,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) {
        return;
    }

    // 输出像素中心对齐映射到输入坐标
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    // Clamp 到有效范围，避免越界
    src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    // 源图像行指针（考虑 pitch，兼容非连续 device 布局）
    const unsigned char* row0 = src + y0 * src_pitch;
    const unsigned char* row1 = src + y1 * src_pitch;

    // 双线性采样 BGR
    float b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1 * 3 + 0]
            + w10 * row1[x0 * 3 + 0] + w11 * row1[x1 * 3 + 0];
    float g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1 * 3 + 1]
            + w10 * row1[x0 * 3 + 1] + w11 * row1[x1 * 3 + 1];
    float r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1 * 3 + 2]
            + w10 * row1[x0 * 3 + 2] + w11 * row1[x1 * 3 + 2];

    // 输出地址（考虑 dst_pitch）
    float* out_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst) + dst_y * dst_pitch);
    float* out_pix = out_row + dst_x * 3;

    // BGR→RGB + Normalize
    out_pix[0] = (r / 255.0f - mean_r) / std_r;
    out_pix[1] = (g / 255.0f - mean_g) / std_g;
    out_pix[2] = (b / 255.0f - mean_b) / std_b;
}

} // anonymous namespace

CudaResizeNormalizeNode::CudaResizeNormalizeNode()
    : IGpuPreprocessNode("GPUResizeNormalize"),
      device_id_(0),
      async_processing_(false),
      stream_(nullptr),
      input_gpu_ptr_(nullptr),
      output_gpu_ptr_(nullptr),
      input_buffer_size_(0),
      output_buffer_size_(0),
      mean_{0.485f, 0.456f, 0.406f},
      std_{0.229f, 0.224f, 0.225f},
      total_processed_(0),
      total_latency_ms_(0.0f) {

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GPUResizeNormalize] Node created with GPU device {}", device_id_);
    } else {
        LOG_WARN_FMT("[GPUResizeNormalize] No GPU device found, will fallback to CPU");
    }

    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

CudaResizeNormalizeNode::~CudaResizeNormalizeNode() {
    stop();

    if (input_gpu_ptr_) {
        cudaFree(input_gpu_ptr_);
        input_gpu_ptr_ = nullptr;
    }
    if (output_gpu_ptr_) {
        cudaFree(output_gpu_ptr_);
        output_gpu_ptr_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);

    LOG_INFO_FMT("[GPUResizeNormalize] Node destroyed");
}

void CudaResizeNormalizeNode::setMean(const std::vector<float>& mean)
{
    mean_ = mean;
    LOG_INFO_FMT("[GPUResizeNormalize] Mean set to: {},{},{}",mean_[0],mean_[1],mean_[2]);
}

void CudaResizeNormalizeNode::setStd(const std::vector<float>& std)
{
    std_ = std;
    LOG_INFO_FMT("[GPUResizeNormalize] Std set to: {},{},{}", std[0], std[1], std[2]);
}

bool CudaResizeNormalizeNode::start() {
    if (!IGpuPreprocessNode::start()) {
        return false;
    }

    if (!stream_) {
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_));
    }

    running_ = true;
    LOG_INFO_FMT("[GPUResizeNormalize] Node started");
    return true;
}

void CudaResizeNormalizeNode::stop() {
    if (!running_) return;

    running_ = false;

    if (stream_) {
        cudaStreamSynchronize(stream_);
    }

    LOG_INFO_FMT("[GPUResizeNormalize] Node stopped");
}

void CudaResizeNormalizeNode::ensureGpuBuffers(int src_w, int src_h, int dst_w, int dst_h) {
    size_t needed_input  = static_cast<size_t>(src_w) * src_h * 3;
    size_t needed_output = static_cast<size_t>(dst_w) * dst_h * 3 * sizeof(float);

    if (input_buffer_size_ < needed_input) {
        if (input_gpu_ptr_) cudaFree(input_gpu_ptr_);
        CUDA_CHECK(cudaMalloc(&input_gpu_ptr_, needed_input));
        input_buffer_size_ = needed_input;
        LOG_INFO_FMT("[GPUResizeNormalize] Reallocated input GPU buffer: {} bytes", needed_input);
    }

    if (output_buffer_size_ < needed_output) {
        if (output_gpu_ptr_) cudaFree(output_gpu_ptr_);
        CUDA_CHECK(cudaMalloc(&output_gpu_ptr_, needed_output));
        output_buffer_size_ = needed_output;
        LOG_INFO_FMT("[GPUResizeNormalize] Reallocated output GPU buffer: {} bytes", needed_output);
    }
}

void CudaResizeNormalizeNode::pushData(std::shared_ptr<core::BasePacket> packet) {
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

    int src_width = frame->mat->cols;
    int src_height = frame->mat->rows;
    int src_channels = frame->mat->channels();

    if (src_channels != 3) {
        LOG_ERROR_FMT("[GPUResizeNormalize] Unsupported channels: %d, expected 3 (BGR)", src_channels);
        return;
    }

    // 计算目标尺寸（考虑宽高比）
    int dst_width = target_width_;
    int dst_height = target_height_;

    if (keep_aspect_ratio_) {
        float src_aspect = static_cast<float>(src_width) / src_height;
        float dst_aspect = static_cast<float>(target_width_) / target_height_;

        if (src_aspect > dst_aspect) {
            dst_width = target_width_;
            dst_height = static_cast<int>(std::round(static_cast<float>(dst_width) / src_aspect));
        } else {
            dst_height = target_height_;
            dst_width = static_cast<int>(std::round(static_cast<float>(dst_height) * src_aspect));
        }
    }

    // 兜底：禁止 0 尺寸
    dst_width = std::max(1, dst_width);
    dst_height = std::max(1, dst_height);

    // 按需分配 GPU 缓冲区（输入 uint8 / 输出 float 独立管理）
    ensureGpuBuffers(src_width, src_height, dst_width, dst_height);

    // 记录处理开始时间
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));

    // 1. 上传数据到 GPU
    //    使用 cudaMemcpy2DAsync 兼容 OpenCV 非连续 Mat / ROI（step 可能大于 width*3）
    size_t src_host_step = frame->mat->step;                    // 主机端行步长
    size_t src_dev_pitch = static_cast<size_t>(src_width) * 3;  // 设备端紧密排列

    if (!frame->mat->data) {
        LOG_ERROR_FMT("[GPUResizeNormalize] Input frame data is null");
        return;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(
        input_gpu_ptr_, src_dev_pitch,
        frame->mat->data, src_host_step,
        src_width * 3, src_height,
        cudaMemcpyHostToDevice, stream_));

    // 2. 启动融合 CUDA Kernel
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);

    size_t dst_pitch = static_cast<size_t>(dst_width) * 3 * sizeof(float);

    // mean_/std_ 顺序：mean_[0]=B, mean_[1]=G, mean_[2]=R（与原代码注释一致）
    if (mean_.size() < 3 || std_.size() < 3) {
        LOG_ERROR_FMT("[GPUResizeNormalize] Mean or std not properly initialized. mean_size={}, std_size={}",
                      mean_.size(), std_.size());
        return;
    }

    resizeNormalizeKernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<const unsigned char*>(input_gpu_ptr_),
        src_width, src_height, src_dev_pitch,
        static_cast<float*>(output_gpu_ptr_),
        dst_width, dst_height, dst_pitch,
        mean_[2], mean_[1], mean_[0],   // R, G, B
        std_[2], std_[1], std_[0]       // R, G, B
    );

    // 立即检查 kernel 启动错误
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        LOG_ERROR_FMT("[GPUResizeNormalize] Kernel launch failed: %s", cudaGetErrorString(kernel_err));
        return;
    }

    // 3. 拷贝回主机内存
    size_t dst_bytes = static_cast<size_t>(dst_width) * dst_height * 3 * sizeof(float);
    auto processed_mat = std::make_shared<cv::Mat>(dst_height, dst_width, CV_32FC3);

    if (!processed_mat || processed_mat->empty() || !processed_mat->data) {
        LOG_ERROR_FMT("[GPUResizeNormalize] Failed to allocate output Mat");
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(processed_mat->data, output_gpu_ptr_, dst_bytes,
                               cudaMemcpyDeviceToHost, stream_));

    // 记录处理结束时间并同步
    cudaEventRecord(stop_event_, stream_);
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    // 计算处理延迟
    float latency_ms = 0.0f;
    cudaEventElapsedTime(&latency_ms, start_event_, stop_event_);
    total_latency_ms_ += static_cast<int64_t>(latency_ms);
    total_processed_++;

    LOG_INFO_FMT("[GpuResizeNormalize] Resized frame to {}x{}", target_width_, target_height_);

    // 4. 构造输出包
    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->source_id = frame->source_id;
    new_packet->mat = processed_mat;
    new_packet->source_mat = frame->mat;
    new_packet->width = dst_width;
    new_packet->height = dst_height;
    new_packet->channels = 3;
    new_packet->frame_id = frame->frame_id;

    broadcast(new_packet);
}

void CudaResizeNormalizeNode::setGpuDeviceId(int device_id) {
    if (device_id_ != device_id) {
        device_id_ = device_id;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GPUResizeNormalize] Set GPU device ID: %d", device_id_);
    }
}

int CudaResizeNormalizeNode::getGpuDeviceId() const {
    return device_id_;
}

void CudaResizeNormalizeNode::setAsyncProcessing(bool async) {
    async_processing_ = async;
    LOG_INFO_FMT("[GPUResizeNormalize] Set async processing: %d", async);
}

void CudaResizeNormalizeNode::setCudaStream(cudaStream_t stream) {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    stream_ = stream;
    LOG_INFO_FMT("[GPUResizeNormalize] Set external CUDA stream");
}

float CudaResizeNormalizeNode::getAverageLatencyMs() const {
    return total_processed_ > 0 ? static_cast<float>(total_latency_ms_) / total_processed_.load() : 0.0f;
}

void CudaResizeNormalizeNode::setTensorRTPreprocessEnabled(bool enable) {
    tensorrt_preprocess_enabled_ = enable;
    LOG_INFO_FMT("[GPUResizeNormalize] Set TensorRT preprocess: %d", enable);
}

// 注册节点
REGISTER_NODE("cuda_resize_normalize", CudaResizeNormalizeNode)

} // namespace nodes
} // namespace ai_stream