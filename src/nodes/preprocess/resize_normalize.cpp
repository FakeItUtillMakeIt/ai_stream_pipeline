// src/nodes/preprocess/resize_normalize.cpp
#include "resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ai_stream {
namespace nodes {

namespace {
constexpr int kChannels = 3;

static size_t bufferSizeFor(int width, int height) {
    return static_cast<size_t>(width) * static_cast<size_t>(height) * kChannels;
}
} // namespace

ResizeNormalizeNode::ResizeNormalizeNode()
    : core::QueuedNode<IPreprocessNode>("ResizeNormalize") {
    LOG_INFO_FMT("[ResizeNormalize] initialized");
}

ResizeNormalizeNode::~ResizeNormalizeNode() {
    stop();
    // STREAM_END may have already cleared running_ before destruction, so
    // ensure resources are released even when QueuedNode::stop() is a no-op.
    onShutdown();
    LOG_INFO_FMT("[ResizeNormalize] deinitialized");
}

bool ResizeNormalizeNode::onStartup() {
    LOG_INFO_FMT("[ResizeNormalize] started");
    return true;
}

void ResizeNormalizeNode::onShutdown() {
#ifdef WITH_CUDA
    if (cuda_stream_ && owns_cuda_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
    }
    cuda_stream_ = nullptr;
    owns_cuda_stream_ = false;
    if (input_gpu_ptr_) {
        cudaFree(input_gpu_ptr_);
        input_gpu_ptr_ = nullptr;
    }
    if (output_gpu_ptr_) {
        cudaFree(output_gpu_ptr_);
        output_gpu_ptr_ = nullptr;
    }
    input_buffer_size_ = 0;
    output_buffer_size_ = 0;
    gpu_accelerator_.reset();
    gpu_backend_selected_ = hal::ImageAcceleratorBackend::AUTO;
    gpu_backend_checked_ = false;
#endif
    cpu_accelerator_.reset();
    cpu_backend_selected_ = hal::ImageAcceleratorBackend::AUTO;
    host_nchw_buffer_.clear();
    host_nchw_buffer_.shrink_to_fit();
    LOG_INFO_FMT("[ResizeNormalize] stopped");
}

void ResizeNormalizeNode::setTargetSize(int width, int height) {
    target_width_ = width;
    target_height_ = height;
}

std::pair<int, int> ResizeNormalizeNode::getTargetSize() const {
    return {target_width_, target_height_};
}

void ResizeNormalizeNode::setMean(const std::vector<float>& mean) {
    mean_ = mean;
    if (mean_.size() >= 3) {
        LOG_INFO_FMT("[ResizeNormalize] mean set to: {},{},{}", mean_[0], mean_[1], mean_[2]);
    }
}

void ResizeNormalizeNode::setStd(const std::vector<float>& std) {
    std_ = std;
    if (std_.size() >= 3) {
        LOG_INFO_FMT("[ResizeNormalize] std set to: {},{},{}", std_[0], std_[1], std_[2]);
    }
}

void ResizeNormalizeNode::setInterpolationMethod(const std::string& method) {
    interpolation_method_ = method;
}

void ResizeNormalizeNode::setKeepAspectRatio(bool keep_aspect_ratio) {
    keep_aspect_ratio_ = keep_aspect_ratio;
}

void ResizeNormalizeNode::setOutputDataType(const std::string& dtype) {
    output_dtype_ = dtype;
}

void ResizeNormalizeNode::setImageAcceleratorBackend(hal::ImageAcceleratorBackend backend) {
    backend_type_ = backend;
    cpu_accelerator_.reset();
    cpu_backend_selected_ = hal::ImageAcceleratorBackend::AUTO;
#ifdef WITH_CUDA
    gpu_accelerator_.reset();
    gpu_backend_selected_ = hal::ImageAcceleratorBackend::AUTO;
    gpu_backend_checked_ = false;
#endif
}

hal::IImageAccelerator* ResizeNormalizeNode::getCpuAccelerator() {
    auto& factory = hal::ImageAcceleratorFactory::instance();

    // Keep the selected backend for the node lifetime. In particular, do not
    // probe unavailable RGA/DVPP backends again for every frame after CPU
    // fallback has already been selected.
    if (cpu_accelerator_) {
        return cpu_accelerator_.get();
    }

    std::vector<hal::ImageAcceleratorBackend> candidates;
    switch (backend_type_) {
    case hal::ImageAcceleratorBackend::RGA:
        candidates = {hal::ImageAcceleratorBackend::RGA, hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::DVPP:
        candidates = {hal::ImageAcceleratorBackend::DVPP, hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::CPU:
        candidates = {hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::NPP:
        LOG_WARN("[ResizeNormalize] NPP requested for CPU path; falling back to CPU");
        candidates = {hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::AUTO:
    default:
        candidates = {hal::ImageAcceleratorBackend::RGA,
                      hal::ImageAcceleratorBackend::DVPP,
                      hal::ImageAcceleratorBackend::CPU};
        break;
    }

    for (auto backend : candidates) {
        auto accel = factory.create(backend);
        if (accel) {
            cpu_backend_selected_ = backend;
            cpu_accelerator_ = std::move(accel);
            LOG_INFO_FMT("[ResizeNormalize] CPU backend: {}", cpu_accelerator_->getName());
            return cpu_accelerator_.get();
        }
    }

    LOG_ERROR("[ResizeNormalize] No CPU image accelerator available");
    return nullptr;
}

#ifdef WITH_CUDA
hal::IImageAccelerator* ResizeNormalizeNode::getGpuAccelerator() {
    auto& factory = hal::ImageAcceleratorFactory::instance();
    if (gpu_backend_checked_) {
        return gpu_accelerator_.get();
    }
    gpu_backend_checked_ = true;
    if (gpu_accelerator_ && gpu_backend_selected_ == hal::ImageAcceleratorBackend::NPP) {
        return gpu_accelerator_.get();
    }

    auto accel = factory.create(hal::ImageAcceleratorBackend::NPP);
    if (!accel) {
        LOG_ERROR("[ResizeNormalize] NPP backend not available for GPU path");
        return nullptr;
    }

    gpu_backend_selected_ = hal::ImageAcceleratorBackend::NPP;
    gpu_accelerator_ = std::move(accel);
    LOG_INFO_FMT("[ResizeNormalize] GPU backend: {}", gpu_accelerator_->getName());
    return gpu_accelerator_.get();
}

bool ResizeNormalizeNode::ensureGpuBuffers(int src_w, int src_h, int dst_w, int dst_h) {
    size_t needed_input = static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * kChannels;
    size_t needed_output = static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * kChannels * sizeof(float);

    if (input_buffer_size_ < needed_input) {
        if (input_gpu_ptr_) {
            cudaFree(input_gpu_ptr_);
            input_gpu_ptr_ = nullptr;
        }
        if (cudaMalloc(&input_gpu_ptr_, needed_input) != cudaSuccess) {
            LOG_ERROR_FMT("[ResizeNormalize] cudaMalloc input failed: {}", needed_input);
            return false;
        }
        input_buffer_size_ = needed_input;
    }

    if (output_buffer_size_ < needed_output) {
        if (output_gpu_ptr_) {
            cudaFree(output_gpu_ptr_);
            output_gpu_ptr_ = nullptr;
        }
        if (cudaMalloc(&output_gpu_ptr_, needed_output) != cudaSuccess) {
            LOG_ERROR_FMT("[ResizeNormalize] cudaMalloc output failed: {}", needed_output);
            return false;
        }
        output_buffer_size_ = needed_output;
    }

    return true;
}
#endif

std::shared_ptr<cv::Mat> ResizeNormalizeNode::convertNchwToMat(const float* nchw, int width, int height) const {
    if (!nchw || width <= 0 || height <= 0) {
        return nullptr;
    }

    auto mat = std::make_shared<cv::Mat>(height, width, CV_32FC3);
    const int plane = width * height;

    for (int y = 0; y < height; ++y) {
        cv::Vec3f* row = mat->ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            row[x][0] = nchw[2 * plane + idx];
            row[x][1] = nchw[1 * plane + idx];
            row[x][2] = nchw[0 * plane + idx];
        }
    }

    return mat;
}

void ResizeNormalizeNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO_FMT("[ResizeNormalize] Received stream end");
        broadcast(packet);
        return;
    }

    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    if (packet->type != core::PacketType::DECODED_FRAME) {
        broadcast(packet);
        return;
    }

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
#ifdef WITH_CUDA
    // NPP accepts device buffers only. CPU-decoded frames are copied to the
    // node-owned GPU input buffer so AUTO preserves the accelerated CUDA path.
    const bool npp_requested = backend_type_ == hal::ImageAcceleratorBackend::AUTO ||
                               backend_type_ == hal::ImageAcceleratorBackend::NPP;
    // getGpuAccelerator() caches the backend instance, so availability is
    // checked once per node instead of constructing NPP on every frame.
    const bool use_gpu_path = (frame->is_gpu && frame->d_ptr) ||
                              (npp_requested && getGpuAccelerator() != nullptr);
    if (use_gpu_path) {
        processGpuFrame(frame);
    } else {
        processCpuFrame(frame);
    }
#else
    if (frame->is_gpu && frame->d_ptr) {
        LOG_ERROR("[ResizeNormalize] GPU frame received but CUDA is disabled");
    } else {
        processCpuFrame(frame);
    }
#endif
}

bool ResizeNormalizeNode::processCpuFrame(const std::shared_ptr<core::VideoFramePacket>& frame) {
    if (!frame || !frame->mat || frame->mat->empty()) {
        LOG_WARN("[ResizeNormalize] Received empty CPU frame");
        return false;
    }

    if (frame->mat->channels() != 3) {
        LOG_ERROR_FMT("[ResizeNormalize] Unsupported CPU frame channels: {}", frame->mat->channels());
        return false;
    }

    auto* accelerator = getCpuAccelerator();
    if (!accelerator) {
        return false;
    }

    const int src_width = frame->mat->cols;
    const int src_height = frame->mat->rows;
    const int dst_width = target_width_;
    const int dst_height = target_height_;

    host_nchw_buffer_.resize(bufferSizeFor(dst_width, dst_height));

    hal::ResizeNormalizeParams params;
    params.src_width = src_width;
    params.src_height = src_height;
    params.src_pitch = frame->mat->step;
    params.dst_width = dst_width;
    params.dst_height = dst_height;
    params.keep_aspect_ratio = keep_aspect_ratio_;
    params.mean = mean_;
    params.std = std_;

    hal::LetterboxResult letter;
    bool ok = accelerator->resizeNormalize(
        frame->mat->data,
        params,
        host_nchw_buffer_.data(),
        keep_aspect_ratio_ ? &letter : nullptr);

    if (!ok) {
        LOG_ERROR("[ResizeNormalize] CPU resizeNormalize failed");
        return false;
    }

    auto processed_mat = convertNchwToMat(host_nchw_buffer_.data(), dst_width, dst_height);
    if (!processed_mat) {
        LOG_ERROR("[ResizeNormalize] Failed to convert NCHW output to Mat");
        return false;
    }

    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->source_id = frame->source_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->mat = processed_mat;
    new_packet->source_mat = frame->source_mat ? frame->source_mat : frame->mat;
    new_packet->width = dst_width;
    new_packet->height = dst_height;
    new_packet->channels = 3;
    new_packet->frame_id = frame->frame_id;
    new_packet->is_gpu = false;
    new_packet->d_ptr = nullptr;
    new_packet->d_width = 0;
    new_packet->d_height = 0;
    new_packet->d_pitch = 0;
    new_packet->d_bgr_ptr = frame->d_bgr_ptr;
    new_packet->d_bgr_pitch = frame->d_bgr_pitch;
    new_packet->d_bgr_width = frame->d_bgr_width;
    new_packet->d_bgr_height = frame->d_bgr_height;

    if (keep_aspect_ratio_) {
        new_packet->letterbox_used = true;
        new_packet->letter_scale = letter.scale;
        new_packet->letter_pad_x = letter.pad_x;
        new_packet->letter_pad_y = letter.pad_y;
    } else {
        new_packet->letterbox_used = false;
        new_packet->letter_scale = 1.0f;
        new_packet->letter_pad_x = 0;
        new_packet->letter_pad_y = 0;
    }

    new_packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
    new_packet->cost_time_map = frame->cost_time_map;
    new_packet->cost_time_map.insert({name_, new_packet->cost_ms});

    LOG_INFO_FMT("[ResizeNormalize] Resized CPU frame to {}x{}", dst_width, dst_height);
    broadcast(new_packet);
    return true;
}

#ifdef WITH_CUDA
bool ResizeNormalizeNode::processGpuFrame(const std::shared_ptr<core::VideoFramePacket>& frame) {
    if (!frame) {
        return false;
    }

    auto* accelerator = getGpuAccelerator();
    if (!accelerator) {
        return false;
    }

    if (!cuda_stream_) {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            LOG_ERROR("[ResizeNormalize] cudaStreamCreate failed");
            return false;
        }
        cuda_stream_ = stream;
        owns_cuda_stream_ = true;
    }
    auto stream = static_cast<cudaStream_t>(cuda_stream_);

    const int dst_width = target_width_;
    const int dst_height = target_height_;

    int src_width = 0;
    int src_height = 0;
    const uint8_t* src_ptr = nullptr;
    size_t src_pitch = 0;

    if (frame->is_gpu && frame->d_ptr) {
        src_width = frame->d_width;
        src_height = frame->d_height;
        src_ptr = static_cast<const uint8_t*>(frame->d_ptr);
        src_pitch = frame->d_pitch > 0 ? static_cast<size_t>(frame->d_pitch) : static_cast<size_t>(src_width) * kChannels;
    } else if (frame->mat && !frame->mat->empty()) {
        src_width = frame->mat->cols;
        src_height = frame->mat->rows;
        src_pitch = static_cast<size_t>(frame->mat->step);
        if (!ensureGpuBuffers(src_width, src_height, dst_width, dst_height)) {
            return false;
        }

        size_t copy_bytes = static_cast<size_t>(src_height) * src_pitch;
        if (frame->mat->isContinuous()) {
            if (cudaMemcpyAsync(input_gpu_ptr_, frame->mat->data, copy_bytes,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                LOG_ERROR("[ResizeNormalize] cudaMemcpy host->device failed");
                return false;
            }
        } else {
            if (cudaMemcpy2DAsync(input_gpu_ptr_, src_width * kChannels,
                                  frame->mat->data, frame->mat->step,
                                  src_width * kChannels, src_height,
                                  cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                LOG_ERROR("[ResizeNormalize] cudaMemcpy2D host->device failed");
                return false;
            }
        }
        src_ptr = static_cast<const uint8_t*>(input_gpu_ptr_);
        src_pitch = static_cast<size_t>(src_width) * kChannels;
    } else {
        LOG_WARN("[ResizeNormalize] GPU path received empty frame");
        return false;
    }

    if (!ensureGpuBuffers(src_width, src_height, dst_width, dst_height)) {
        return false;
    }

    hal::ResizeNormalizeParams params;
    params.src_width = src_width;
    params.src_height = src_height;
    params.src_pitch = src_pitch;
    params.dst_width = dst_width;
    params.dst_height = dst_height;
    params.keep_aspect_ratio = keep_aspect_ratio_;
    params.mean = mean_;
    params.std = std_;
    params.stream = stream;

    hal::LetterboxResult letter;
    bool ok = accelerator->resizeNormalize(
        src_ptr,
        params,
        static_cast<float*>(output_gpu_ptr_),
        keep_aspect_ratio_ ? &letter : nullptr);

    if (!ok) {
        LOG_ERROR("[ResizeNormalize] GPU resizeNormalize failed");
        return false;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        LOG_ERROR("[ResizeNormalize] CUDA stream synchronization failed");
        return false;
    }

    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->source_id = frame->source_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->width = dst_width;
    new_packet->height = dst_height;
    new_packet->channels = 3;
    new_packet->frame_id = frame->frame_id;
    new_packet->is_gpu = true;
    new_packet->d_ptr = output_gpu_ptr_; // NCHW float buffer
    new_packet->d_pitch = dst_width * static_cast<int>(sizeof(float));
    new_packet->d_width = dst_width;
    new_packet->d_height = dst_height;
    new_packet->source_mat = frame->source_mat ? frame->source_mat : frame->mat;
    new_packet->mat = frame->mat;
    new_packet->d_bgr_ptr = frame->d_bgr_ptr;
    new_packet->d_bgr_pitch = frame->d_bgr_pitch;
    new_packet->d_bgr_width = frame->d_bgr_width;
    new_packet->d_bgr_height = frame->d_bgr_height;

    if (keep_aspect_ratio_) {
        new_packet->letterbox_used = true;
        new_packet->letter_scale = letter.scale;
        new_packet->letter_pad_x = letter.pad_x;
        new_packet->letter_pad_y = letter.pad_y;
    } else {
        new_packet->letterbox_used = false;
        new_packet->letter_scale = 1.0f;
        new_packet->letter_pad_x = 0;
        new_packet->letter_pad_y = 0;
    }

    new_packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
    new_packet->cost_time_map = frame->cost_time_map;
    new_packet->cost_time_map.insert({name_, new_packet->cost_ms});

    LOG_INFO_FMT("[ResizeNormalize] Resized GPU frame to {}x{}", dst_width, dst_height);
    broadcast(new_packet);
    return true;
}
#endif

REGISTER_NODE("resize_normalize", ResizeNormalizeNode)

} // namespace nodes
} // namespace ai_stream
