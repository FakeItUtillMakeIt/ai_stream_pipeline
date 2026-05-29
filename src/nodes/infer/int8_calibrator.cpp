// src/nodes/infer/int8_calibrator.cpp
#include "int8_calibrator.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace ai_stream {
namespace nodes {

Int8EntropyCalibrator::Int8EntropyCalibrator()
    : batch_size_(1),
      input_height_(640),
      input_width_(640),
      input_channels_(3),
      input_size_(0),
      current_batch_index_(0),
      d_input_buffer_(nullptr) {
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
}

bool Int8EntropyCalibrator::initialize(
    const std::vector<std::string>& calibration_images,
    int batch_size,
    int input_height,
    int input_width,
    const std::string& input_name,
    const std::string& cache_file)
{
    batch_size_ = batch_size;
    input_height_ = input_height;
    input_width_ = input_width;
    input_name_ = input_name;
    cache_file_ = cache_file;
    input_channels_ = 3;

    // 收集校准图片
    calibration_image_paths_.clear();
    for (const auto& img_path : calibration_images) {
        if (std::filesystem::is_directory(img_path)) {
            // 目录：扫描所有图片
            for (const auto& entry : std::filesystem::directory_iterator(img_path)) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    calibration_image_paths_.push_back(entry.path().string());
                }
            }
        } else if (std::filesystem::exists(img_path)) {
            calibration_image_paths_.push_back(img_path);
        }
    }

    if (calibration_image_paths_.empty()) {
        LOG_ERROR_FMT("[Int8Calibrator] No calibration images found");
        return false;
    }

    // 至少需要 batch_size 张图片
    if (static_cast<int>(calibration_image_paths_.size()) < batch_size_) {
        LOG_WARN_FMT("[Int8Calibrator] Only {} images, need at least {}",
                     calibration_image_paths_.size(), batch_size_);
    }

    // 打乱顺序
    std::random_shuffle(calibration_image_paths_.begin(), calibration_image_paths_.end());

    input_size_ = static_cast<size_t>(batch_size_) * input_channels_
                  * input_height_ * input_width_ * sizeof(float);

    // 分配 GPU 缓冲区
    if (d_input_buffer_) cudaFree(d_input_buffer_);
    CUDA_CHECK_BOOL(cudaMalloc(&d_input_buffer_, input_size_));

    h_input_buffer_.resize(static_cast<size_t>(batch_size_) * input_channels_
                           * input_height_ * input_width_);

    current_batch_index_ = 0;

    LOG_INFO_FMT("[Int8Calibrator] Initialized: {} images, batch={}, input={}x{}, cache={}",
                 calibration_image_paths_.size(), batch_size_,
                 input_width_, input_height_, cache_file_);
    return true;
}

int Int8EntropyCalibrator::getBatchSize() const noexcept {
    return batch_size_;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    int total_batches = static_cast<int>(calibration_image_paths_.size()) / batch_size_;
    int current = current_batch_index_.load();
    if (current >= total_batches) {
        return false; // 所有 batch 已用完
    }

    int start_idx = current * batch_size_;
    std::vector<cv::Mat> batch_images(batch_size_);

    for (int i = 0; i < batch_size_; ++i) {
        const std::string& path = calibration_image_paths_[start_idx + i];
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            LOG_WARN_FMT("[Int8Calibrator] Failed to read: {}", path);
            // 用黑色图替代
            img = cv::Mat(input_height_, input_width_, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        batch_images[i] = img;
    }

    // 预处理：resize + HWC→NCHW + 归一化
    preprocessBatch(batch_images, h_input_buffer_.data());

    // 拷贝到 GPU
    CUDA_CHECK_BOOL(cudaMemcpy(d_input_buffer_, h_input_buffer_.data(),
                          input_size_, cudaMemcpyHostToDevice));

    // 设置 binding
    for (int i = 0; i < nbBindings; ++i) {
        if (std::string(names[i]) == input_name_) {
            bindings[i] = d_input_buffer_;
        }
    }

    current_batch_index_++;
    if (current_batch_index_ % 10 == 0) {
        LOG_INFO_FMT("[Int8Calibrator] Progress: {}/{} batches", current_batch_index_.load(), total_batches);
    }
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    calibration_cache_.clear();

    if (cache_file_.empty() || !std::filesystem::exists(cache_file_)) {
        LOG_INFO_FMT("[Int8Calibrator] No cache file found, will generate new calibration");
        length = 0;
        return nullptr;
    }

    std::ifstream file(cache_file_, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR_FMT("[Int8Calibrator] Cannot open cache: {}", cache_file_);
        length = 0;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t sz = file.tellg();
    file.seekg(0, std::ios::beg);

    calibration_cache_.resize(sz);
    file.read(calibration_cache_.data(), sz);
    length = sz;

    LOG_INFO_FMT("[Int8Calibrator] Loaded calibration cache: {} bytes from {}", sz, cache_file_);
    return calibration_cache_.data();
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* ptr, size_t length) noexcept {
    if (cache_file_.empty()) return;

    // 确保父目录存在
    auto parent = std::filesystem::path(cache_file_).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream file(cache_file_, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(ptr), length);
        LOG_INFO_FMT("[Int8Calibrator] Wrote calibration cache: {} bytes to {}", length, cache_file_);
    } else {
        LOG_ERROR_FMT("[Int8Calibrator] Failed to write cache: {}", cache_file_);
    }
}

nvinfer1::CalibrationAlgoType Int8EntropyCalibrator::getAlgorithm() noexcept {
    return nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2;
}

void Int8EntropyCalibrator::preprocessBatch(const std::vector<cv::Mat>& images, float* gpu_buffer) {
    int hw = input_height_ * input_width_;
    int batch_stride = input_channels_ * hw;

    for (int b = 0; b < batch_size_; ++b) {
        cv::Mat resized;
        cv::resize(images[b], resized, cv::Size(input_width_, input_height_));

        // BGR → float, 归一化到 [0, 1]
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        const float* img_ptr = float_img.ptr<float>();
        float* batch_ptr = gpu_buffer + b * batch_stride;

        // HWC → NCHW (BGR → RGB for YOLOv8)
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                int src_idx = (h * input_width_ + w) * 3;
                int dst_idx = h * input_width_ + w;
                batch_ptr[0 * hw + dst_idx] = img_ptr[src_idx + 2]; // R
                batch_ptr[1 * hw + dst_idx] = img_ptr[src_idx + 1]; // G
                batch_ptr[2 * hw + dst_idx] = img_ptr[src_idx + 0]; // B
            }
        }
    }
}

void Int8EntropyCalibrator::loadCalibrationImages(const std::string& image_dir) {
    if (!std::filesystem::exists(image_dir)) return;

    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            calibration_image_paths_.push_back(entry.path().string());
        }
    }
}

} // namespace nodes
} // namespace ai_stream
