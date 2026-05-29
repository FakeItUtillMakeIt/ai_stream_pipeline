// int8_calibrator.h
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <memory>
#include <atomic>

namespace ai_stream {
namespace nodes {

/**
 * @brief INT8 Entropy Calibrator for TensorRT
 *
 * Implements INT8 calibration for TensorRT model quantization.
 * Supports both Legacy calibrator and Entropy calibrator.
 */
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator();
    ~Int8EntropyCalibrator() override;

    /**
     * @brief Initialize calibrator with calibration data
     * @param calibration_images List of image paths for calibration
     * @param batch_size Batch size for calibration
     * @param input_height Input image height
     * @param input_width Input image width
     * @param input_name Name of the input tensor
     * @param cache_file Path to calibration cache file
     */
    bool initialize(const std::vector<std::string>& calibration_images,
                   int batch_size,
                   int input_height,
                   int input_width,
                   const std::string& input_name,
                   const std::string& cache_file);

    // IInt8Calibrator interface
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* ptr, size_t length) noexcept override;
    nvinfer1::CalibrationAlgoType getAlgorithm() noexcept override;

private:
    void preprocessBatch(const std::vector<cv::Mat>& images, float* gpu_buffer);
    void loadCalibrationImages(const std::string& image_dir);

    int batch_size_;
    int input_height_;
    int input_width_;
    int input_channels_;
    size_t input_size_;
    std::string input_name_;
    std::string cache_file_;

    std::vector<std::string> calibration_image_paths_;
    std::atomic<int> current_batch_index_;

    void* d_input_buffer_;
    std::vector<float> h_input_buffer_;

    std::vector<char> calibration_cache_;
};

} // namespace nodes
} // namespace ai_stream