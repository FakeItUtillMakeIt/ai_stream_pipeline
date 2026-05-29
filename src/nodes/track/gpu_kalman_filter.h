// src/nodes/track/gpu_kalman_filter.h
// 【加速优化】GPU 加速卡尔曼滤波器
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>

namespace ai_stream {
namespace nodes {

/**
 * @brief GPU 加速卡尔曼滤波器
 *
 * 使用 CUDA 批量处理多个跟踪器的预测和更新步骤。
 * 适合高跟踪数量场景（>50 个目标）。
 */
class GpuKalmanFilter {
public:
    struct Measurement {
        float x, y, w, h;
    };

    struct StateVector {
        float x, y, w, h;     // 位置
        float vx, vy, vw, vh;  // 速度
    };

    using CovarianceMatrix = std::array<float, 64>; // 8x8 矩阵

    explicit GpuKalmanFilter(int max_tracks = 100);
    ~GpuKalmanFilter();

    // 预测步骤
    void predict(float dt = 1.0f);

    // 更新步骤
    void update(
        const std::vector<Measurement>& measurements,
        const std::vector<int>& track_indices,
        const std::vector<int>& measurement_indices
    );

    // 获取/设置状态
    void getStateVectors(std::vector<StateVector>& states);
    void setStateVectors(const std::vector<StateVector>& states);
    void getCovarianceMatrices(std::vector<CovarianceMatrix>& covariances);

    int getActiveTrackCount() const { return num_active_tracks_; }
    float getAverageLatencyMs() const;

    // 设备管理
    void setGpuDeviceId(int device_id) {
        if (device_id_ != device_id) {
            device_id_ = device_id;
            cudaSetDevice(device_id_);
        }
    }
    int getGpuDeviceId() const { return device_id_; }

private:
    int max_tracks_;
    int num_active_tracks_;
    int device_id_ = 0;

    // CUDA 资源
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

    // GPU 缓冲区
    float* d_state_vectors_ = nullptr;          // [max_tracks, 8]
    float* d_covariance_matrices_ = nullptr;    // [max_tracks, 64]
    float* d_transition_matrix_ = nullptr;      // [64]
    float* d_observation_matrix_ = nullptr;     // [32]
    float* d_process_noise_ = nullptr;          // [64]
    float* d_measurement_noise_ = nullptr;      // [16]
};

} // namespace nodes
} // namespace ai_stream
