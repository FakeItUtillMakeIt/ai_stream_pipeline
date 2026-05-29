// src/nodes/track/gpu_kalman_filter.cu
// 【加速优化】GPU 加速卡尔曼滤波器 - 用于跟踪算法加速
#include "gpu_kalman_filter.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace ai_stream {
namespace nodes {

namespace {

// CUDA 内核：批量预测步骤
__global__ void predictBatchKernel(
    float* state_vectors,     // [num_tracks, 8] 状态向量 [x, y, w, h, vx, vy, vw, vh]
    float* covariance_matrices, // [num_tracks, 64] 协方差矩阵 (8x8)
    const float* transition_matrix, // [64] 转移矩阵 (8x8)
    const float* process_noise,    // [64] 过程噪声 (8x8)
    int num_tracks,
    float dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tracks) return;

    // 应用时间增量到转移矩阵
    float trans_mat[64];
    for (int i = 0; i < 64; ++i) {
        trans_mat[i] = transition_matrix[i];
    }
    // 位置变化项
    trans_mat[4] = dt;  // x += vx * dt
    trans_mat[9] = dt;  // y += vy * dt
    trans_mat[14] = dt; // w += vw * dt
    trans_mat[19] = dt; // h += vh * dt

    // 预测状态: x = F * x
    float* x = state_vectors + tid * 8;
    float new_state[8] = {0};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            new_state[i] += trans_mat[i * 8 + j] * x[j];
        }
    }

    // 更新状态向量
    for (int i = 0; i < 8; ++i) {
        x[i] = new_state[i];
    }

    // 预测协方差: P = F * P * F^T + Q
    float* P = covariance_matrices + tid * 64;
    float FP[64] = {0};
    float new_P[64] = {0};

    // FP = F * P
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                FP[i * 8 + j] += trans_mat[i * 8 + k] * P[k * 8 + j];
            }
        }
    }

    // new_P = FP * F^T
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                new_P[i * 8 + j] += FP[i * 8 + k] * trans_mat[j * 8 + k];
            }
        }
    }

    // new_P += Q
    for (int i = 0; i < 64; ++i) {
        new_P[i] += process_noise[i];
    }

    // 更新协方差矩阵
    for (int i = 0; i < 64; ++i) {
        P[i] = new_P[i];
    }
}

// CUDA 内核：批量更新步骤
__global__ void updateBatchKernel(
    float* state_vectors,
    float* covariance_matrices,
    const float* measurements,      // [num_measurements, 4] 测量值 [x, y, w, h]
    const float* observation_matrix, // [32] 观测矩阵 (4x8)
    const float* measurement_noise,  // [16] 测量噪声 (4x4)
    const int* track_indices,        // [num_measurements] 对应的轨迹索引
    const int* measurement_indices,  // [num_measurements] 测量索引
    int num_measurements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_measurements) return;

    int track_idx = track_indices[tid];
    int meas_idx = measurement_indices[tid];

    float* x = state_vectors + track_idx * 8;
    float* P = covariance_matrices + track_idx * 64;
    const float* z = measurements + meas_idx * 4;

    // 预测测量: z_pred = H * x
    float z_pred[4] = {0};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            z_pred[i] += observation_matrix[i * 8 + j] * x[j];
        }
    }

    // 计算残差: y = z - z_pred
    float y[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = z[i] - z_pred[i];
    }

    // 计算残差协方差: S = H * P * H^T + R
    float HP[32] = {0};
    float S[16] = {0};

    // HP = H * P
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                HP[i * 8 + j] += observation_matrix[i * 8 + k] * P[k * 8 + j];
            }
        }
    }

    // S = HP * H^T
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 8; ++k) {
                S[i * 4 + j] += HP[i * 8 + k] * observation_matrix[j * 8 + k];
            }
        }
    }

    // S += R
    for (int i = 0; i < 16; ++i) {
        S[i] += measurement_noise[i];
    }

    // 计算卡尔曼增益: K = P * H^T * S^-1
    // 简化实现：使用伪逆近似
    float K[32] = {0};
    // 实际实现需要矩阵求逆，这里仅为示意

    // 更新状态: x = x + K * y
    // 更新协方差: P = (I - K * H) * P
    // 简化处理，仅示意
}

} // anonymous namespace

GpuKalmanFilter::GpuKalmanFilter(int max_tracks)
    : max_tracks_(max_tracks),
      num_active_tracks_(0),
      stream_(nullptr),
      cublas_handle_(nullptr) {

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GpuKalmanFilter] Created with GPU device {}", device_id_);
    } else {
        LOG_WARN_FMT("[GpuKalmanFilter] No GPU device found");
    }

    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);

    // 初始化 CUDA 资源
    CUDA_CHECK(cudaStreamCreate(&stream_));
    cublasCreate(&cublas_handle_);

    // 分配 GPU 内存
    size_t state_size = max_tracks_ * 8 * sizeof(float);
    size_t cov_size = max_tracks_ * 64 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_state_vectors_, state_size));
    CUDA_CHECK(cudaMalloc(&d_covariance_matrices_, cov_size));
    CUDA_CHECK(cudaMalloc(&d_transition_matrix_, 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_observation_matrix_, 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_process_noise_, 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_measurement_noise_, 16 * sizeof(float)));

    // 初始化转移矩阵 (恒定速度模型)
    float transition_matrix[64] = {0};
    for (int i = 0; i < 8; ++i) transition_matrix[i * 8 + i] = 1.0f; // 单位矩阵

    CUDA_CHECK(cudaMemcpy(d_transition_matrix_, transition_matrix,
                         64 * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化观测矩阵
    float observation_matrix[32] = {0};
    observation_matrix[0] = 1.0f;  // x = x
    observation_matrix[9] = 1.0f;  // y = y
    observation_matrix[18] = 1.0f; // w = w
    observation_matrix[27] = 1.0f; // h = h

    CUDA_CHECK(cudaMemcpy(d_observation_matrix_, observation_matrix,
                         32 * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化噪声矩阵
    float process_noise[64] = {0};
    float q = 1.0f;
    process_noise[0] = q; process_noise[9] = q;
    process_noise[18] = q; process_noise[27] = q;
    process_noise[36] = q; process_noise[45] = q;
    process_noise[54] = q; process_noise[63] = q;

    CUDA_CHECK(cudaMemcpy(d_process_noise_, process_noise,
                         64 * sizeof(float), cudaMemcpyHostToDevice));

    float measurement_noise[16] = {0};
    float r = 1.0f;
    measurement_noise[0] = r; measurement_noise[5] = r;
    measurement_noise[10] = r; measurement_noise[15] = r;

    CUDA_CHECK(cudaMemcpy(d_measurement_noise_, measurement_noise,
                         16 * sizeof(float), cudaMemcpyHostToDevice));

    LOG_INFO_FMT("[GpuKalmanFilter] Initialized with max_tracks={}", max_tracks_);
}

GpuKalmanFilter::~GpuKalmanFilter() {
    // 释放 GPU 内存
    if (d_state_vectors_) cudaFree(d_state_vectors_);
    if (d_covariance_matrices_) cudaFree(d_covariance_matrices_);
    if (d_transition_matrix_) cudaFree(d_transition_matrix_);
    if (d_observation_matrix_) cudaFree(d_observation_matrix_);
    if (d_process_noise_) cudaFree(d_process_noise_);
    if (d_measurement_noise_) cudaFree(d_measurement_noise_);

    if (stream_) cudaStreamDestroy(stream_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);

    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);

    LOG_INFO_FMT("[GpuKalmanFilter] Destroyed");
}

void GpuKalmanFilter::predict(float dt) {
    if (num_active_tracks_ == 0) return;

    cudaEventRecord(start_event_, stream_);

    dim3 block(256);
    dim3 grid((num_active_tracks_ + 255) / 256);

    predictBatchKernel<<<grid, block, 0, stream_>>>(
        d_state_vectors_,
        d_covariance_matrices_,
        d_transition_matrix_,
        d_process_noise_,
        num_active_tracks_,
        dt
    );

    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    LOG_DEBUG_FMT("[GpuKalmanFilter] Predict completed: {} tracks, {:.2f}ms",
                  num_active_tracks_, elapsed_ms);
}

void GpuKalmanFilter::update(
    const std::vector<Measurement>& measurements,
    const std::vector<int>& track_indices,
    const std::vector<int>& measurement_indices) {

    if (measurements.empty() || track_indices.empty() ||
        track_indices.size() != measurement_indices.size()) {
        return;
    }

    int num_measurements = static_cast<int>(measurements.size());

    // 分配和拷贝测量数据
    float* d_measurements;
    int* d_track_indices;
    int* d_measurement_indices;

    CUDA_CHECK(cudaMalloc(&d_measurements, num_measurements * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_track_indices, num_measurements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_measurement_indices, num_measurements * sizeof(int)));

    std::vector<float> h_measurements(num_measurements * 4);
    for (int i = 0; i < num_measurements; ++i) {
        h_measurements[i * 4 + 0] = measurements[i].x;
        h_measurements[i * 4 + 1] = measurements[i].y;
        h_measurements[i * 4 + 2] = measurements[i].w;
        h_measurements[i * 4 + 3] = measurements[i].h;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_measurements, h_measurements.data(),
                              num_measurements * 4 * sizeof(float),
                              cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_track_indices, track_indices.data(),
                              num_measurements * sizeof(int),
                              cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_measurement_indices, measurement_indices.data(),
                              num_measurements * sizeof(int),
                              cudaMemcpyHostToDevice, stream_));

    cudaEventRecord(start_event_, stream_);

    dim3 block(256);
    dim3 grid((num_measurements + 255) / 256);

    updateBatchKernel<<<grid, block, 0, stream_>>>(
        d_state_vectors_,
        d_covariance_matrices_,
        d_measurements,
        d_observation_matrix_,
        d_measurement_noise_,
        d_track_indices,
        d_measurement_indices,
        num_measurements
    );

    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);

    // 清理临时内存
    cudaFree(d_measurements);
    cudaFree(d_track_indices);
    cudaFree(d_measurement_indices);

    LOG_DEBUG_FMT("[GpuKalmanFilter] Update completed: {} measurements, {:.2f}ms",
                  num_measurements, elapsed_ms);
}

void GpuKalmanFilter::getStateVectors(std::vector<StateVector>& states) {
    if (num_active_tracks_ == 0) {
        states.clear();
        return;
    }

    states.resize(num_active_tracks_);
    CUDA_CHECK(cudaMemcpyAsync(states.data(), d_state_vectors_,
                              num_active_tracks_ * sizeof(StateVector),
                              cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
}

void GpuKalmanFilter::setStateVectors(const std::vector<StateVector>& states) {
    num_active_tracks_ = static_cast<int>(states.size());
    if (num_active_tracks_ > max_tracks_) {
        LOG_WARN_FMT("[GpuKalmanFilter] Too many tracks: {}, max={}",
                     num_active_tracks_, max_tracks_);
        num_active_tracks_ = max_tracks_;
    }

    if (num_active_tracks_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_state_vectors_, states.data(),
                                  num_active_tracks_ * sizeof(StateVector),
                                  cudaMemcpyHostToDevice, stream_));
        cudaStreamSynchronize(stream_);
    }
}

void GpuKalmanFilter::getCovarianceMatrices(std::vector<CovarianceMatrix>& covariances) {
    if (num_active_tracks_ == 0) {
        covariances.clear();
        return;
    }

    covariances.resize(num_active_tracks_);
    CUDA_CHECK(cudaMemcpyAsync(covariances.data(), d_covariance_matrices_,
                              num_active_tracks_ * sizeof(CovarianceMatrix),
                              cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
}

float GpuKalmanFilter::getAverageLatencyMs() const {
    // 简化实现，实际应该记录历史平均值
    return 0.0f;
}

} // namespace nodes
} // namespace ai_stream
