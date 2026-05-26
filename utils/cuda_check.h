#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG_ERROR_FMT(" CUDA error : {}", cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

#define CUDA_CHECK_BOOL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG_ERROR_FMT(" CUDA error : {}", cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)