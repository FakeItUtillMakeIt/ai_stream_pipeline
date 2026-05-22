// src/nodes/infer/tensor_rt_logger.h
#include "3rd_party/log_mgr/log_mgr.h"
#include <NvInfer.h>

// ============================================================
// TensorRT Logger
// ============================================================
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINFO) return;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                LOG_ERROR_FMT("[TensorRT] INTERNAL_ERROR: {}", msg);
                break;
            case Severity::kERROR:
                LOG_ERROR_FMT("[TensorRT] ERROR: {}", msg);
                break;
            case Severity::kWARNING:
                LOG_WARN_FMT("[TensorRT] WARNING: {}", msg);
                break;
            default:
                LOG_DEBUG_FMT("[TensorRT] {}", msg);
                break;
        }
    }
};

static TensorRTLogger g_logger;