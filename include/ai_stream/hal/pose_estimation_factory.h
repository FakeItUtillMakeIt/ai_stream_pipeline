// include/ai_stream/hal/pose_estimation_factory.h
// 姿态估计引擎工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_pose_estimation.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ai_stream {
namespace hal {

/**
 * @brief 姿态估计后端类型
 */
enum class PoseEstimationBackend {
    AUTO,       // 自动选择可用后端
    TENSORRT,   // NVIDIA TensorRT
    RKNN,       // Rockchip RKNN
    ASCEND,     // Huawei Ascend OM
    CPU         // CPU fallback
};

/**
 * @brief 姿态估计引擎工厂
 */
class PoseEstimationFactory {
public:
    using Creator = std::function<PoseEstimationEnginePtr()>;

    static PoseEstimationFactory& instance();

    void registerBackend(PoseEstimationBackend type, Creator creator);
    PoseEstimationEnginePtr create(PoseEstimationBackend type = PoseEstimationBackend::AUTO);
    std::vector<std::pair<PoseEstimationBackend, std::string>> getAvailableBackends() const;
    bool isBackendAvailable(PoseEstimationBackend type) const;

private:
    PoseEstimationFactory() = default;
    std::unordered_map<PoseEstimationBackend, Creator> creators_;
};

#define REGISTER_POSE_ESTIMATION_BACKEND(backend_type, class_name) \
    static struct _PoseRegistrar_##class_name { \
        _PoseRegistrar_##class_name() { \
            PoseEstimationFactory::instance().registerBackend( \
                backend_type, []() -> PoseEstimationEnginePtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _pose_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
