// include/ai_stream/hal/backend_diagnostics.h
// HAL 后端诊断——聚合各工厂的可用后端信息，供启动日志与 HTTP API 使用
#pragma once

#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief 输出一次各 HAL 工厂的可用后端诊断日志
 *
 * 覆盖：inference / detection / pose_estimation / action_recognition /
 *       image_accelerator / video_codec 六类工厂。
 * 典型调用点：服务进程启动时（ApiServer 构造）。
 */
void logAvailableBackends();

/**
 * @brief 聚合各 HAL 工厂可用后端的 JSON 字符串
 *
 * 形如：
 * {
 *   "inference":         [{"backend": "tensorrt", "name": "TensorRT (NVIDIA)"}, ...],
 *   "detection":         [...],
 *   "pose_estimation":   [...],
 *   "action_recognition":[...],
 *   "image_accelerator": [...],
 *   "video_codec":       [...]
 * }
 */
std::string availableBackendsJson();

} // namespace hal
} // namespace ai_stream
