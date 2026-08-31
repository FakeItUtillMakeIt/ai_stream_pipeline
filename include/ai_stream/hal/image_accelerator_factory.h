// include/ai_stream/hal/image_accelerator_factory.h
// 图像加速器工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace ai_stream {
namespace hal {

/**
 * @brief 图像加速器后端类型
 */
enum class ImageAcceleratorBackend {
    AUTO,       // 自动选择可用后端
    NPP,        // NVIDIA NPP
    RGA,        // Rockchip RGA (RK3588 2D 加速)
    DVPP,       // Huawei Ascend DVPP
    CPU         // CPU fallback (OpenCV)
};

/**
 * @brief 图像加速器工厂
 */
class ImageAcceleratorFactory {
public:
    using Creator = std::function<ImageAcceleratorPtr()>;

    static ImageAcceleratorFactory& instance();

    void registerBackend(ImageAcceleratorBackend type, Creator creator);
    ImageAcceleratorPtr create(ImageAcceleratorBackend type = ImageAcceleratorBackend::AUTO);
    std::vector<std::pair<ImageAcceleratorBackend, std::string>> getAvailableBackends() const;
    bool isBackendAvailable(ImageAcceleratorBackend type) const;

private:
    ImageAcceleratorFactory() = default;
    std::unordered_map<ImageAcceleratorBackend, Creator> creators_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<ImageAcceleratorBackend, std::pair<bool, std::string>> availability_cache_;
};

#define REGISTER_IMAGE_ACCELERATOR(backend_type, class_name) \
    static struct _ImageAcceleratorRegistrar_##class_name { \
        _ImageAcceleratorRegistrar_##class_name() { \
            ImageAcceleratorFactory::instance().registerBackend( \
                backend_type, []() -> ImageAcceleratorPtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _image_accelerator_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
