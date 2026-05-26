// 测试硬件解码功能的简单示例
#include <iostream>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
}

void test_hardware_decode() {
    std::cout << "=== 硬件解码测试 ===" << std::endl;

    // 检查CUDA硬件解码器
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        std::cout << "❌ h264_cuvid 解码器未找到，尝试其他硬件解码器..." << std::endl;

        // 尝试其他CUDA解码器
        codec = avcodec_find_decoder_by_name("h264_cuda");
        if (!codec) {
            std::cout << "❌ 未找到CUDA硬件解码器，将使用软件解码" << std::endl;
        }
    } else {
        std::cout << "✅ 找到CUDA硬件解码器: " << codec->name << std::endl;
    }

    // 检查支持的硬件设备类型
    enum AVHWDeviceType type = AV_HWDEVICE_TYPE_CUDA;
    std::cout << "✅ CUDA硬件设备类型支持: " << type << std::endl;

    // 列出所有支持的硬件设备类型
    std::cout << "\n=== 支持的硬件设备类型 ===" << std::endl;
    for (int i = 0; i < AV_HWDEVICE_TYPE_DXVA2; i++) {
        const char* name = av_hwdevice_get_type_name((enum AVHWDeviceType)i);
        if (name) {
            std::cout << "  - " << name << std::endl;
        }
    }

    std::cout << "\n=== 测试完成 ===" << std::endl;
}

int main() {
    // 初始化FFmpeg
    avcodec_register_all();

    test_hardware_decode();
    return 0;
}