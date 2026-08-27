// include/ai_stream/hal/i_pose_estimation.h
// 姿态估计抽象接口——隔离 TensorRT / RKNN / Ascend 等后端
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief 姿态估计配置
 */
struct PoseEstimationConfig {
    std::string model_path;
    int input_width = 640;
    int input_height = 640;
    int max_batch = 8;         // 单帧最大人数
    std::string precision = "fp16";
};

/**
 * @brief 姿态估计引擎抽象接口
 *
 * 输入：按人 crop + letterbox 预处理后的 NCHW float 数据 [batch, 3, H, W]。
 * 输出：原始模型输出张量（host），由节点层做关键点解码（与业务数据结构耦合）。
 *
 * 三种推理路径，后端按能力实现，不支持的路径返回 false：
 * - inferHost            : host 输入缓冲区
 * - inferDevice          : 设备端输入缓冲区
 * - inferFromDeviceImage : 设备端原图 + 检测框，后端内部完成 GPU 预处理
 */
class IPoseEstimationEngine {
public:
    virtual ~IPoseEstimationEngine() = default;

    /**
     * @brief 加载模型
     */
    virtual bool loadModel(const PoseEstimationConfig& config) = 0;

    /**
     * @brief host 输入推理
     * @param input_nchw 已预处理的 NCHW float 数据（host）
     * @param num_persons batch 大小
     * @param output_host 输出张量（函数内 resize）
     * @return 是否支持并成功
     */
    virtual bool inferHost(const float* input_nchw, int num_persons,
                           std::vector<float>& output_host) {
        (void)input_nchw; (void)num_persons; (void)output_host;
        return false;
    }

    /**
     * @brief 设备端输入推理
     * @param d_input_nchw 已预处理的 NCHW float 数据（device）
     * @return 是否支持并成功
     */
    virtual bool inferDevice(const void* d_input_nchw, int num_persons,
                             std::vector<float>& output_host) {
        (void)d_input_nchw; (void)num_persons; (void)output_host;
        return false;
    }

    /**
     * @brief 设备端原图端到端推理（GPU 预处理 + 推理）
     * @param d_src_bgr 设备端 BGR 图像（uint8 packed）
     * @param boxes_7 检测框参数 [x1,y1,x2,y2,scale,pad_x,pad_y] x num_persons
     * @return 是否支持并成功
     */
    virtual bool inferFromDeviceImage(const void* d_src_bgr,
                                      int src_w, int src_h, size_t src_pitch,
                                      const std::vector<float>& boxes_7,
                                      int num_persons,
                                      std::vector<float>& output_host) {
        (void)d_src_bgr; (void)src_w; (void)src_h; (void)src_pitch;
        (void)boxes_7; (void)num_persons; (void)output_host;
        return false;
    }

    /** 单人输出张量大小（float 元素个数） */
    virtual size_t getOutputFloatsPerPerson() const = 0;

    /** 模型输入尺寸 {width, height} */
    virtual std::pair<int, int> getInputSize() const = 0;

    virtual std::string getBackendName() const = 0;
    virtual bool isAvailable() const = 0;

    /** 设置外部 CUDA 流（以 void* 传递，避免接口泄漏 CUDA 头文件） */
    virtual void setCudaStream(void* stream) { (void)stream; }
};

using PoseEstimationEnginePtr = std::unique_ptr<IPoseEstimationEngine>;

} // namespace hal
} // namespace ai_stream
