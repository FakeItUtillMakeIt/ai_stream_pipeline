// src/nodes/infer/tensorrt_infer.h
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "src/core/frame_queue.h"
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>

// 前向声明 TensorRT 类型
namespace nvinfer1 { 
    class ICudaEngine; 
    class IExecutionContext; 
}

namespace ai_stream {
namespace nodes {

class TensorRTInferNode : public IInferNode {
public:
    TensorRTInferNode();
    ~TensorRTInferNode() override;

    // IInferNode 接口
    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    std::pair<int, int> getInputSize() const override;
    std::vector<std::string> getClassNames() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();
    bool initEngine(const std::string& engine_path);
    
    // 使用自定义删除器的 unique_ptr
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 1;
    std::vector<std::string> class_names_;
    
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{5};
    std::thread worker_;
    std::atomic<bool> running_{false};
};

} // namespace nodes
} // namespace ai_stream