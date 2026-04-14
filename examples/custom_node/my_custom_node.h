#pragma once

#include "ai_stream/core/node.h"
#include "ai_stream/core/packet.h"
#include <opencv2/opencv.hpp>

namespace my_project {

class GrayscaleNode : public ai_stream::core::Node {
public:
    GrayscaleNode();
    ~GrayscaleNode() override = default;

    // Node 接口实现
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<ai_stream::core::BasePacket> packet) override;
};

} // namespace my_project