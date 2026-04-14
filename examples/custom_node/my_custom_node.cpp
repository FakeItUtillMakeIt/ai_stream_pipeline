#include "my_custom_node.h"
#include "registry/node_factory.h"
#include "ai_stream/core/packet.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace my_project {

GrayscaleNode::GrayscaleNode() : ai_stream::core::Node("Grayscale") {}

bool GrayscaleNode::start() {
    LOG_INFO_FMT("[GrayscaleNode] Started");
    return true;
}

void GrayscaleNode::stop() {
    LOG_INFO_FMT("[GrayscaleNode] Stopped");
}

void GrayscaleNode::pushData(std::shared_ptr<ai_stream::core::BasePacket> packet) {
    if (packet->type != ai_stream::core::PacketType::DECODED_FRAME) {
        // 非视频帧直接透传
        broadcast(packet);
        return;
    }

    auto frame = std::static_pointer_cast<ai_stream::core::VideoFramePacket>(packet);
    if (!frame->mat || frame->mat->empty()) return;

    // 创建新的 Mat 存放灰度图
    auto gray_mat = std::make_shared<cv::Mat>();
    cv::cvtColor(*frame->mat, *gray_mat, cv::COLOR_BGR2GRAY);
    // 再转换回三通道以便后续画框节点正常工作
    cv::cvtColor(*gray_mat, *gray_mat, cv::COLOR_GRAY2BGR);

    // 构造新的数据包
    auto new_frame = std::make_shared<ai_stream::core::VideoFramePacket>();
    new_frame->stream_id = frame->stream_id;
    new_frame->timestamp_ms = frame->timestamp_ms;
    new_frame->mat = gray_mat;
    new_frame->width = gray_mat->cols;
    new_frame->height = gray_mat->rows;
    new_frame->channels = 3;

    broadcast(new_frame);
}

} // namespace my_project

// 注册节点到全局工厂，使其可以通过 JSON 配置中的 "type": "grayscale" 创建
REGISTER_NODE("grayscale", my_project::GrayscaleNode)