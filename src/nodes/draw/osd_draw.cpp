// src/nodes/draw/osd_draw.cpp
#include "utils/time_util.h"
#include "osd_draw.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

namespace ai_stream {
namespace nodes {

OSDDrawNode::OSDDrawNode() : core::QueuedNode<IDrawNode>("OSDDraw") {
#ifdef HAVE_OPENCV_FREETYPE
    m_ft2_ = cv::freetype::createFreeType2();
#endif
}

void OSDDrawNode::setBoxColor(int b, int g, int r) {
    box_color_ = cv::Scalar(b, g, r);
}

void OSDDrawNode::setFontThickness(int thickness) {
    font_thickness_ = thickness;
}

void OSDDrawNode::setShowConfidence(bool show) {
    show_confidence_ = show;
}

void OSDDrawNode::setClassFilter(const std::vector<int>& class_ids) {
    class_filter_ = class_ids;
}

bool OSDDrawNode::onStartup()
{
    // 如果启用了快照，创建保存目录
    if (snapshot_enabled_) {
        if (!std::filesystem::exists(snapshot_dir_)) {
            try {
                std::filesystem::create_directories(snapshot_dir_);
                LOG_INFO_FMT("[OSDDraw] Created snapshot directory: {}", snapshot_dir_);
            } catch (const std::exception& e) {
                LOG_ERROR_FMT("[OSDDraw] Failed to create snapshot directory: {}", e.what());
                snapshot_enabled_ = false;
            }
        }
    }
    
    LOG_INFO_FMT("[OSDDraw] Started (snapshot: {}, interval: {})", 
                 snapshot_enabled_.load(), snapshot_interval_.load());
    return true;
}

void OSDDrawNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[OSDDraw] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 workerLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }
    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer_result->source_frame || infer_result->source_frame->source_mat->empty() || !infer_result->source_frame->mat || infer_result->source_frame->mat->empty()) {
        LOG_WARN_FMT("[OSDDraw] Inference result missing source frame");
        return;
    }

    // 克隆一份图像避免影响其他分支（如果该帧还要被其他节点使用）
    auto draw_mat = std::make_shared<cv::Mat>(infer_result->source_frame->source_mat->clone());
    
    for (const auto& det : infer_result->detections) {
        // 类别过滤
        if (!class_filter_.empty() && 
            std::find(class_filter_.begin(), class_filter_.end(), det.class_id) == class_filter_.end()) {
            continue;
        }

        cv::Rect rect(static_cast<int>(det.x), static_cast<int>(det.y),
                      static_cast<int>(det.w), static_cast<int>(det.h));
        LOG_DEBUG_FMT("[OSDDraw] Drawing detection: {} {} ({}, {}, {}, {})",std::to_string(det.class_id).c_str(), det.confidence, rect.x, rect.y, rect.width, rect.height);
        cv::rectangle(*draw_mat, rect, box_color_, font_thickness_);
   
        if (det.has_keypoints)
        {
            // 1. 先画骨架连线（先画线，再画点，避免点盖住线）
            for (const auto& [i, j] : core::SKELETON)
            {
                const auto& kp1 = det.keypoints[i];
                const auto& kp2 = det.keypoints[j];
                
                // 两个点都可见才连线
                if (kp1.visible && kp2.visible)
                {
                    cv::Point p1(static_cast<int>(kp1.x), static_cast<int>(kp1.y));
                    cv::Point p2(static_cast<int>(kp2.x), static_cast<int>(kp2.y));
                    cv::line(*draw_mat, p1, p2, cv::Scalar(255, 128, 0), 1, cv::LINE_AA);
                }
            }

            // 2. 再画关键点（点和序号）
            for (int k = 0; k < 17; ++k)
            {
                const auto& kp = det.keypoints[k];
                if (!kp.visible) continue;  // 不可见跳过，或画灰色点
                
                cv::Point pt(static_cast<int>(kp.x), static_cast<int>(kp.y));
                
                // 画实心圆
                cv::circle(*draw_mat, pt, 5, cv::Scalar(0, 255, 0), -1);
                // 白边
                cv::circle(*draw_mat, pt, 5, cv::Scalar(255, 255, 255), 1);
                
                // 标注序号（小字体）
                cv::putText(*draw_mat, std::to_string(k), 
                            cv::Point(pt.x + 8, pt.y - 8),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }
        std::string label = det.class_name;
        std::string track_info = "ID:" + std::to_string(det.track_id);
        if (show_confidence_) {
            label += " " + std::to_string(det.confidence).substr(0, 4);
        }
        if (det.track_id >= 0) {
            label += " " + track_info;
        }
        cv::putText(*draw_mat, label, 
                    cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color_, font_thickness_);
        LOG_DEBUG_FMT("[OSDDraw] Label: {}, track_id: {}", label.c_str(), det.track_id);
    }
    addPanel(*draw_mat, infer_result->alert_result);
    LOG_DEBUG_FMT("[OSDDraw] Drawing {} detections", infer_result->detections.size());

    // 构造新的视频帧包（包含绘制后的图像）
    auto drawn_frame = std::make_shared<core::VideoFramePacket>();
    drawn_frame->stream_id = infer_result->stream_id;
    drawn_frame->timestamp_ms = infer_result->timestamp_ms;
    drawn_frame->mat = draw_mat;
    drawn_frame->width = draw_mat->cols;
    drawn_frame->height = draw_mat->rows;
    drawn_frame->channels = draw_mat->channels();

    frame_count_++;
    if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) { 
        saveSnapshot(drawn_frame,frame_count_);
    }

    broadcast(drawn_frame);
}

void OSDDrawNode::setSnapshotEnabled(bool enabled)
{
    snapshot_enabled_ = enabled;
    LOG_INFO_FMT("[OSDDraw] Snapshot enabled: {}", enabled);
}

void OSDDrawNode::setSnapshotInterval(int interval)
{
    snapshot_interval_ = interval;
    snapshot_interval_ = interval > 0 ? interval : 100;
    LOG_INFO_FMT("[OSDDraw] Snapshot interval: {} frames", snapshot_interval_.load());
}

void OSDDrawNode::setSnapshotDir(const std::string& dir)
{
    snapshot_dir_ = dir;
    LOG_INFO_FMT("[OSDDraw] Snapshot directory: {}", dir);
}

void OSDDrawNode::saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num)
{
    if(!frame || !frame->mat || frame->mat->empty())
        return;
    try {
        // 生成文件名
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count() % 1000000;
        
        std::tm tm_buf = utils::TimeUtil::safeLocaltime(time_t);
        
        std::stringstream ss;
        ss << snapshot_dir_ << "/"
           << "stream_" << frame->stream_id << "_"
           << "frame_" << std::setfill('0') << std::setw(6) << frame_num << "_"
           << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_"
           << std::setfill('0') << std::setw(6) << us
           << ".jpg";
        
        std::string filename = ss.str();
        
        // 保存图片
        bool success = cv::imwrite(filename, *frame->mat);
        
        if (success) {
            snapshot_count_++;
            LOG_INFO_FMT("[OSDDraw] Snapshot saved: {} (frame #{}, total: {})", 
                         filename, frame_num, snapshot_count_);
        }
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[OSDDraw] Exception saving snapshot: {}", e.what());
    }
}

REGISTER_NODE("osd_draw", OSDDrawNode)

} // namespace nodes
} // namespace ai_stream