// src/nodes/draw/osd_draw.cpp
#include "osd_draw.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

namespace ai_stream {
namespace nodes {

OSDDrawNode::OSDDrawNode() : IDrawNode("OSDDraw") {
    m_ft2 = cv::freetype::createFreeType2();
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

bool OSDDrawNode::start()
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

void OSDDrawNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[OSDDraw] Received stream end");
        stop();
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
                    cv::line(*draw_mat, p1, p2, cv::Scalar(255, 128, 0), 2, cv::LINE_AA);
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

void OSDDrawNode::setLogoFile(const std::string& logo_path)
{
    logo_file_ = logo_path;
    LOG_INFO_FMT("[OSDDraw] Logo file set: {}", logo_path);
}

void OSDDrawNode::setFontFile(const std::string& font_path)
{
    font_file_ = font_path;
    try {
        m_ft2->loadFontData(font_path, 0);
        LOG_INFO_FMT("[OSDDraw] Loaded font file: {}", font_path);
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[OSDDraw] Failed to load font file: {}, error: {}", font_path, e.what());
    }
}

void OSDDrawNode::addPanel(cv::Mat& origin, const std::vector<rules::AlertResult>& alert_results)
{
    if (origin.empty()) return;

    // ============================================================
    // 1. 按业务类别分组
    // ============================================================
    std::vector<rules::AlertEvent> person_behaviors;
    std::vector<rules::AlertEvent> safety_items;
    
    for (const auto& alert : alert_results) {
        for (const auto& alert_event : alert.alert_events) { 
            // 根据实际业务调整分类；示例：ALERT_UNKNOWN 归为劳保，其他为人员行为
            if (alert_event.alert_type == rules::AlertType::ALERT_UNKNOWN) {
                safety_items.push_back(alert_event);
            } else {
                person_behaviors.push_back(alert_event);
            }
        }
    }

    // ============================================================
    // 2. 分辨率自适应
    // ============================================================
    const int BASE_WIDTH = 1920;
    const int BASE_HEIGHT = 1080;
    double scale_ratio = std::min(static_cast<double>(origin.cols) / BASE_WIDTH,
                                  static_cast<double>(origin.rows) / BASE_HEIGHT);

    int margin = static_cast<int>(20 * scale_ratio);
    int padding = static_cast<int>(10 * scale_ratio);
    int line_height = static_cast<int>(32 * scale_ratio);
    double font_scale = 1.2 * scale_ratio;
    int font_thickness = std::max(1, static_cast<int>(1 * scale_ratio));

    // ============================================================
    // 3. 计算画布尺寸（含 Logo 区域）
    // ============================================================
    int panel_width = static_cast<int>(464 * scale_ratio);
    panel_width = std::min(panel_width, origin.cols - margin * 2);

    int logo_area_h = 0;
    cv::Mat logo;
    {
        std::string logo_path = logo_file_;
        logo = cv::imread(logo_path, cv::IMREAD_UNCHANGED);
        if (!logo.empty()) {
            double logo_scale = static_cast<double>(panel_width) / (logo.cols * 2.0);
            int target_h = static_cast<int>(logo.rows * logo_scale);
            logo_area_h = target_h + static_cast<int>(10 * scale_ratio);
        }
    }

    int total_lines = 0;
    int person_rows = static_cast<int>((person_behaviors.size() + 3 - 1) / 3);
    int safety_rows = static_cast<int>((safety_items.size() + 2 - 1) / 2);

    if (!person_behaviors.empty()) total_lines += person_rows + 2;
    if (!safety_items.empty())     total_lines += safety_rows + 2;
    if (total_lines == 0)          total_lines = 2;

    int logo_lines = (logo_area_h + line_height - 1) / line_height + 1;
    total_lines += logo_lines;

    int panel_height = total_lines * line_height + padding * 2;
    panel_height = std::min(panel_height, origin.rows - margin * 2);
    
    cv::Rect panel_rect(0, 0, panel_width, panel_height);
    panel_rect &= cv::Rect(0, 0, origin.cols, origin.rows);
    if (panel_rect.width <= 0 || panel_rect.height <= 0) return;

    // ============================================================
    // 4. 创建灰色画布
    // ============================================================
    cv::Mat panel(panel_rect.height, panel_rect.width, CV_8UC3, cv::Scalar(200, 200, 200));
    int content_y = padding;

    // ============================================================
    // 5. 绘制 Logo
    // ============================================================
    if (!logo.empty()) {
        double logo_scale = static_cast<double>(panel_width) / (logo.cols * 2.0);
        int target_w = static_cast<int>(logo.cols * logo_scale);
        int target_h = static_cast<int>(logo.rows * logo_scale);
        cv::Mat resized_logo;
        cv::resize(logo, resized_logo, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

        int logo_x = (panel_width - target_w) / 2;
        int logo_y = static_cast<int>(5 * scale_ratio);
        cv::Rect logo_roi(logo_x, logo_y, target_w, target_h);
        logo_roi &= cv::Rect(0, 0, panel_width, panel_height);

        if (resized_logo.channels() == 4) {
            cv::Mat logo_bgr, logo_alpha;
            cv::cvtColor(resized_logo, logo_bgr, cv::COLOR_BGRA2BGR);
            cv::extractChannel(resized_logo, logo_alpha, 3);
            cv::Mat roi = panel(logo_roi);
            for (int y = 0; y < roi.rows; ++y) {
                for (int x = 0; x < roi.cols; ++x) {
                    cv::Vec3b& p = roi.at<cv::Vec3b>(y, x);
                    cv::Vec3b l = logo_bgr.at<cv::Vec3b>(y, x);
                    float a = logo_alpha.at<uchar>(y, x) / 255.0f;
                    p[0] = static_cast<uchar>(p[0] * (1.0f - a) + l[0] * a);
                    p[1] = static_cast<uchar>(p[1] * (1.0f - a) + l[1] * a);
                    p[2] = static_cast<uchar>(p[2] * (1.0f - a) + l[2] * a);
                }
            }
        } else {
            resized_logo.copyTo(panel(logo_roi));
        }
        content_y = logo_roi.y + logo_roi.height + static_cast<int>(5 * scale_ratio);
    }

    // ============================================================
    // 6. 文字绘制辅助
    // ============================================================
    auto putTextEx = [&](cv::Mat& img, const std::string& text, cv::Point pos,
                         double scale, cv::Scalar color, int thickness) {
        if (m_ft2 && !text.empty()) {
            // 包含非 ASCII 字符时用 FreeType
            bool has_cn = false;
            for (unsigned char c : text) { if (c > 127) { has_cn = true; break; } }
            if (has_cn) {
                int ft_height = static_cast<int>(scale * 20);
                m_ft2->putText(img, text, pos, ft_height, color, -1, cv::LINE_AA, true);
                return;
            }
        }
        cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2, cv::LINE_AA);
    };

    // ============================================================
    // 7. 绘制单类别区块（标题 + 多列项）
    // ============================================================
    auto drawCategory = [&](const std::vector<rules::AlertEvent>& items,
                            const std::string& title,
                            int items_per_row,
                            int x_spacing_factor)
    {
        if (items.empty()) return;

        // 标题（蓝色）
        cv::Scalar title_color(200, 0, 0);
        putTextEx(panel, title, cv::Point(padding, content_y + line_height),
                  font_scale * 1.5, title_color, font_thickness);
        content_y += static_cast<int>(line_height * 1.2);

        int item_count = 0;
        int line_x = padding;
        int x_spacing = static_cast<int>(padding * x_spacing_factor);

        for (const auto& item : items) {
            std::string text = rules::alertTypeChMap.at(item.alert_type);
            // 防御：如果名称为空，用 alert_type 兜底
            if (text.empty()) {
                text = "Alert_" + rules::alertTypeChMap.at(item.alert_type);
            }

            // 告警状态非默认时红色，否则绿色
            cv::Scalar color = (item.status != rules::AlertStatus::ALERT_STATUS_DEFAULT)
                               ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 150, 0);

            LOG_INFO_FMT("[OSDDraw] Panel item: {}", text);

            putTextEx(panel, text, cv::Point(padding, content_y + line_height),
                      font_scale * 1.2, color, font_thickness);

            item_count++;
            line_x += x_spacing;

            // 每行满额换行
            if (item_count % items_per_row == 0) {
                content_y += line_height;
                line_x = padding;
            }
        }
        if (item_count % items_per_row != 0) {
            content_y += line_height;
        }
        content_y += static_cast<int>(line_height * 0.5); // 类别间留白
    };

    LOG_INFO_FMT("[OSDDraw] person behavior: {}, safety item: {}", 
                 person_behaviors.size(), safety_items.size());
    drawCategory(person_behaviors, "人员行为", 3, 8);
    drawCategory(safety_items, "劳保用品", 2, 12);

    // ============================================================
    // 8. 半透明叠加到原图
    // ============================================================
    cv::Mat roi_img = origin(panel_rect);
    double alpha = 0.8;
    double beta = 1.0 - alpha;
    cv::addWeighted(panel, alpha, roi_img, beta, 0, roi_img);
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
        
        std::tm tm_buf;
        localtime_r(&time_t, &tm_buf);
        
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