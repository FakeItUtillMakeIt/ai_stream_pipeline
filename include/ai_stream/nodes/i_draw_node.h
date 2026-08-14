// include/ai_stream/nodes/i_draw_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <string>
#include <vector>
#include <opencv2/freetype.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace ai_stream {
namespace nodes {

/**
 * @brief 画框节点接口
 * 
 * 将推理结果（检测框、关键点等）绘制到原始图像帧上，输出带标注的图像。
 */
class IDrawNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置边框颜色 (B, G, R)
     */
    virtual void setBoxColor(int b, int g, int r) = 0;

    /**
     * @brief 设置字体厚度
     */
    virtual void setFontThickness(int thickness) = 0;

    /**
     * @brief 设置是否显示置信度
     */
    virtual void setShowConfidence(bool show) = 0;

    /**
     * @brief 设置类别过滤列表（只绘制这些类别的框）
     * @param class_ids 要显示的类别 ID 列表，空表示全部显示
     */
    virtual void setClassFilter(const std::vector<int>& class_ids) = 0;
    /**
     * @brief 是否启用快照
     * @param packet 包含原始视频数据的数据包
     */
    virtual void setSnapshotEnabled(bool enabled) = 0;

    /**
     * @brief 设置快照间隔
     */
    virtual void setSnapshotInterval(int interval) = 0;

    /**
     * @brief 设置快照保存目录
     *  @param packet 待处理的数据包
     */
    virtual void setSnapshotDir(const std::string& dir) = 0;

    virtual void setLogoFile(const std::string& logo_path)
    {
        logo_file_ = logo_path;
        LOG_INFO_FMT("[IDrawNode] Logo file set: {}", logo_path);
    }

    virtual void setFontFile(const std::string& font_path)
    {
        font_file_ = font_path;
        try {
            m_ft2_->loadFontData(font_path, 0);
            LOG_INFO_FMT("[IDrawNode] Loaded font file: {}", font_path);
        } catch (const std::exception& e) {
            LOG_ERROR_FMT("[IDrawNode] Failed to load font file: {}, error: {}", font_path, e.what());
        }
    }

    virtual void addPanel(cv::Mat& origin, const std::vector<rules::AlertResult>& alert_results)
    {
        if (origin.empty()) return;

        // ============================================================
        // 1. 按业务类别分组
        // ============================================================
        std::vector<rules::AlertEvent> person_behaviors;
        std::vector<rules::AlertEvent> safety_items;
        std::vector<rules::AlertEvent> scene_recognitions;
        
        for (const auto& alert : alert_results) {
            for (const auto& alert_event : alert.alert_events) { 
                LOG_INFO_FMT("[IDrawNode] Alert: {}, Type: {}", alert_event.alert_name, static_cast<int>(alert_event.alert_type));
                // 根据实际业务调整分类；示例：ALERT_UNKNOWN 归为劳保，其他为人员行为
                if (alert_event.alert_item_type == rules::AlertItemType::ITEM_SAFETY_ITEM) {
                    safety_items.push_back(alert_event);
                }
                else if(alert_event.alert_item_type == rules::AlertItemType::ITEM_PERSON_BEHAVIOR)
                {
                    person_behaviors.push_back(alert_event);
                }
                else if (alert_event.alert_item_type == rules::AlertItemType::ITEM_SCENE_RECOGNITION){
                    scene_recognitions.push_back(alert_event);
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
        int scene_rows = static_cast<int>((scene_recognitions.size() + 2 - 1) / 2);

        if (!person_behaviors.empty()) total_lines += person_rows + 2;
        if (!safety_items.empty())     total_lines += safety_rows + 2;
        if (!scene_recognitions.empty()) total_lines += scene_rows + 2;
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
            if (m_ft2_ && !text.empty()) {
                // 包含非 ASCII 字符时用 FreeType
                bool has_cn = false;
                for (unsigned char c : text) { if (c > 127) { has_cn = true; break; } }
                if (has_cn) {
                    int ft_height = static_cast<int>(scale * 20);
                    m_ft2_->putText(img, text, pos, ft_height, color, -1, cv::LINE_AA, true);
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
            int x_spacing = static_cast<int>(padding * x_spacing_factor)* 2.1;

            for (const auto& item : items) {
                std::string text;
                if (rules::alertTypeChMap.find(item.alert_type) != rules::alertTypeChMap.end()) {
                    text = rules::alertTypeChMap.at(item.alert_type);
                } else {
                    LOG_INFO_FMT("[OSDDraw] Unknown alert type: {}", static_cast<uint8_t>(item.alert_type));
                    text = "Alert_" + std::to_string(static_cast<uint8_t>(item.alert_type));
                }
                
                // 告警状态非默认时红色，否则绿色
                cv::Scalar color = (item.status != rules::AlertStatus::ALERT_STATUS_DEFAULT)
                                ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 150, 0);

                //LOG_INFO_FMT("[OSDDraw] Panel item: {}", text);

                putTextEx(panel, text, cv::Point(line_x, content_y + line_height),
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

        // LOG_INFO_FMT("[OSDDraw] person behavior: {}, safety item: {}", 
        //             person_behaviors.size(), safety_items.size());
        drawCategory(person_behaviors, "人员行为", 3, 8);
        drawCategory(safety_items, "劳保用品", 2, 12);
        drawCategory(scene_recognitions, "场景识别", 2, 12);

        // ============================================================
        // 8. 半透明叠加到原图
        // ============================================================
        cv::Mat roi_img = origin(panel_rect);
        double alpha = 0.8;
        double beta = 1.0 - alpha;
        cv::addWeighted(panel, alpha, roi_img, beta, 0, roi_img);
    }

    public:
        std::string logo_file_;
        std::string font_file_;
        cv::Ptr<cv::freetype::FreeType2> m_ft2_;
        bool m_ft2_initialized_ = false;
};

} // namespace nodes
} // namespace ai_stream