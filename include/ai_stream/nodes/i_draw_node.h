// include/ai_stream/nodes/i_draw_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <string>
#include <vector>
#ifdef HAVE_OPENCV_FREETYPE
#include <opencv2/freetype.hpp>
#endif
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

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        if (params.contains("snapshot")) {
            const auto& snapshot_cfg = params["snapshot"];
            if (snapshot_cfg.contains("enabled")) {
                setSnapshotEnabled(snapshot_cfg["enabled"].get<bool>());
            }
            if (snapshot_cfg.contains("interval")) {
                setSnapshotInterval(snapshot_cfg["interval"].get<int>());
            }
            if (snapshot_cfg.contains("dir")) {
                setSnapshotDir(snapshot_cfg["dir"].get<std::string>());
            }
            if (snapshot_cfg.contains("font_file")) {
                setFontFile(snapshot_cfg["font_file"].get<std::string>());
            }
            if (snapshot_cfg.contains("logo_file")) {
                setLogoFile(snapshot_cfg["logo_file"].get<std::string>());
            }
        }
        return true;
    }

    virtual void setLogoFile(const std::string& logo_path)
    {
        logo_file_ = logo_path;
        LOG_INFO_FMT("[IDrawNode] Logo file set: {}", logo_path);
        logo_mat_ = cv::imread(logo_file_, cv::IMREAD_UNCHANGED);
    }

    virtual void setFontFile(const std::string& font_path)
    {
#ifdef HAVE_OPENCV_FREETYPE
        font_file_ = font_path;
        try {
            m_ft2_->loadFontData(font_path, 0);
            LOG_INFO_FMT("[IDrawNode] Loaded font file: {}", font_path);
        } catch (const std::exception& e) {
            LOG_ERROR_FMT("[IDrawNode] Failed to load font file: {}, error: {}", font_path, e.what());
        }
#else
        LOG_WARN_FMT("[IDrawNode] freetype support not compiled, font file ignored: {}", font_path);
#endif
    }

    /**
     * @brief 在图像左上角绘制告警信息面板（实现见 src/nodes/draw/draw_panel.cpp）
     */
    virtual void addPanel(cv::Mat& origin, const std::vector<rules::AlertResult>& alert_results);

protected:
    std::string logo_file_;
    std::string font_file_;
    cv::Mat logo_mat_;
#ifdef HAVE_OPENCV_FREETYPE
    cv::Ptr<cv::freetype::FreeType2> m_ft2_;
#endif
};

} // namespace nodes
} // namespace ai_stream