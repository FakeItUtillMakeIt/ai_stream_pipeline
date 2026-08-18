// include/ai_stream/nodes/i_tracker_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "ai_stream/core/packet.h"
#include <string>

namespace ai_stream {
namespace nodes {

/**
 * @brief 跟踪器类型枚举
 */
enum class TrackerType {
    OCSORT,
    BYTETRACK,
    SORT
};

/**
 * @brief OCSort 专用配置
 */
struct OCSortConfig {
    float det_thresh = 0.3f;        // 检测阈值
    int max_age = 30;               // 最大丢失帧数
    int min_hits = 3;              // 最小命中次数
    float iou_threshold = 0.3f;    // IoU 阈值
    int delta_t = 3;               // 时间窗口
    std::string asso_func = "iou"; // 关联函数 ("iou" 或 "diou")
    float inertia = 0.2f;          // 惯性系数
    bool use_byte = false;         // 是否使用 ByteTrack 的低分检测策略
};

/**
 * @brief ByteTrack 专用配置
 */
struct ByteTrackConfig {
    int frame_rate = 30;           // 帧率
    int track_buffer = 30;         // 跟踪缓冲区（等同于 max_age）
    float track_thresh = 0.5f;     // 跟踪阈值
    float high_thresh = 0.6f;      // 高置信度阈值
    float match_thresh = 0.8f;     // 匹配阈值
};

/**
 * @brief 跟踪节点接口
 */
class ITrackerNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置跟踪器类型
     */
    virtual void setTrackerType(TrackerType type) = 0;
    
    /**
     * @brief 获取跟踪器类型
     */
    virtual TrackerType getTrackerType() const = 0;

    /**
     * @brief 设置订阅的流 ID（用于多流跟踪器区分输入数据来源）
     */
    virtual void setSubStreamId(const std::string& stream_id) = 0;

    /**
     * @brief 设置跟踪器 ID
     */
    virtual void setTrackerId(const std::string& id) = 0;
    /**
     * @brief 设置 OCSort 配置
     */
    virtual void setOCSortConfig(const OCSortConfig& config) = 0;
    
    /**
     * @brief 设置 ByteTrack 配置
     */
    virtual void setByteTrackConfig(const ByteTrackConfig& config) = 0;
    
    /**
     * @brief 获取当前活跃的跟踪数量
     */
    virtual int getActiveTrackCount() const = 0;

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        setTrackerId(node_id);
        if (params.contains("tracker_type")) {
            std::string t = params["tracker_type"].get<std::string>();
            if (t == "ocsort") {
                setTrackerType(TrackerType::OCSORT);
                OCSortConfig ocsort_config;
                if (params.contains("ocsort_config")) {
                    const auto& cfg = params["ocsort_config"];
                    ocsort_config.det_thresh = cfg.value("det_thresh", 0.3f);
                    ocsort_config.max_age = cfg.value("max_age", 30);
                    ocsort_config.min_hits = cfg.value("min_hits", 3);
                    ocsort_config.iou_threshold = cfg.value("iou_threshold", 0.3f);
                    ocsort_config.delta_t = cfg.value("delta_t", 3);
                    ocsort_config.asso_func = cfg.value("asso_func", "iou");
                    ocsort_config.inertia = cfg.value("inertia", 0.2f);
                    ocsort_config.use_byte = cfg.value("use_byte", false);
                }
                setOCSortConfig(ocsort_config);
            } else if (t == "bytetrack") {
                setTrackerType(TrackerType::BYTETRACK);
                ByteTrackConfig bytetrack_config;
                if (params.contains("bytetrack_config")) {
                    const auto& cfg = params["bytetrack_config"];
                    bytetrack_config.frame_rate = cfg.value("frame_rate", 30);
                    bytetrack_config.track_buffer = cfg.value("track_buffer", 30);
                    bytetrack_config.track_thresh = cfg.value("track_thresh", 0.5f);
                    bytetrack_config.high_thresh = cfg.value("high_thresh", 0.6f);
                    bytetrack_config.match_thresh = cfg.value("match_thresh", 0.8f);
                }
                setByteTrackConfig(bytetrack_config);
            }
        }
        if (params.contains("sub_stream_id") && params["sub_stream_id"].is_string()) {
            setSubStreamId(params["sub_stream_id"].get<std::string>());
        }
        return true;
    }
};

} // namespace nodes
} // namespace ai_stream