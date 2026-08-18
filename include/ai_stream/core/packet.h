// include/ai_stream/core/packet.h
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <opencv2/core/mat.hpp>
#include <nlohmann/json.hpp>

namespace ai_stream {

namespace rules
{
    enum class RuleStatus : uint8_t
    {
        RULE_STATUS_OK = 0,
        RULE_STATUS_FAIL,
        RULE_STATUS_NOT_INITIALIZED,
        RULE_STATUS_NOT_SUPPORTED
    };

    /**
     * @brief 告警级别
     */
    enum class AlertLevel
    {
        INFO = 0,
        WARNING = 1,
        ERROR = 2,
        CRITICAL = 3
    };

    enum class AlertType
    {
        ALERT_UNKNOWN = 0,
        PERSON_INTRUSION = 1,
        MISSING_HELMET = 2,
        MISSING_WORK_CLOTHES = 3,
        PHONE_CALL = 4,
        SMOKING = 5,
        FALL_DOWN = 6,
        MISSING_SAFETY_BELT = 7,
        HUMAN_GATHERING = 8,
        ABSENCE = 9,
        SLEEPING_ON_DUTY = 10,
        CLIMBING = 11,
        FIGHTING = 12,
        UNLICENSED_VENDOR = 13,
        PHOTOGRAPHER = 14,
        FIRE_LANE_OCCUPANCY = 15,
        DISCOVER_CRYSTAL = 16,
        DISCOVER_VISIBLE_FIRE = 17,
        DISCOVER_SMOKE = 18,
        DISCOVER_HOSE_CUTOFF = 19,
        ACTION_RECOGNITION = 100
    };

    inline std::map<AlertType, std::string> alertTypeMap = {
        {AlertType::ALERT_UNKNOWN, "alert_unknown"},
        {AlertType::PERSON_INTRUSION, "person_intrusion"},
        {AlertType::MISSING_HELMET, "missing_helmet"},
        {AlertType::MISSING_WORK_CLOTHES, "missing_work_clothes"},
        {AlertType::PHONE_CALL, "phone_call"},
        {AlertType::SMOKING, "smoking"},
        {AlertType::FALL_DOWN, "fall_down"},
        {AlertType::MISSING_SAFETY_BELT, "missing_safety_belt"},
        {AlertType::HUMAN_GATHERING, "human_gathering"},
        {AlertType::ABSENCE, "absence"},
        {AlertType::SLEEPING_ON_DUTY, "sleeping_on_duty"},
        {AlertType::CLIMBING, "climbing"},
        {AlertType::FIGHTING, "fighting"},
        {AlertType::UNLICENSED_VENDOR, "unlicensed_vendor"},
        {AlertType::PHOTOGRAPHER, "photographer"},
        {AlertType::FIRE_LANE_OCCUPANCY, "fire_lane_occupancy"},
        {AlertType::DISCOVER_CRYSTAL, "discover_crystal"},
        {AlertType::DISCOVER_VISIBLE_FIRE, "discover_visible_fire"},
        {AlertType::DISCOVER_SMOKE, "discover_smoke"},
        {AlertType::DISCOVER_HOSE_CUTOFF, "discover_hose_cutoff"},
        {AlertType::ACTION_RECOGNITION, "action_recognition"}
    };

    inline std::map<AlertType, std::string> alertTypeChMap = {
        {AlertType::ALERT_UNKNOWN, "未知告警"},
        {AlertType::PERSON_INTRUSION, "人员入侵"},
        {AlertType::MISSING_HELMET, "未戴安全帽"},
        {AlertType::MISSING_WORK_CLOTHES, "未穿工作服"},
        {AlertType::PHONE_CALL, "打电话"},
        {AlertType::SMOKING, "吸烟"},
        {AlertType::FALL_DOWN, "跌倒"},
        {AlertType::MISSING_SAFETY_BELT, "未戴安全带"},
        {AlertType::HUMAN_GATHERING, "人员聚集"},
        {AlertType::ABSENCE, "离岗"},
        {AlertType::SLEEPING_ON_DUTY, "睡岗"},
        {AlertType::CLIMBING, "攀爬"},
        {AlertType::FIGHTING, "打架"},
        {AlertType::UNLICENSED_VENDOR, "无证摊贩"},
        {AlertType::PHOTOGRAPHER, "揽拍"},
        {AlertType::FIRE_LANE_OCCUPANCY, "消防通道占用"},
        {AlertType::DISCOVER_CRYSTAL, "发现结晶"},
        {AlertType::DISCOVER_VISIBLE_FIRE, "发现火焰"},
        {AlertType::DISCOVER_SMOKE, "发现烟雾"},
        {AlertType::DISCOVER_HOSE_CUTOFF, "发现软管断流"},
        {AlertType::ACTION_RECOGNITION, "动作识别"}
    };

    enum class ActionRecongnitionType : uint8_t
    {
        ACTION_RECOGNITION_UNKNOWN = 0,
        ACTION_RECOGNITION_POSE = 1, //使用姿态识别
        ACTION_RECOGNITION_MODEL = 2,//使用模型识别
    };

    enum class AlertItemType : uint8_t
    {
        ITEM_MASTER_UNKNOWN = 0,
        ITEM_PERSON_BEHAVIOR = 1,
        ITEM_SAFETY_ITEM = 2,
        ITEM_SCENE_RECOGNITION = 3
    };

    inline std::map<AlertType, AlertItemType> alertItemTypeMap = {
        {AlertType::ALERT_UNKNOWN, AlertItemType::ITEM_MASTER_UNKNOWN},
        {AlertType::PERSON_INTRUSION, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::MISSING_HELMET, AlertItemType::ITEM_SAFETY_ITEM},
        {AlertType::MISSING_WORK_CLOTHES, AlertItemType::ITEM_SAFETY_ITEM},
        {AlertType::PHONE_CALL, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::SMOKING, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::FALL_DOWN, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::MISSING_SAFETY_BELT, AlertItemType::ITEM_SAFETY_ITEM},
        {AlertType::HUMAN_GATHERING, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::ABSENCE, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::SLEEPING_ON_DUTY, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::CLIMBING, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::FIGHTING, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::UNLICENSED_VENDOR, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::PHOTOGRAPHER, AlertItemType::ITEM_PERSON_BEHAVIOR},
        {AlertType::FIRE_LANE_OCCUPANCY, AlertItemType::ITEM_SCENE_RECOGNITION},
        {AlertType::DISCOVER_CRYSTAL, AlertItemType::ITEM_SCENE_RECOGNITION},
        {AlertType::DISCOVER_VISIBLE_FIRE, AlertItemType::ITEM_SCENE_RECOGNITION},
        {AlertType::DISCOVER_SMOKE, AlertItemType::ITEM_SCENE_RECOGNITION},
        {AlertType::DISCOVER_HOSE_CUTOFF, AlertItemType::ITEM_SCENE_RECOGNITION},
        {AlertType::ACTION_RECOGNITION, AlertItemType::ITEM_PERSON_BEHAVIOR}
    };

    enum class AlertStatus : uint8_t
    {
        ALERT_STATUS_OCCUR = 0,
        ALERT_STATUS_LAST = 1,
        ALERT_STATUS_END = 2,
        ALERT_STATUS_DEFAULT = 3
    };

    inline std::map<AlertStatus, std::string> alert_status_map =
    {
        {AlertStatus::ALERT_STATUS_OCCUR, " occur"},
        {AlertStatus::ALERT_STATUS_LAST, " last"},
        {AlertStatus::ALERT_STATUS_END, " end"},
        {AlertStatus::ALERT_STATUS_DEFAULT, " default"}
    };

    /**
     * @brief 告警事件
     */
    struct AlertEvent
    {
        std::string alert_id;
        int64_t detect_ms;
        int64_t duration_ms;
        uint16_t non_update_count;
        uint8_t zone_no;
        std::vector<int> object_ids;

        AlertLevel level;
        std::string alert_name;
        AlertItemType alert_item_type = AlertItemType::ITEM_MASTER_UNKNOWN;
        AlertType alert_type = AlertType::ALERT_UNKNOWN;
        std::string description;
        AlertStatus status;
        nlohmann::json extra_data;
        
        nlohmann::json toJson() const
        {
            static const std::map<AlertLevel, std::string> level_names = {
                {AlertLevel::INFO, "info"},
                {AlertLevel::WARNING, "warning"},
                {AlertLevel::ERROR, "error"},
                {AlertLevel::CRITICAL, "critical"}
            };
            static const std::map<AlertStatus, std::string> status_names = {
                {AlertStatus::ALERT_STATUS_OCCUR, "occur"},
                {AlertStatus::ALERT_STATUS_LAST, "last"},
                {AlertStatus::ALERT_STATUS_END, "end"},
                {AlertStatus::ALERT_STATUS_DEFAULT, "default"}
            };

            nlohmann::json j;
            j["alert_id"] = alert_id;
            j["alert_name"] = alert_name;
            j["alert_type"] = static_cast<int>(alert_type);
            auto type_it = alertTypeMap.find(alert_type);
            if (type_it != alertTypeMap.end()) {
                j["alert_type_name"] = type_it->second;
            }
            j["alert_item_type"] = static_cast<int>(alert_item_type);
            j["level"] = static_cast<int>(level);
            auto level_it = level_names.find(level);
            if (level_it != level_names.end()) {
                j["level_name"] = level_it->second;
            }
            j["status"] = static_cast<int>(status);
            auto status_it = status_names.find(status);
            if (status_it != status_names.end()) {
                j["status_name"] = status_it->second;
            }
            j["description"] = description;
            j["detect_ms"] = detect_ms;
            j["duration_ms"] = duration_ms;
            j["non_update_count"] = non_update_count;
            j["zone_no"] = zone_no;
            j["object_ids"] = object_ids;
            if (!extra_data.is_null()) {
                j["extra_data"] = extra_data;
            }
            return j;
        }
        AlertEvent() : detect_ms(0), duration_ms(0), non_update_count(0), zone_no(0), level(AlertLevel::INFO), status(AlertStatus::ALERT_STATUS_DEFAULT) {}
    };

    struct AlertResult
    {
        uint32_t stream_id;
        AlertType alert_type;
        uint8_t alert_count;
        std::vector<AlertEvent> alert_events;
        
        std::string rule_name;
        RuleStatus rule_status;
        uint64_t process_time_ms;
        std::string error_message;
        AlertResult() : stream_id(0), alert_type(AlertType::ALERT_UNKNOWN), alert_count(0), rule_status(RuleStatus::RULE_STATUS_NOT_INITIALIZED),
            alert_events(){}
    };
}
namespace core {

// ========== COCO骨架定义 ==========
static const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},           // 脸部
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // 上肢
    {5, 11}, {6, 12}, {11, 12},              // 躯干
    {11, 13}, {13, 15}, {12, 14}, {14, 16}  // 下肢
};

static const cv::Scalar COLOR_HEAD(0, 255, 0);      // 绿
static const cv::Scalar COLOR_ARMS(255, 0, 0);      // 蓝
static const cv::Scalar COLOR_BODY(0, 165, 255);    // 橙
static const cv::Scalar COLOR_LEGS(0, 0, 255);      // 红

static const cv::Scalar KPT_COLORS[17] = {
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),     // 0-4 脸部绿
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),     // 5-6 肩蓝
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),     // 7-8 肘蓝
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),     // 9-10 手腕蓝
    cv::Scalar(0, 165, 255), cv::Scalar(0, 165, 255), // 11-12 髋橙
    cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255),     // 13-14 膝红
    cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255)      // 15-16 踝红
};


/**
 * @brief 数据包类型枚举
 */
enum class PacketType : uint8_t {
    UNKNOWN = 0,
    RAW_VIDEO,      // 原始编码视频数据 (H264/H265 字节流)
    DECODED_FRAME,  // 解码后的图像帧 (YUV/RGB)
    DETECTION,
    POSE,
    META_DATA,      // 推理结果等元数据
    TENSOR,          // 推理输入/输出 Tensor
    STREAM_END,
};

/**
 * @brief 基础数据包结构
 * 
 * 所有在管道中流动的数据都继承自此结构。
 * 使用 std::dynamic_pointer_cast 进行安全的类型转换。
 */
struct BasePacket {
    PacketType type = PacketType::UNKNOWN;
    int64_t timestamp_ms = 0;          // 毫秒时间戳
    uint32_t stream_id = 0;            // 多流并行的关键：用于区分不同的源流
    std::string source_id;             // 可选的源标识符
    std::string producer_id;           // 最近一次 broadcast 该包的节点名（用于下游区分数据来源）
    int64_t frame_id = 0;
    uint64_t cost_ms = 0;
    std::map<std::string,uint64_t> cost_time_map;

    virtual ~BasePacket() = default;
};

/**
 * @brief 原始视频包（编码数据）
 * 
 * 例如从 RTSP 读出的 H264/H265 NALU 单元
 */
struct RawVideoPacket : public BasePacket {
    RawVideoPacket() { type = PacketType::RAW_VIDEO; }

    std::vector<uint8_t> data;          // 编码数据缓冲区
    bool is_key_frame = false;          // 是否为关键帧
    int codec_id = 0;                   // 编码格式 ID（如 AV_CODEC_ID_H264）
    std::vector<uint8_t> extradata;     // 编码器 extradata (SPS/PPS/VPS)
};

/**
 * @brief 解码后的视频帧包
 * 
 * 使用 shared_ptr 管理 cv::Mat，避免深拷贝
 */
struct VideoFramePacket : public BasePacket {
    VideoFramePacket() { type = PacketType::DECODED_FRAME; }
    
    std::shared_ptr<cv::Mat> source_mat;
    std::shared_ptr<cv::Mat> mat;       // 图像数据（通常为 BGR 格式）

    void* d_ptr = nullptr;
    int d_width = 0;
    int d_height = 0;
    int d_pitch = 0;
    bool is_gpu = false;

    int width = 0;
    int height = 0;
    int channels = 3;

    // 用于后续osd gpu 绘制
    void* d_bgr_ptr = nullptr;
    int d_bgr_pitch = 0;
    int d_bgr_height = 0;
    int d_bgr_width = 0;

    // Letterbox 参数（用于坐标反变换）
    bool letterbox_used = false;  // 是否使用了 letterbox resize
    float letter_scale = 1.0f;    // letterbox 缩放比例 (letter_w / source_w)
    int letter_pad_x = 0;         // 水平 padding
    int letter_pad_y = 0;         // 垂直 padding
};

/**
 * @brief 推理结果元数据包
 * 
 * 包含检测框、分类结果等信息
 */
struct InferenceResultPacket : public BasePacket {
    InferenceResultPacket() { type = PacketType::META_DATA; }
    struct KeyPoint {
        float x;
        float y;
        float confidence;
        bool visible;
        KeyPoint() : x(0.0f), y(0.0f), confidence(0.0f), visible(false) {}
        KeyPoint(float x_, float y_, float conf_, bool visible_)
            : x(x_), y(y_), confidence(conf_), visible(visible_) {}
    };
    /**
     * @brief 边界框结构
     */
    struct BBox {
        float x, y, w, h;               // 边界框坐标和尺寸
        float confidence;               // 置信度 [0.0, 1.0]
        int class_id;                   // 类别 ID
        std::string class_name;         // 类别名称
        bool has_keypoints;
        std::array<KeyPoint, 17> keypoints;
        float keypoints_conf;

        int track_id;
        int track_age;
        bool track_active;
        float smooth_x;
        float smooth_y;
        float smooth_w;
        float smooth_h;
        
        BBox() = default;
        BBox(float x_, float y_, float w_, float h_, float conf, int cls_id, const std::string& cls_name = "")
            : x(x_), y(y_), w(w_), h(h_), confidence(conf), class_id(cls_id), class_name(cls_name),
                track_id(-1), track_age(0), track_active(false), smooth_x(0.0f), smooth_y(0.0f), smooth_w(0.0f), smooth_h(0.0f) {}
    };

    struct PoseResult{
        float person_score;
        std::array<KeyPoint, 17> keypoints;
        cv::Rect2f person_box;
        int matched_det_idx;
        PoseResult() : person_score(0.0f), matched_det_idx(-1) {}
    };

    /**
     * @brief 动作识别结果结构
     */
    struct ActionResult {
        int track_id = -1;                    // 关联的目标ID
        std::string action_label;             // 动作标签
        float confidence = 0.0f;              // 置信度
        int64_t timestamp_ms = 0;             // 时间戳
        std::vector<float> action_scores;     // 所有类别的分数
    };

    std::vector<BBox> detections;                       // 检测结果列表
    std::vector<PoseResult> pose_results;
    std::vector<ActionResult> action_results;           // 动作识别结果
    std::shared_ptr<VideoFramePacket> source_frame;     // 关联的原始帧，用于后续画框等操作
    std::vector<rules::AlertResult> alert_result;
};

} // namespace core
} // namespace ai_stream