// src/core/pipeline_manager.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "ai_stream/core/pipeline.h"
#include "nodes/registry/node_factory.h" // 内部工厂头文件
#include <nlohmann/json.hpp>

#include "ai_stream/nodes/i_decode_node.h"
#include "ai_stream/nodes/i_sink_node.h"
#include "ai_stream/nodes/i_source_node.h"
#include "ai_stream/nodes/i_infer_node.h"
#include "ai_stream/nodes/i_tracker_node.h"
#include "ai_stream/nodes/i_draw_node.h"
#include "ai_stream/nodes/i_preprocess_node.h"
#include "ai_stream/nodes/i_action_recognition_node.h"
#include "src/nodes/alert/alert_node.h"
#include "src/rules/alert/alert_rule_factory.h"

namespace ai_stream
{
    namespace core
    {

        Pipeline::~Pipeline()
        {
            stop();
        }

        bool Pipeline::buildFromJson(const nlohmann::json &config)
        {

            try
            {

                // 1. 解析节点配置并创建实例
                if (!config.contains("nodes") || !config["nodes"].is_array())
                {
                    LOG_ERROR_FMT("[Pipeline {}] Invalid config: missing 'nodes' array", id_);
                    return false;
                }

                for (const auto &node_cfg : config["nodes"])
                {
                    std::string id = node_cfg.value("id", "");
                    std::string type = node_cfg.value("type", "");
                    nlohmann::json params = node_cfg.value("params", nlohmann::json::object());

                    LOG_INFO_FMT("[Pipeline {}] Node: id:{},type:{}", id_, id, type);

                    if (id.empty() || type.empty())
                    {
                        LOG_ERROR_FMT("[Pipeline {}] Node missing 'id' or 'type'", id_);
                        return false;
                    }

                    auto node = NodeFactory::instance().create(type, params);
                    if (!node)
                    {
                        LOG_ERROR_FMT("[Pipeline {}] Failed to create node type: {}", id_, type);
                        return false;
                    }

                    if (type.find("source") != std::string::npos)
                    {
                        auto source_node = std::dynamic_pointer_cast<nodes::ISourceNode>(node);
                        source_node->setUrl(params["url"].get<std::string>());
                        source_node->setSourceId(id);
                        source_node->setSkipFrames(params["skip_frames"].get<int>());
                    }

                    if (type.find("decode") != std::string::npos)
                    {
                        auto decode_node = std::dynamic_pointer_cast<nodes::IDecodeNode>(node);
                        if (decode_node)
                        {
                            // 原有配置
                            if (params.contains("codec"))
                            {
                                decode_node->setDecoderType(params["codec"].get<std::string>());
                            }
                            if (params.contains("output_bgr"))
                            {
                                decode_node->setOutputBGR(params["output_bgr"].get<bool>());
                            }
                            if (params.contains("hw_decoder"))
                            {
                                decode_node->setHwDecodeEnabled(params["hw_decoder"].get<bool>());
                            }

                            // 快照配置
                            if (params.contains("snapshot"))
                            {
                                LOG_INFO_FMT("decode config:{}",params.dump());
                                auto &snapshot_cfg = params["snapshot"];

                                if (snapshot_cfg.contains("enabled"))
                                {
                                    decode_node->setSnapshotEnabled(snapshot_cfg["enabled"].get<bool>());
                                }
                                if (snapshot_cfg.contains("interval"))
                                {
                                    decode_node->setSnapshotInterval(snapshot_cfg["interval"].get<int>());
                                }
                                if (snapshot_cfg.contains("dir"))
                                {
                                    decode_node->setSnapshotDir(snapshot_cfg["dir"].get<std::string>());
                                }
                            }
                        }
                    }
                    
                    if (type.find("resize_normalize") != std::string::npos)
                    { 
                        auto preprocess_node = std::dynamic_pointer_cast<nodes::IPreprocessNode>(node);
                        if (params.contains("output_width") && params.contains("output_height"))
                        {
                            preprocess_node->setTargetSize(params["output_width"].get<int>(), params["output_height"].get<int>());
                        }
                        else
                        {
                            preprocess_node->setTargetSize(640, 640);
                        }
                        if (params.contains("keep_aspect_ratio"))
                        {
                            preprocess_node->setKeepAspectRatio(params["keep_aspect_ratio"].get<bool>());
                        }
                        if (params.contains("mean"))
                        {
                            preprocess_node->setMean(params["mean"].get<std::vector<float>>());
                        }
                        if (params.contains("std"))
                        {
                            preprocess_node->setStd(params["std"].get<std::vector<float>>());
                        }
                    }

                    if (type.find("infer") != std::string::npos)
                    {
                        auto infer_node = std::dynamic_pointer_cast<nodes::IInferNode>(node);
                        if (infer_node)
                        {
                            // 根据具体类型设置检测器类型
                            if (type == "detection_infer")
                            {
                                infer_node->setDetectorType(nodes::DetectorType::DETECTION);
                            }
                            else if (type == "segmentation_infer")
                            {
                                infer_node->setDetectorType(nodes::DetectorType::SEGMENTATION);
                            }
                            else if (type == "classification_infer")
                            {
                                infer_node->setDetectorType(nodes::DetectorType::CLASSIFICATION);
                            }
                            else if (type == "pose_infer")
                            {
                                infer_node->setDetectorType(nodes::DetectorType::POSE);
                            }
                            else
                            {
                                // 默认设置为检测器类型
                                infer_node->setDetectorType(nodes::DetectorType::DETECTION);
                            }

                            // 加载模型
                            if(params.contains("detector_config"))
                            {
                                nlohmann::json detector_config = params["detector_config"];
                                if (detector_config.contains("input_size"))
                                {
                                    auto input_size = detector_config["input_size"];
                                    if (input_size.contains("width") && input_size.contains("height"))
                                    {
                                        infer_node->setInputSize(input_size["width"].get<int>(), input_size["height"].get<int>());
                                    }
                                    else
                                    {
                                        infer_node->setInputSize(640, 640);
                                    }
                                }
                                if(detector_config.contains("batch_size"))
                                {
                                    infer_node->setBatchSize(detector_config["batch_size"].get<int>());
                                }
                                if (detector_config.contains("model_path"))
                                {
                                    std::string model_path = detector_config["model_path"].get<std::string>();
                                    if (!infer_node->loadModel(model_path))
                                    {
                                        LOG_ERROR_FMT("[Pipeline {}] Failed to load model: {}", id_, model_path);
                                        return false;
                                    }
                                }
                                
                            }

                        }
                    }

                    if (type.find("action_recognition") != std::string::npos)
                    {
                        auto action_node = std::dynamic_pointer_cast<nodes::IActionRecognitionNode>(node);
                        if (action_node)
                        {
                            if (params.contains("input_height") && params.contains("input_width"))
                            {
                                action_node->setInputSize(params["input_height"].get<int>(), params["input_width"].get<int>());
                            }
                            if (params.contains("num_frames") && params.contains("frame_interval"))
                            {
                                action_node->setClipParams(params["num_frames"].get<int>(), params["frame_interval"].get<int>());
                            }
                            if (params.contains("window_size") && params.contains("stride"))
                            {
                                action_node->setSlidingWindow(params["window_size"].get<int>(), params["stride"].get<int>());
                            }
                            if (params.contains("action_labels") && params["action_labels"].is_array())
                            {
                                std::vector<std::string> labels = params["action_labels"].get<std::vector<std::string>>();
                                action_node->setActionLabels(labels);
                            }
                            if (params.contains("confidence_threshold"))
                            {
                                action_node->setConfidenceThreshold(params["confidence_threshold"].get<float>());
                            }
                            if (params.contains("batch_size"))
                            {
                                action_node->setBatchSize(params["batch_size"].get<int>());
                            }
                            if (params.contains("model_path"))
                            {
                                action_node->setModelPath(params["model_path"].get<std::string>());
                            }
                        }
                    }

                    if (type.find("tracker") != std::string::npos) {
                        auto tracker_node = std::dynamic_pointer_cast<nodes::ITrackerNode>(node);
                        tracker_node->setTrackerId(id); // 设置跟踪器 ID，便于日志区分
                        if (tracker_node) {
                            // 设置跟踪器类型
                            if (params.contains("tracker_type")) {
                                std::string t = params["tracker_type"].get<std::string>();
                                if (t == "ocsort") {
                                    tracker_node->setTrackerType(nodes::TrackerType::OCSORT);
                                    
                                    nodes::OCSortConfig ocsort_config;
                                    if (params.contains("ocsort_config")) {
                                        auto& cfg = params["ocsort_config"];
                                        ocsort_config.det_thresh = cfg.value("det_thresh", 0.3f);
                                        ocsort_config.max_age = cfg.value("max_age", 30);
                                        ocsort_config.min_hits = cfg.value("min_hits", 3);
                                        ocsort_config.iou_threshold = cfg.value("iou_threshold", 0.3f);
                                        ocsort_config.delta_t = cfg.value("delta_t", 3);
                                        ocsort_config.asso_func = cfg.value("asso_func", "iou");
                                        ocsort_config.inertia = cfg.value("inertia", 0.2f);
                                        ocsort_config.use_byte = cfg.value("use_byte", false);
                                    }
                                    tracker_node->setOCSortConfig(ocsort_config);
                                    
                                } else if (t == "bytetrack") {
                                    tracker_node->setTrackerType(nodes::TrackerType::BYTETRACK);
                                    
                                    nodes::ByteTrackConfig bytetrack_config;
                                    if (params.contains("bytetrack_config")) {
                                        auto& cfg = params["bytetrack_config"];
                                        bytetrack_config.frame_rate = cfg.value("frame_rate", 30);
                                        bytetrack_config.track_buffer = cfg.value("track_buffer", 30);
                                        bytetrack_config.track_thresh = cfg.value("track_thresh", 0.5f);
                                        bytetrack_config.high_thresh = cfg.value("high_thresh", 0.6f);
                                        bytetrack_config.match_thresh = cfg.value("match_thresh", 0.8f);
                                    }
                                    tracker_node->setByteTrackConfig(bytetrack_config);
                                }
                            }
                            if (params.contains("sub_stream_id") && params["sub_stream_id"].is_string()) {
                                tracker_node->setSubStreamId(params["sub_stream_id"].get<std::string>());
                            }

                        }
                    }

                    if (type.find("draw") != std::string::npos)
                    { 
                        auto draw_node = std::dynamic_pointer_cast<nodes::IDrawNode>(node);
                        if(draw_node)
                        {
                            // 快照配置
                            if (params.contains("snapshot"))
                            {
                                LOG_INFO_FMT("draw config:{}",params.dump());
                                auto &snapshot_cfg = params["snapshot"];

                                if (snapshot_cfg.contains("enabled"))
                                {
                                    draw_node->setSnapshotEnabled(snapshot_cfg["enabled"].get<bool>());
                                }
                                if (snapshot_cfg.contains("interval"))
                                {
                                    draw_node->setSnapshotInterval(snapshot_cfg["interval"].get<int>());
                                }
                                if (snapshot_cfg.contains("dir"))
                                {
                                    draw_node->setSnapshotDir(snapshot_cfg["dir"].get<std::string>());
                                }
                                if (snapshot_cfg.contains("font_file"))
                                {
                                    draw_node->setFontFile(snapshot_cfg["font_file"].get<std::string>());
                                }
                                if (snapshot_cfg.contains("logo_file"))
                                {
                                    draw_node->setLogoFile(snapshot_cfg["logo_file"].get<std::string>());
                                }

                            }
                        }

                    }

                    if (type.find("alert") != std::string::npos)
                    { 
                        auto alert_node = std::dynamic_pointer_cast<nodes::AlertNode>(node);
                        if (alert_node)
                        {
                            if (params.contains("rules") && params["rules"].is_array())
                            {
                                for (const auto& rule_cfg : params["rules"])
                                {
                                    std::string alert_type = rule_cfg.value("type", "");
                                    if(alert_type.empty())
                                    {
                                        LOG_ERROR_FMT("[Pipeline {}] Alert rule missing 'type'", id_);
                                        continue;
                                    }
                                    try
                                    {
                                        auto rule = rules::AlertRuleFactory::instance().create(alert_type);
                                        if(rule == nullptr)
                                        {
                                            LOG_ERROR_FMT("[Pipeline {}] No factory found for alert rule type: {}", id_, alert_type);
                                            continue;
                                        }
                                        
                                        if (rule)
                                        {
                                            rule->initialize(rule_cfg.value("params", nlohmann::json::object()));
                                            alert_node->addRule(rule);
                                        }
                                        else
                                        {
                                            LOG_ERROR_FMT("[Pipeline {}] Failed to create alert rule: {}", id_, alert_type);
                                        }
                                        
                                    }
                                    catch(const std::exception& e)
                                    {
                                        LOG_ERROR_FMT("[Pipeline {}] Exception while creating alert rule: {}", id_, e.what());
                                    }
                                }
                            }
                        }
                    }

                    if (type.find("sink") != std::string::npos || type.find("save") != std::string::npos)
                    {
                        auto sink_node = std::dynamic_pointer_cast<nodes::ISinkNode>(node);
                        if (!sink_node)
                        {
                            continue;
                        }
                        sink_node->setTarget(params["output_url"].get<std::string>());
                        sink_node->setOutputSize(params["output_width"].get<int>(), params["output_height"].get<int>());
                    }

                    // 设置节点名称（覆盖工厂默认名）
                    // 由于 Node 构造函数已设置名称，可在此处通过 setter 修改（如有需要）
                    // 简单起见，我们假设节点内部已处理好
                    nodes_[id] = node;
                    LOG_INFO_FMT("[Pipeline {}] Created node: {} (type: {})", id_, id, type);
                }

                // 2. 建立边连接（一流多用的关键）
                if (config.contains("edges") && config["edges"].is_array())
                {
                    for (const auto &edge : config["edges"])
                    {
                        std::string from = edge.value("from", "");
                        std::string to = edge.value("to", "");

                        auto it_from = nodes_.find(from);
                        auto it_to = nodes_.find(to);
                        if (it_from == nodes_.end() || it_to == nodes_.end())
                        {
                            LOG_ERROR_FMT("[Pipeline {}] Invalid edge: {} -> {}", id_, from, to);
                            return false;
                        }

                        it_from->second->addDownstream(it_to->second);
                        LOG_INFO_FMT("[Pipeline {}] Connected: {} -> {}", id_, from, to);
                    }
                }

                // 3. 为每个节点设置管道弱引用（便于节点访问全局上下文）
                auto weak_self = weak_from_this();
                for (auto &[_, node] : nodes_)
                {
                    node->setPipeline(weak_self);
                }

                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR_FMT("[Pipeline {}] Exception during build: {}", id_, e.what());
                return false;
            }
        }

        bool Pipeline::start()
        {
            if (running_)
                return true;

            // 启动顺序：源节点先启动，其他节点无所谓（因为它们是被动接收数据的）
            // 但为了简化，统一调用 start()
            bool all_started = true;
            for (auto &[name, node] : nodes_)
            {
                LOG_INFO_FMT("[Pipeline {}] Starting node: {}", id_, name);
                if (!node->start())
                {
                    LOG_ERROR_FMT("[Pipeline {}] Failed to start node: {}", id_, name);
                    all_started = false;
                    break;
                }
            }

            if (all_started)
            {
                running_ = true;
                LOG_INFO_FMT("[Pipeline {}] All nodes started successfully", id_);
            }
            else
            {
                // 回滚已启动的节点
                stop();
            }
            return all_started;
        }

        void Pipeline::stop()
        {
            if (!running_)
                return;
            running_ = false;

            // 逆序停止节点，通常先停源节点以防止新数据进入，但这里简化统一停止
            for (auto &[name, node] : nodes_)
            {
                LOG_INFO_FMT("[Pipeline {}] Stopping node: {}", id_, name);
                node->stop();
            }
            LOG_INFO_FMT("[Pipeline {}] Pipeline stopped", id_);
        }

        std::shared_ptr<Node> Pipeline::getNode(const std::string &name)
        {
            auto it = nodes_.find(name);
            if (it != nodes_.end())
            {
                return it->second;
            }
            return nullptr;
        }

        bool Pipeline::isRunning() const{
            std::lock_guard<std::mutex> lock(mutex_);
            for(const auto& node : nodes_)
            {
                if(node.second->isRunning())
                {
                    //LOG_INFO_FMT("node: {} is running",node.first);
                    return true;
                }
                //LOG_INFO_FMT("node: {} is not running",node.first);
            }
            LOG_INFO_FMT("Pipeline {} is not running", id_);
            return false;
        }

    } // namespace core
} // namespace ai_stream