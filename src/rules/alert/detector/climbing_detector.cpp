// src/rules/alert/detector/climbing_detector.cpp

#include "3rd_party/log_mgr/log_mgr.h"
#include "climbing_detector.h"
#include <iostream>
#include <unordered_set>
#include <numeric>
#include <fstream>

// ========== StateMachine ==========

void ClimbingDetector::StateMachine::reset() {
    current_state = ClimbingState::IDLE;
    previous_state = ClimbingState::IDLE;
    state_duration = 0;
    total_climbing_frames = 0;
    suspicious_count = 0;
    normal_count = 0;
    climbing_count = 0;
    cooldown_count = 0;
}

std::string ClimbingDetector::StateMachine::getStateString() const {
    switch (current_state) {
        case ClimbingState::IDLE:       return "IDLE";
        case ClimbingState::SUSPICIOUS: return "SUSPICIOUS";
        case ClimbingState::CLIMBING:   return "CLIMBING";
        case ClimbingState::COOLDOWN:   return "COOLDOWN";
        default: return "UNKNOWN";
    }
}

void ClimbingDetector::StateMachine::update(StateTransitionEvent event, const Config& cfg) {
    previous_state = current_state;
    state_duration++;

    switch (current_state) {
        case ClimbingState::IDLE: {
            if (event == StateTransitionEvent::INDIVIDUAL_SUSPICIOUS) {
                suspicious_count++;
                normal_count = 0;
            } else if (event == StateTransitionEvent::MULTI_PERSON_SUSPICIOUS) {
                suspicious_count += 2;
                normal_count = 0;
            } else if (event == StateTransitionEvent::CONFIRMED_CLIMBING) {
                suspicious_count += 3;
                normal_count = 0;
            } else {
                suspicious_count = std::max(0, suspicious_count - 1);
                normal_count++;
            }

            if (suspicious_count >= cfg.suspicious_enter_threshold) {
                current_state = ClimbingState::SUSPICIOUS;
                state_duration = 0;
                suspicious_count = 0;
                LOG_INFO("[攀爬] 状态转换: 空闲 -> 可疑");
            }
            break;
        }

        case ClimbingState::SUSPICIOUS: {
            if (event == StateTransitionEvent::CONFIRMED_CLIMBING) {
                climbing_count++;
                normal_count = 0;
                if (climbing_count >= cfg.climbing_enter_threshold) {
                    current_state = ClimbingState::CLIMBING;
                    state_duration = 0;
                    climbing_count = 0;
                    LOG_INFO("[攀爬] 状态转换: 可疑 -> 攀爬");
                }
            } else if (event == StateTransitionEvent::MULTI_PERSON_SUSPICIOUS) {
                climbing_count += 2;
                normal_count = 0;
                if (climbing_count >= cfg.climbing_enter_threshold) {
                    current_state = ClimbingState::CLIMBING;
                    state_duration = 0;
                    climbing_count = 0;
                    LOG_INFO("[攀爬] 状态转换: 可疑 -> 攀爬 (加速)");
                }
            } else if (event == StateTransitionEvent::NO_EVENT) {
                normal_count++;
                climbing_count = std::max(0, climbing_count - 1);
                if (normal_count >= cfg.suspicious_exit_threshold) {
                    current_state = ClimbingState::IDLE;
                    state_duration = 0;
                    normal_count = 0;
                    LOG_INFO("[攀爬] 状态转换: 可疑 -> 空闲");
                }
            } else {
                normal_count = std::max(0, normal_count - 1);
            }
            break;
        }

        case ClimbingState::CLIMBING: {
            total_climbing_frames++;

            if (event == StateTransitionEvent::NO_EVENT ||
                event == StateTransitionEvent::INDIVIDUAL_SUSPICIOUS) {
                normal_count++;
                if (normal_count >= cfg.climbing_exit_threshold) {
                    current_state = ClimbingState::COOLDOWN;
                    state_duration = 0;
                    normal_count = 0;
                    cooldown_count = 0;
                    LOG_INFO_FMT("[攀爬] 状态转换: 攀爬 -> 冷却 (总攀爬帧数={})",
                                 total_climbing_frames);
                }
            } else {
                normal_count = std::max(0, normal_count - 2);
            }
            break;
        }

        case ClimbingState::COOLDOWN: {
            cooldown_count++;
            if (event == StateTransitionEvent::CONFIRMED_CLIMBING) {
                current_state = ClimbingState::CLIMBING;
                state_duration = 0;
                cooldown_count = 0;
                normal_count = 0;
                LOG_INFO("[攀爬] 状态转换: 冷却 -> 攀爬 (再次触发)");
            } else if (cooldown_count >= cfg.cooldown_frames) {
                current_state = ClimbingState::IDLE;
                state_duration = 0;
                cooldown_count = 0;
                total_climbing_frames = 0;
                LOG_INFO("[攀爬] 状态转换: 冷却 -> 空闲");
            }
            break;
        }
    }
}

// ========== TrackState ==========

void ClimbingDetector::TrackState::reset() {
    center_history.clear();
    left_wrist_y_history.clear();
    right_wrist_y_history.clear();
    left_wrist_x_history.clear();
    right_wrist_x_history.clear();
    left_ankle_y_history.clear();
    right_ankle_y_history.clear();
    avg_center_y = -1.0f;
    posture_frame_count = 0;
    ascent_frame_count = 0;
    last_bbox = cv::Rect2f();
    feature_vector.clear();
    feature_valid = false;
}

// ========== FrameAnalysis ==========

void ClimbingDetector::FrameAnalysis::clear() {
    individual_scores.clear();
    individual_events.clear();
    is_suspicious.clear();
    suspicious_person_count = 0;
    total_score = 0.0f;
    has_ascent = false;
}

// ========== 构造 ==========

ClimbingDetector::ClimbingDetector(const Config& cfg) : cfg_(cfg), frame_counter_(0) {
    climb_history_.clear();
    if (cfg_.use_ml_score && !cfg_.ml_model_path.empty()) {
        svm_loaded_ = svm_model_.load(cfg_.ml_model_path);
        if (svm_loaded_) {
            LOG_INFO_FMT("[攀爬] SVM模型已加载: {}, 特征维度={}",
                         cfg_.ml_model_path, svm_model_.feature_dim);
            if (!cfg_.ml_scaler_path.empty()) {
                bool scaler_ok = svm_model_.loadScaler(cfg_.ml_scaler_path);
                LOG_INFO_FMT("[攀爬] SVM归一化参数加载: {}", scaler_ok ? "成功" : "失败");
            }
        } else {
            LOG_WARN_FMT("[攀爬] SVM模型加载失败: {}", cfg_.ml_model_path);
        }
    }
}

void ClimbingDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tracks_.clear();
    climb_history_.clear();
    state_machine_.reset();
    frame_counter_ = 0;
}

// ========== 状态查询 ==========

ClimbingState ClimbingDetector::getCurrentState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.current_state;
}

std::string ClimbingDetector::getCurrentStateString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.getStateString();
}

int ClimbingDetector::getStateDuration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.state_duration;
}

int ClimbingDetector::getTotalClimbingFrames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.total_climbing_frames;
}

// ========== 状态回调 ==========

void ClimbingDetector::onEnterIdle() {
    if (state_machine_.previous_state != ClimbingState::IDLE) {
        LOG_INFO("[攀爬] 进入空闲状态");
    }
}

void ClimbingDetector::onEnterSuspicious() {
    LOG_INFO("[攀爬] 进入可疑状态");
}

void ClimbingDetector::onEnterClimbing() {
    LOG_INFO("[攀爬] 进入攀爬状态 - 检测到攀爬!");
}

void ClimbingDetector::onEnterCooldown() {
    LOG_INFO("[攀爬] 进入冷却状态");
}

void ClimbingDetector::onStateTick(ClimbingState state) {
    (void)state;
}

// ========== 主处理 ==========

ClimbingResult ClimbingDetector::process(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    std::lock_guard<std::mutex> lock(mutex_);
    frame_counter_++;

    ClimbingResult result;

    std::vector<int> active_ids;
    for (const auto& det : detections) {
        if (det.track_id >= 0) {
            active_ids.push_back(det.track_id);
        }
    }
    result.active_track_ids = active_ids;

    if (detections.empty()) {
        cleanupInactive(active_ids);
        climb_history_.push_back(false);
        if (climb_history_.size() > MAX_HISTORY) climb_history_.pop_front();

        state_machine_.update(StateTransitionEvent::NO_EVENT, cfg_);
        result.is_climbing = (state_machine_.current_state == ClimbingState::CLIMBING);
        result.current_state = state_machine_.getStateString();
        result.climbing_duration_frames = state_machine_.state_duration;
        result.total_climbing_frames = state_machine_.total_climbing_frames;

        onStateTick(state_machine_.current_state);
        return result;
    }

    FrameAnalysis analysis = analyzeFrame(detections);
    StateTransitionEvent transition_event = determineTransitionEvent(analysis);
    state_machine_.update(transition_event, cfg_);

    if (state_machine_.previous_state != state_machine_.current_state) {
        switch (state_machine_.current_state) {
            case ClimbingState::IDLE:       onEnterIdle(); break;
            case ClimbingState::SUSPICIOUS: onEnterSuspicious(); break;
            case ClimbingState::CLIMBING:   onEnterClimbing(); break;
            case ClimbingState::COOLDOWN:   onEnterCooldown(); break;
        }
    }

    result = buildResult(analysis, active_ids);
    result.current_state = state_machine_.getStateString();
    result.climbing_duration_frames = state_machine_.state_duration;
    result.total_climbing_frames = state_machine_.total_climbing_frames;
    result.is_climbing = (state_machine_.current_state == ClimbingState::CLIMBING);

    climb_history_.push_back(result.is_climbing);
    if (climb_history_.size() > MAX_HISTORY) climb_history_.pop_front();

    cleanupInactive(active_ids);
    onStateTick(state_machine_.current_state);

    if (!cfg_.training_data_path.empty() && !active_ids.empty()) {
        exportTrainingData(tracks_, analysis, frame_counter_);
    }

    LOG_INFO_FMT("[攀爬] 第{}帧: 状态={}, 持续帧数={}, 事件={}, 是否攀爬={}",
                 frame_counter_,
                 state_machine_.getStateString(),
                 state_machine_.state_duration,
                 static_cast<int>(transition_event),
                 result.is_climbing);

    return result;
}

// ========== 帧级分析 ==========

ClimbingDetector::FrameAnalysis ClimbingDetector::analyzeFrame(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    FrameAnalysis analysis;

    for (const auto& det : detections) {
        if (det.track_id < 0) continue;
        if (!det.has_keypoints) continue;

        auto& state = tracks_[det.track_id];
        state.last_bbox = cv::Rect2f(det.x, det.y, det.w, det.h);

        cv::Point2f center(det.x + det.w / 2, det.y + det.h / 2);
        state.center_history.push_back(center);
        while (state.center_history.size() > static_cast<size_t>(cfg_.history_frames)) {
            state.center_history.pop_front();
        }

        const auto& kpts = det.keypoints;
        if (isKeypointValid(kpts[Config::LEFT_WRIST])) {
            state.left_wrist_y_history.push_back(kpts[Config::LEFT_WRIST].y);
            state.left_wrist_x_history.push_back(kpts[Config::LEFT_WRIST].x);
            while (state.left_wrist_y_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.left_wrist_y_history.pop_front();
            while (state.left_wrist_x_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.left_wrist_x_history.pop_front();
        }
        if (isKeypointValid(kpts[Config::RIGHT_WRIST])) {
            state.right_wrist_y_history.push_back(kpts[Config::RIGHT_WRIST].y);
            state.right_wrist_x_history.push_back(kpts[Config::RIGHT_WRIST].x);
            while (state.right_wrist_y_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.right_wrist_y_history.pop_front();
            while (state.right_wrist_x_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.right_wrist_x_history.pop_front();
        }
        if (isKeypointValid(kpts[Config::LEFT_ANKLE])) {
            state.left_ankle_y_history.push_back(kpts[Config::LEFT_ANKLE].y);
            while (state.left_ankle_y_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.left_ankle_y_history.pop_front();
        }
        if (isKeypointValid(kpts[Config::RIGHT_ANKLE])) {
            state.right_ankle_y_history.push_back(kpts[Config::RIGHT_ANKLE].y);
            while (state.right_ankle_y_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.right_ankle_y_history.pop_front();
        }

        float score = 0.0f;

        // ===== 静态特征（单帧） =====
        float hand_conf = 0.0f, arm_conf = 0.0f, knee_conf = 0.0f;
        float center_conf = 0.0f, tilt_conf = 0.0f, limb_conf = 0.0f;

        float conf = 0.0f;

        if (detectHandAboveShoulder(det, conf)) {
            hand_conf = conf;
            score += conf * 0.20f;
            analysis.individual_events[det.track_id].push_back(
                {"hand_above_shoulder", det.track_id, conf, {}, "hand above shoulder", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 手高于肩 置信度={:.2f} 累计分={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectArmBend(det, conf)) {
            arm_conf = conf;
            score += conf * 0.20f;
            analysis.individual_events[det.track_id].push_back(
                {"arm_bend", det.track_id, conf, {}, "arm bent", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 手臂弯曲 置信度={:.2f} 累计分={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectKneeRaise(det, conf)) {
            knee_conf = conf;
            score += conf * 0.20f;
            analysis.individual_events[det.track_id].push_back(
                {"knee_raise", det.track_id, conf, {}, "knee raised", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 膝盖抬起 置信度={:.2f} 累计分={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectCenterRaise(det, state, conf)) {
            center_conf = conf;
            score += conf * 0.15f;
            analysis.individual_events[det.track_id].push_back(
                {"center_raise", det.track_id, conf, {}, "center raised or compressed", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 重心抬高/蜷缩 置信度={:.2f} 累计分={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectBodyTilt(det, conf)) {
            tilt_conf = conf;
            score += conf * 0.15f;
            analysis.individual_events[det.track_id].push_back(
                {"body_tilt", det.track_id, conf, {}, "body tilted", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 身体倾斜 置信度={:.2f} 累计分={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectLimbSpan(det, conf)) {
            limb_conf = conf;
            score += conf * 0.10f;
            analysis.individual_events[det.track_id].push_back(
                {"limb_span", det.track_id, conf, {}, "limbs spread", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 四肢张开 置信度={:.2f} 累计分={:.2f}",
                         det.track_id, conf, score);
        }

        // ===== 动态特征（多帧） =====
        float dyn_conf = 0.0f;

        bool has_alternation = detectAlternatingLimb(det, state, dyn_conf);
        float alt_conf = 0.0f;
        bool has_ascent = false;

        float dynamic_score = 0.0f;
        if (has_alternation) {
            detectAlternatingLimb(det, state, alt_conf);
            dynamic_score += alt_conf * 0.50f;
            analysis.individual_events[det.track_id].push_back(
                {"alternating_limb", det.track_id, alt_conf, {}, "alternating limb movement", frame_counter_});
            LOG_INFO_FMT("[攀爬] 目标{} 交替抬手抬脚 置信度={:.2f}", det.track_id, alt_conf);
        }

        float ascent_conf = 0.0f;
        has_ascent = detectOverallAscent(det, state, ascent_conf);
        if (has_ascent) {
            dynamic_score += ascent_conf * 0.50f;
            analysis.individual_events[det.track_id].push_back(
                {"overall_ascent", det.track_id, ascent_conf, {}, "overall upward movement", frame_counter_});
            analysis.has_ascent = true;
            LOG_INFO_FMT("[攀爬] 目标{} 整体上升 置信度={:.2f}", det.track_id, ascent_conf);
        }

        // ===== 计算过滤器值 =====
        float osc = filterByOscillation(state);
        float lat = filterByLateralMovement(state);
        float burst_val = filterByMovementBurst(state);

        // ===== 收集特征向量（训练数据采集 + ML推理共用） =====
        collectFeatureVector(state, hand_conf, arm_conf, knee_conf,
                             center_conf, tilt_conf, limb_conf,
                             alt_conf, ascent_conf, has_ascent,
                             osc, lat, burst_val);

        // ===== 综合得分 =====
        float final_score;
        if (cfg_.use_ml_score && svm_loaded_) {
            final_score = computeMLScore(state);
            LOG_INFO_FMT("[攀爬] 目标{} ML模型得分={:.2f}", det.track_id, final_score);
        } else {
            float raw_score = score * 0.40f + dynamic_score * 0.60f;

            float penalty = osc * lat ;//* burst_val;

            final_score = raw_score * penalty;

            if (!has_ascent) {
                final_score = 0.0f;
            }

            LOG_INFO_FMT("[攀爬] 目标{} 静态分={:.2f} 动态分={:.2f} 原始分={:.2f} 惩罚={:.2f},震荡={:.2f},横向={:.2f},突变={:.2f} 最终分={:.2f} 上升={}",
                         det.track_id, score, dynamic_score, raw_score, penalty, osc, lat, burst_val, final_score, has_ascent);
        }

        analysis.individual_scores[det.track_id] = final_score;
        analysis.total_score += final_score;
    }

    // 个体综合判定
    for (const auto& det : detections) {
        if (det.track_id < 0) continue;

        float final_score = analysis.individual_scores[det.track_id];
        bool is_suspicious = (final_score > cfg_.climb_score_threshold);

        analysis.is_suspicious[det.track_id] = is_suspicious;
        if (is_suspicious) {
            analysis.suspicious_person_count++;
        }
    }

    return analysis;
}

// ========== 确定状态转换事件 ==========

StateTransitionEvent ClimbingDetector::determineTransitionEvent(
    const FrameAnalysis& analysis) {

    int climbing_person_count = 0;
    for (const auto& [tid, is_suspicious] : analysis.is_suspicious) {
        if (is_suspicious && analysis.individual_scores.at(tid) > cfg_.climb_score_threshold) {
            climbing_person_count++;
        }
    }

    if (climbing_person_count >= cfg_.min_involved_persons) {
        return StateTransitionEvent::CONFIRMED_CLIMBING;
    }

    if (analysis.suspicious_person_count >= cfg_.min_involved_persons) {
        return StateTransitionEvent::MULTI_PERSON_SUSPICIOUS;
    }

    if (analysis.suspicious_person_count > 0) {
        return StateTransitionEvent::INDIVIDUAL_SUSPICIOUS;
    }

    return StateTransitionEvent::NO_EVENT;
}

// ========== 构建结果 ==========

ClimbingResult ClimbingDetector::buildResult(
    const FrameAnalysis& analysis,
    const std::vector<int>& active_ids) {

    ClimbingResult result;
    result.active_track_ids = active_ids;

    for (const auto& [tid, evts] : analysis.individual_events) {
        for (const auto& evt : evts) {
            result.events.push_back(evt);
        }
    }

    if (!analysis.individual_scores.empty()) {
        result.climb_score = std::min(1.0f, analysis.total_score / analysis.individual_scores.size());
    } else {
        result.climb_score = 0.0f;
    }

    return result;
}

// ================================================================
// 辅助函数
// ================================================================

float ClimbingDetector::calculateAngle(
    const core::InferenceResultPacket::KeyPoint& a,
    const core::InferenceResultPacket::KeyPoint& b,
    const core::InferenceResultPacket::KeyPoint& c) {

    float angle1 = std::atan2(a.y - b.y, a.x - b.x);
    float angle2 = std::atan2(c.y - b.y, c.x - b.x);
    float diff = std::abs(angle1 - angle2) * 180.0f / CV_PI;
    if (diff > 180.0f) diff = 360.0f - diff;
    return diff;
}

float ClimbingDetector::calculateOscillation(const std::deque<float>& y_history) {
    if (y_history.size() < 3) return 0.5f;

    int direction_changes = 0;
    for (size_t i = 2; i < y_history.size(); i++) {
        float diff1 = y_history[i - 1] - y_history[i - 2];
        float diff2 = y_history[i] - y_history[i - 1];
        if ((diff1 > 0 && diff2 < 0) || (diff1 < 0 && diff2 > 0)) {
            direction_changes++;
        }
    }
    LOG_INFO_FMT("[攀爬] 振荡过滤: 方向变化次数={}, 历史长度={}", direction_changes, y_history.size());
    return static_cast<float>(direction_changes) / static_cast<float>(y_history.size() - 2);
}

float ClimbingDetector::calculateLateralMovement(const std::deque<cv::Point2f>& center_history) {
    if (center_history.size() < 2) return 0.5f;

    float total_x = 0.0f;
    float total_y = 0.0f;
    for (size_t i = 1; i < center_history.size(); i++) {
        total_x += std::abs(center_history[i].x - center_history[i - 1].x);
        total_y += std::abs(center_history[i].y - center_history[i - 1].y);
    }

    // 两者都几乎不动 → 不是横移
    if (total_x < 1.0f && total_y < 1.0f) return 0.5f;
    // 有横向位移但几乎没纵向位移 → 纯横移
    if (total_y < 1.0f) return 1.0f;

    float ratio = total_x / total_y;
    LOG_INFO_FMT("[攀爬] 横向移动过滤: 比例={:.2f}, 横向总量={}, 纵向总量={}", ratio, total_x, total_y);
    return std::min(1.0f, ratio);
}

float ClimbingDetector::calculateMovementBurst(const std::deque<cv::Point2f>& center_history) {
    if (center_history.size() < 3) return 0.5f;

    std::vector<float> speeds;
    for (size_t i = 1; i < center_history.size(); i++) {
        float dx = center_history[i].x - center_history[i - 1].x;
        float dy = center_history[i].y - center_history[i - 1].y;
        speeds.push_back(std::sqrt(dx * dx + dy * dy));
    }

    float mean = std::accumulate(speeds.begin(), speeds.end(), 0.0f) / speeds.size();
    float variance = 0.0f;
    for (float s : speeds) {
        variance += (s - mean) * (s - mean);
    }
    variance /= speeds.size();

    float cv = (mean > 1e-3f) ? std::sqrt(variance) / mean : 0.0f;
    LOG_INFO_FMT("[攀爬] 运动突变过滤: 均值={}, 方差={}, 变异系数={:.2f}", mean,variance,cv);
    return std::min(1.0f, cv / 2.0f);
}

bool ClimbingDetector::isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp) {
    return kp.visible && kp.confidence > 0.3f;
}

float ClimbingDetector::euclideanDistance(
    const core::InferenceResultPacket::KeyPoint& a,
    const core::InferenceResultPacket::KeyPoint& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

float ClimbingDetector::euclideanDistance(const cv::Point2f& a, const cv::Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

void ClimbingDetector::cleanupInactive(const std::vector<int>& active_ids) {
    std::unordered_set<int> active_set(active_ids.begin(), active_ids.end());
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (active_set.find(it->first) == active_set.end()) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

// ================================================================
// 静态特征检测
// ================================================================

bool ClimbingDetector::detectHandAboveShoulder(
    const core::InferenceResultPacket::BBox& det,
    float& out_conf) {

    if (!det.has_keypoints) { out_conf = 0.0f; return false; }

    const auto& kpts = det.keypoints;
    float offset = det.h * cfg_.arm_raise_offset_ratio;

    bool left_up = isKeypointValid(kpts[Config::LEFT_WRIST]) &&
                   isKeypointValid(kpts[Config::LEFT_SHOULDER]) &&
                   kpts[Config::LEFT_WRIST].y < kpts[Config::LEFT_SHOULDER].y - offset;

    bool right_up = isKeypointValid(kpts[Config::RIGHT_WRIST]) &&
                     isKeypointValid(kpts[Config::RIGHT_SHOULDER]) &&
                     kpts[Config::RIGHT_WRIST].y < kpts[Config::RIGHT_SHOULDER].y - offset;

    if (left_up || right_up) {
        out_conf = (left_up && right_up) ? 1.0f : 0.7f;
        return true;
    }

    out_conf = 0.0f;
    return false;
}

bool ClimbingDetector::detectArmBend(
    const core::InferenceResultPacket::BBox& det,
    float& out_conf) {

    if (!det.has_keypoints) { out_conf = 0.0f; return false; }

    const auto& kpts = det.keypoints;
    float best_conf = 0.0f;
    bool found = false;

    // 左臂
    if (isKeypointValid(kpts[Config::LEFT_SHOULDER]) &&
        isKeypointValid(kpts[Config::LEFT_ELBOW]) &&
        isKeypointValid(kpts[Config::LEFT_WRIST])) {
        float angle = calculateAngle(kpts[Config::LEFT_SHOULDER],
                                     kpts[Config::LEFT_ELBOW],
                                     kpts[Config::LEFT_WRIST]);
        if (angle >= cfg_.arm_bend_min && angle <= cfg_.arm_bend_max) {
            float dist = std::abs(angle - cfg_.arm_bend_ideal);
            float max_dist = std::max(cfg_.arm_bend_ideal - cfg_.arm_bend_min,
                                      cfg_.arm_bend_max - cfg_.arm_bend_ideal);
            float conf = 1.0f - dist / max_dist;
            if (conf > 0) { best_conf = std::max(best_conf, conf); found = true; }
        }
    }

    // 右臂
    if (isKeypointValid(kpts[Config::RIGHT_SHOULDER]) &&
        isKeypointValid(kpts[Config::RIGHT_ELBOW]) &&
        isKeypointValid(kpts[Config::RIGHT_WRIST])) {
        float angle = calculateAngle(kpts[Config::RIGHT_SHOULDER],
                                     kpts[Config::RIGHT_ELBOW],
                                     kpts[Config::RIGHT_WRIST]);
        if (angle >= cfg_.arm_bend_min && angle <= cfg_.arm_bend_max) {
            float dist = std::abs(angle - cfg_.arm_bend_ideal);
            float max_dist = std::max(cfg_.arm_bend_ideal - cfg_.arm_bend_min,
                                      cfg_.arm_bend_max - cfg_.arm_bend_ideal);
            float conf = 1.0f - dist / max_dist;
            if (conf > 0) { best_conf = std::max(best_conf, conf); found = true; }
        }
    }

    out_conf = found ? best_conf : 0.0f;
    return found;
}

bool ClimbingDetector::detectKneeRaise(
    const core::InferenceResultPacket::BBox& det,
    float& out_conf) {

    if (!det.has_keypoints) { out_conf = 0.0f; return false; }

    const auto& kpts = det.keypoints;
    float best_conf = 0.0f;
    bool found = false;

    // 左腿
    if (isKeypointValid(kpts[Config::LEFT_HIP]) &&
        isKeypointValid(kpts[Config::LEFT_KNEE]) &&
        isKeypointValid(kpts[Config::LEFT_ANKLE])) {
        float angle = calculateAngle(kpts[Config::LEFT_HIP],
                                     kpts[Config::LEFT_KNEE],
                                     kpts[Config::LEFT_ANKLE]);
        if (angle >= cfg_.knee_bend_min && angle <= cfg_.knee_bend_max) {
            float dist = std::abs(angle - cfg_.knee_bend_ideal);
            float max_dist = std::max(cfg_.knee_bend_ideal - cfg_.knee_bend_min,
                                      cfg_.knee_bend_max - cfg_.knee_bend_ideal);
            float conf = 1.0f - dist / max_dist;
            if (conf > 0) { best_conf = std::max(best_conf, conf); found = true; }
        }
    }

    // 右腿
    if (isKeypointValid(kpts[Config::RIGHT_HIP]) &&
        isKeypointValid(kpts[Config::RIGHT_KNEE]) &&
        isKeypointValid(kpts[Config::RIGHT_ANKLE])) {
        float angle = calculateAngle(kpts[Config::RIGHT_HIP],
                                     kpts[Config::RIGHT_KNEE],
                                     kpts[Config::RIGHT_ANKLE]);
        if (angle >= cfg_.knee_bend_min && angle <= cfg_.knee_bend_max) {
            float dist = std::abs(angle - cfg_.knee_bend_ideal);
            float max_dist = std::max(cfg_.knee_bend_ideal - cfg_.knee_bend_min,
                                      cfg_.knee_bend_max - cfg_.knee_bend_ideal);
            float conf = 1.0f - dist / max_dist;
            if (conf > 0) { best_conf = std::max(best_conf, conf); found = true; }
        }
    }

    out_conf = found ? best_conf : 0.0f;
    return found;
}

bool ClimbingDetector::detectCenterRaise(
    const core::InferenceResultPacket::BBox& det,
    TrackState& state,
    float& out_conf) {

    if (!det.has_keypoints) { out_conf = 0.0f; return false; }

    const auto& kpts = det.keypoints;
    float sum_y = 0.0f;
    int count = 0;

    for (int i = 0; i < 17; i++) {
        if (isKeypointValid(kpts[i])) {
            sum_y += kpts[i].y;
            count++;
        }
    }

    if (count < 3) { out_conf = 0.0f; return false; }

    float center_y = sum_y / count;

    // 蜷缩检测
    float head_y = -1, ankle_y = -1;
    for (int idx : {Config::NOSE, Config::LEFT_EYE, Config::RIGHT_EYE}) {
        if (isKeypointValid(kpts[idx])) {
            if (head_y < 0 || kpts[idx].y < head_y) head_y = kpts[idx].y;
        }
    }
    for (int idx : {Config::LEFT_ANKLE, Config::RIGHT_ANKLE}) {
        if (isKeypointValid(kpts[idx])) {
            if (ankle_y < 0 || kpts[idx].y > ankle_y) ankle_y = kpts[idx].y;
        }
    }

    bool compressed = false;
    if (head_y > 0 && ankle_y > 0 && det.h > 0) {
        float stretch_ratio = (ankle_y - head_y) / det.h;
        compressed = (stretch_ratio < cfg_.stretch_compress_ratio);
    }

    // 重心抬高：当前center_y vs 历史窗口均值
    bool raised = false;
    float baseline_y = center_y;
    if (state.center_history.size() >= static_cast<size_t>(cfg_.ascent_min_frames)) {
        float acc = 0.0f;
        int w = std::min(static_cast<int>(state.center_history.size()), cfg_.history_frames);
        int start = static_cast<int>(state.center_history.size()) - w;
        for (int i = start; i < static_cast<int>(state.center_history.size()); i++) {
            acc += state.center_history[i].y;
        }
        baseline_y = acc / w;
        float raise_px = cfg_.center_raise_ratio * det.h;
        raised = (center_y < baseline_y - raise_px);
    }

    if (raised || compressed) {
        out_conf = compressed ? 0.8f : (raised ? 0.6f : 0.0f);
        return true;
    }

    out_conf = 0.0f;
    return false;
}

bool ClimbingDetector::detectBodyTilt(
    const core::InferenceResultPacket::BBox& det,
    float& out_conf) {

    if (!det.has_keypoints) { out_conf = 0.0f; return false; }

    const auto& kpts = det.keypoints;
    if (!isKeypointValid(kpts[Config::LEFT_SHOULDER]) ||
        !isKeypointValid(kpts[Config::RIGHT_SHOULDER]) ||
        !isKeypointValid(kpts[Config::LEFT_HIP]) ||
        !isKeypointValid(kpts[Config::RIGHT_HIP])) {
        out_conf = 0.0f;
        return false;
    }

    float shoulder_x = (kpts[Config::LEFT_SHOULDER].x + kpts[Config::RIGHT_SHOULDER].x) / 2.0f;
    float shoulder_y = (kpts[Config::LEFT_SHOULDER].y + kpts[Config::RIGHT_SHOULDER].y) / 2.0f;
    float hip_x = (kpts[Config::LEFT_HIP].x + kpts[Config::RIGHT_HIP].x) / 2.0f;
    float hip_y = (kpts[Config::LEFT_HIP].y + kpts[Config::RIGHT_HIP].y) / 2.0f;

    float angle = std::abs(std::atan2(shoulder_x - hip_x, hip_y - shoulder_y) * 180.0f / CV_PI);

    if (angle >= cfg_.tilt_min && angle <= cfg_.tilt_max) {
        float mid = (cfg_.tilt_min + cfg_.tilt_max) / 2.0f;
        float conf = 1.0f - std::abs(angle - mid) / (cfg_.tilt_max - cfg_.tilt_min) * 2.0f;
        out_conf = std::max(0.3f, conf);
        return true;
    }

    out_conf = 0.0f;
    return false;
}

bool ClimbingDetector::detectLimbSpan(
    const core::InferenceResultPacket::BBox& det,
    float& out_conf) {

    if (!det.has_keypoints) { out_conf = 0.0f; return false; }

    const auto& kpts = det.keypoints;
    bool has_lw = isKeypointValid(kpts[Config::LEFT_WRIST]);
    bool has_rw = isKeypointValid(kpts[Config::RIGHT_WRIST]);
    bool has_la = isKeypointValid(kpts[Config::LEFT_ANKLE]);
    bool has_ra = isKeypointValid(kpts[Config::RIGHT_ANKLE]);

    if ((!has_lw && !has_rw) || (!has_la && !has_ra)) {
        out_conf = 0.0f;
        return false;
    }

    float limb_span = 0.0f;
    if (has_lw && has_rw) {
        limb_span += euclideanDistance(kpts[Config::LEFT_WRIST], kpts[Config::RIGHT_WRIST]);
    } else {
        limb_span += det.w * 0.8f;
    }

    if (has_la && has_ra) {
        limb_span += euclideanDistance(kpts[Config::LEFT_ANKLE], kpts[Config::RIGHT_ANKLE]);
    } else {
        limb_span += det.w * 0.8f;
    }

    float bbox_diag = std::sqrt(det.w * det.w + det.h * det.h);
    if (bbox_diag < 1.0f) { out_conf = 0.0f; return false; }

    float normalize = limb_span / bbox_diag;

    if (normalize > cfg_.limb_span_threshold) {
        out_conf = std::min(1.0f, normalize / (cfg_.limb_span_threshold * 2.0f));
        return true;
    }

    out_conf = 0.0f;
    return false;
}

// ================================================================
// 动态特征检测
// ================================================================

bool ClimbingDetector::detectAlternatingLimb(
    const core::InferenceResultPacket::BBox& det,
    TrackState& state,
    float& out_conf) {

    int window = cfg_.alternation_window;
    if (static_cast<int>(state.left_wrist_y_history.size()) < window ||
        static_cast<int>(state.right_wrist_y_history.size()) < window ||
        static_cast<int>(state.left_wrist_x_history.size()) < window ||
        static_cast<int>(state.right_wrist_x_history.size()) < window) {
        out_conf = 0.0f;
        return false;
    }

    int alternation_count = 0;
    int total = 0;

    auto& ly = state.left_wrist_y_history;
    auto& ry = state.right_wrist_y_history;
    auto& lx = state.left_wrist_x_history;
    auto& rx = state.right_wrist_x_history;
    int n = std::min({static_cast<int>(ly.size()), static_cast<int>(ry.size()),
                      static_cast<int>(lx.size()), static_cast<int>(rx.size())});
    int start = std::max(0, n - window);

    float dx_thresh = det.w * 0.005f;
    if (dx_thresh < cfg_.alternation_dx_threshold) {
        dx_thresh = cfg_.alternation_dx_threshold;
    }

    for (int i = start + 1; i < n; i++) {
        float left_dy = ly[i] - ly[i - 1];
        float right_dy = ry[i] - ry[i - 1];
        float left_dx = lx[i] - lx[i - 1];
        float right_dx = rx[i] - rx[i - 1];

        bool dy_alternation = (left_dy * right_dy < 0);
        bool dx_exceeds = (std::abs(left_dx) > dx_thresh || std::abs(right_dx) > dx_thresh);

        if (dy_alternation && dx_exceeds) {
            alternation_count++;
        }
        total++;
    }

    if (total == 0) { out_conf = 0.0f; return false; }

    float ratio = static_cast<float>(alternation_count) / total;

    if (ratio > cfg_.alternation_ratio_threshold) {
        out_conf = ratio;
        return true;
    }

    out_conf = 0.0f;
    return false;
}

bool ClimbingDetector::detectOverallAscent(
    const core::InferenceResultPacket::BBox& det,
    TrackState& state,
    float& out_conf) {

    int n = state.center_history.size();
    int min_frames = cfg_.ascent_min_frames;
    if (n < min_frames) { out_conf = 0.0f; return false; }

    float bbox_h = det.h;
    if (bbox_h < 1.0f) { out_conf = 0.0f; return false; }

    float net_disp_required = cfg_.net_displacement_ratio * bbox_h;

    int window = std::min(n, min_frames * 2);
    int start = n - window;

    float y_first = state.center_history[start].y;
    float y_last = state.center_history[start + window - 1].y;
    float net_displacement = y_first - y_last;

    LOG_INFO_FMT("[攀爬] 目标{} 上升检测: 净位移={:.2f} 起点Y={:.2f} 终点Y={:.2f} 窗口={} 要求={:.2f}, 身高={}",
                 det.track_id, net_displacement, y_first, y_last, window, net_disp_required, bbox_h);

    if (net_displacement < net_disp_required) {
        state.ascent_frame_count = 0;
        out_conf = 0.0f;
        LOG_INFO_FMT("[攀爬] 目标{} 上升被拦截: 净位移 < 要求值({})", det.track_id, net_disp_required);
        return false;
    }

    float slope_required = cfg_.ascent_slope_ratio * bbox_h;

    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < window; i++) {
        float x = static_cast<float>(i);
        float y = state.center_history[start + i].y;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    float denom = window * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-6f) { out_conf = 0.0f; return false; }

    float slope = (window * sum_xy - sum_x * sum_y) / denom;

    if (slope < slope_required) {
        state.ascent_frame_count++;
        if (state.ascent_frame_count >= 2) {
            out_conf = std::min(1.0f, std::abs(slope) / std::abs(slope_required));
            return true;
        }
    } else {
        state.ascent_frame_count = std::max(0, state.ascent_frame_count - 1);
    }

    out_conf = 0.0f;
    return false;
}

// ================================================================
// 过滤器
// ================================================================

float ClimbingDetector::filterByOscillation(const TrackState& state) const {
    if (state.center_history.size() < 3) return 1.0f;

    std::deque<float> y_hist;
    for (const auto& p : state.center_history) y_hist.push_back(p.y);

    float osc = calculateOscillation(y_hist);

    if (osc > cfg_.oscillation_threshold_high) return 0.3f;
    if (osc > cfg_.oscillation_threshold_low) return 0.7f;
    return 1.0f;
}

float ClimbingDetector::filterByLateralMovement(const TrackState& state) const {
    float lat = calculateLateralMovement(state.center_history);

    if (lat > cfg_.lateral_threshold_high) return 0.3f;
    if (lat > cfg_.lateral_threshold_low) return 0.7f;
    return 1.0f;
}

float ClimbingDetector::filterByMovementBurst(const TrackState& state) const {
    float burst = calculateMovementBurst(state.center_history);

    if (burst > cfg_.burst_threshold_high) return 0.3f;
    if (burst > cfg_.burst_threshold_low) return 0.7f;
    return 1.0f;
}

// ================================================================
// ML 相关
// ================================================================

/* 特征向量定义 (14维，归一化为比例值):
 *   [0] hand_above_shoulder_conf   - 手高于肩置信度 (0-1)
 *   [1] arm_bend_conf              - 手臂弯曲置信度 (0-1)
 *   [2] knee_raise_conf            - 膝盖抬起置信度 (0-1)
 *   [3] center_raise_conf          - 重心抬高置信度 (0-1)
 *   [4] body_tilt_conf             - 身体倾斜置信度 (0-1)
 *   [5] limb_span_conf             - 四肢张开置信度 (0-1)
 *   [6] alternating_limb_conf      - 交替运动置信度 (0-1)
 *   [7] overall_ascent_conf        - 整体上升置信度 (0-1)
 *   [8] has_ascent                 - 是否有上升 (0/1)
 *   [9] oscillation                - 振荡程度 (0-1)
 *  [10] lateral_movement           - 横向移动比例 (0-1)
 *  [11] movement_burst             - 运动突变程度 (0-1)
 *  [12] net_displacement_ratio     - 净Y位移/bbox_h
 *  [13] ascent_slope_ratio         - 上升斜率/bbox_h
 */

static constexpr int FEATURE_DIM = 14;

void ClimbingDetector::collectFeatureVector(
    TrackState& state, float hand_conf, float arm_conf, float knee_conf,
    float center_conf, float tilt_conf, float limb_conf, float alt_conf,
    float ascent_conf, bool has_ascent, float osc, float lat, float burst_val) {

    state.feature_vector.resize(FEATURE_DIM);
    state.feature_vector[0] = hand_conf;
    state.feature_vector[1] = arm_conf;
    state.feature_vector[2] = knee_conf;
    state.feature_vector[3] = center_conf;
    state.feature_vector[4] = tilt_conf;
    state.feature_vector[5] = limb_conf;
    state.feature_vector[6] = alt_conf;
    state.feature_vector[7] = ascent_conf;
    state.feature_vector[8] = has_ascent ? 1.0f : 0.0f;
    state.feature_vector[9] = osc;
    state.feature_vector[10] = lat;
    state.feature_vector[11] = burst_val;

    float net_disp = 0.0f;
    if (state.center_history.size() >= 2 && state.last_bbox.height > 0) {
        float raw_disp = state.center_history.front().y - state.center_history.back().y;
        net_disp = raw_disp / state.last_bbox.height;
    }
    state.feature_vector[12] = net_disp;

    float slope_ratio = 0.0f;
    if (has_ascent && state.last_bbox.height > 0) {
        slope_ratio = ascent_conf * std::abs(cfg_.ascent_slope_ratio * state.last_bbox.height) / (cfg_.ascent_slope_ratio * state.last_bbox.height);
        slope_ratio = has_ascent ? ascent_conf : 0.0f;
    }
    state.feature_vector[13] = slope_ratio;

    state.feature_valid = true;
}

float ClimbingDetector::computeMLScore(const TrackState& state) const {
    if (!state.feature_valid || !svm_loaded_) return 0.0f;
    return svm_model_.predictProbability(state.feature_vector);
}

void ClimbingDetector::exportTrainingData(
    const std::unordered_map<int, TrackState>& tracks,
    const FrameAnalysis& analysis, int64_t frame_id) {

    static std::ofstream csv_file;
    static bool header_written = false;
    static int sample_count = 0;

    if (!csv_file.is_open()) {
        csv_file.open(cfg_.training_data_path, std::ios::app);
        if (!csv_file.is_open()) {
            LOG_WARN_FMT("[攀爬] 无法打开训练数据文件: {}", cfg_.training_data_path);
            return;
        }
    }

    if (!header_written) {
        csv_file << "sample_id,video_id,frame_id,track_id,label,hand_above_shoulder,arm_bend,knee_raise,center_raise,"
                 << "body_tilt,limb_span,alternating_limb,overall_ascent,has_ascent,"
                 << "oscillation,lateral_movement,movement_burst,net_displacement_ratio,ascent_slope_ratio\n";
        header_written = true;
    }

    for (const auto& [tid, state] : tracks) {
        if (!state.feature_valid) continue;

        int label = 0;
        auto it = analysis.is_suspicious.find(tid);
        if (it != analysis.is_suspicious.end() && it->second) {
            auto score_it = analysis.individual_scores.find(tid);
            if (score_it != analysis.individual_scores.end() && score_it->second > cfg_.climb_score_threshold) {
                label = 1;
            }
        }

        sample_count++;
        csv_file << sample_count << ","
                 << cfg_.video_id << ","
                 << frame_id << ","
                 << tid << ","
                 << label;
        for (float f : state.feature_vector) {
            csv_file << "," << f;
        }
        csv_file << "\n";
    }

    csv_file.flush();
}
