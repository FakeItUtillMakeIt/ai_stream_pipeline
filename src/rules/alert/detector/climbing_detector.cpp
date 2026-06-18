// src/rules/alert/detector/climbing_detector.cpp

#include "3rd_party/log_mgr/log_mgr.h"
#include "climbing_detector.h"
#include <iostream>
#include <unordered_set>

// ========== StateMachine实现 ==========

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
                LOG_INFO("[Climb] State transition: IDLE -> SUSPICIOUS");
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
                    LOG_INFO("[Climb] State transition: SUSPICIOUS -> CLIMBING");
                }
            } else if (event == StateTransitionEvent::MULTI_PERSON_SUSPICIOUS) {
                climbing_count += 2;
                normal_count = 0;
                if (climbing_count >= cfg.climbing_enter_threshold) {
                    current_state = ClimbingState::CLIMBING;
                    state_duration = 0;
                    climbing_count = 0;
                    LOG_INFO("[Climb] State transition: SUSPICIOUS -> CLIMBING (accelerated)");
                }
            } else if (event == StateTransitionEvent::NO_EVENT) {
                normal_count++;
                climbing_count = std::max(0, climbing_count - 1);
                if (normal_count >= cfg.suspicious_exit_threshold) {
                    current_state = ClimbingState::IDLE;
                    state_duration = 0;
                    normal_count = 0;
                    LOG_INFO("[Climb] State transition: SUSPICIOUS -> IDLE");
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
                    LOG_INFO_FMT("[Climb] State transition: CLIMBING -> COOLDOWN (total_climbing_frames={})",
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
                LOG_INFO("[Climb] State transition: COOLDOWN -> CLIMBING (re-trigger)");
            } else if (cooldown_count >= cfg.cooldown_frames) {
                current_state = ClimbingState::IDLE;
                state_duration = 0;
                cooldown_count = 0;
                total_climbing_frames = 0;
                LOG_INFO("[Climb] State transition: COOLDOWN -> IDLE");
            }
            break;
        }
    }
}

// ========== TrackState实现 ==========

void ClimbingDetector::TrackState::reset() {
    center_history.clear();
    left_wrist_y_history.clear();
    right_wrist_y_history.clear();
    left_ankle_y_history.clear();
    right_ankle_y_history.clear();
    posture_frame_count = 0;
    ascent_frame_count = 0;
    last_bbox = cv::Rect2f();
}

// ========== FrameAnalysis实现 ==========

void ClimbingDetector::FrameAnalysis::clear() {
    individual_scores.clear();
    individual_events.clear();
    is_suspicious.clear();
    suspicious_person_count = 0;
    total_climbing_person_count = 0;
    total_score = 0.0f;
    has_ascent = false;
}

// ========== ClimbingDetector构造函数 ==========

ClimbingDetector::ClimbingDetector(const Config& cfg) : cfg_(cfg), frame_counter_(0) {
    climb_history_.clear();
}

// ========== 重置 ==========

void ClimbingDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tracks_.clear();
    climb_history_.clear();
    state_machine_.reset();
    frame_counter_ = 0;
}

// ========== 状态查询接口 ==========

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

// ========== 状态行为函数 ==========

void ClimbingDetector::onEnterIdle() {
    if (state_machine_.previous_state != ClimbingState::IDLE) {
        LOG_INFO("[Climb] Entered IDLE state");
    }
}

void ClimbingDetector::onEnterSuspicious() {
    LOG_INFO("[Climb] Entered SUSPICIOUS state - potential climbing detected");
}

void ClimbingDetector::onEnterClimbing() {
    LOG_INFO("[Climb] Entered CLIMBING state - CLIMBING DETECTED!");
}

void ClimbingDetector::onEnterCooldown() {
    LOG_INFO("[Climb] Entered COOLDOWN state - climbing ended, monitoring for re-trigger");
}

void ClimbingDetector::onStateTick(ClimbingState state) {
    (void)state;
}

// ========== 主处理函数（状态机驱动）==========

ClimbingResult ClimbingDetector::process(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    std::lock_guard<std::mutex> lock(mutex_);
    frame_counter_++;

    ClimbingResult result;

    // 收集活跃track_id
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

    // === 阶段1：帧级分析（行为层）===
    FrameAnalysis analysis = analyzeFrame(detections);

    // === 阶段2：确定状态转换事件 ===
    StateTransitionEvent transition_event = determineTransitionEvent(analysis);

    // === 阶段3：驱动状态机 ===
    state_machine_.update(transition_event, cfg_);

    // === 阶段4：状态进入钩子 ===
    if (state_machine_.previous_state != state_machine_.current_state) {
        switch (state_machine_.current_state) {
            case ClimbingState::IDLE:       onEnterIdle(); break;
            case ClimbingState::SUSPICIOUS: onEnterSuspicious(); break;
            case ClimbingState::CLIMBING:   onEnterClimbing(); break;
            case ClimbingState::COOLDOWN:   onEnterCooldown(); break;
        }
    }

    // === 阶段5：构建结果 ===
    result = buildResult(analysis, active_ids);
    result.current_state = state_machine_.getStateString();
    result.climbing_duration_frames = state_machine_.state_duration;
    result.total_climbing_frames = state_machine_.total_climbing_frames;
    result.is_climbing = (state_machine_.current_state == ClimbingState::CLIMBING);

    // === 阶段6：历史平滑与清理 ===
    climb_history_.push_back(result.is_climbing);
    if (climb_history_.size() > MAX_HISTORY) climb_history_.pop_front();

    cleanupInactive(active_ids);
    onStateTick(state_machine_.current_state);

    LOG_INFO_FMT("[Climb] Frame {}: State={}, Duration={}, TotalClimbFrames={}, Event={}, Climbing={}",
                 frame_counter_,
                 state_machine_.getStateString(),
                 state_machine_.state_duration,
                 state_machine_.total_climbing_frames,
                 static_cast<int>(transition_event),
                 result.is_climbing);

    return result;
}

// ========== 帧级分析 ==========

ClimbingDetector::FrameAnalysis ClimbingDetector::analyzeFrame(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    FrameAnalysis analysis;

    // 逐人检测攀爬特征
    for (const auto& det : detections) {
        if (det.track_id < 0) continue;
        if (!det.has_keypoints) continue;

        auto& state = tracks_[det.track_id];
        state.last_bbox = cv::Rect2f(det.x, det.y, det.w, det.h);

        // 更新中心点历史
        cv::Point2f center(det.x + det.w/2, det.y + det.h/2);
        state.center_history.push_back(center);
        while (state.center_history.size() > static_cast<size_t>(cfg_.history_frames)) {
            state.center_history.pop_front();
        }

        // 更新手腕/脚踝Y坐标历史
        const auto& kpts = det.keypoints;
        if (isKeypointValid(kpts[Config::LEFT_WRIST])) {
            state.left_wrist_y_history.push_back(kpts[Config::LEFT_WRIST].y);
            while (state.left_wrist_y_history.size() > static_cast<size_t>(cfg_.history_frames)) {
                state.left_wrist_y_history.pop_front();
            }
        }
        if (isKeypointValid(kpts[Config::RIGHT_WRIST])) {
            state.right_wrist_y_history.push_back(kpts[Config::RIGHT_WRIST].y);
            while (state.right_wrist_y_history.size() > static_cast<size_t>(cfg_.history_frames)) {
                state.right_wrist_y_history.pop_front();
            }
        }
        if (isKeypointValid(kpts[Config::LEFT_ANKLE])) {
            state.left_ankle_y_history.push_back(kpts[Config::LEFT_ANKLE].y);
            while (state.left_ankle_y_history.size() > static_cast<size_t>(cfg_.history_frames)) {
                state.left_ankle_y_history.pop_front();
            }
        }
        if (isKeypointValid(kpts[Config::RIGHT_ANKLE])) {
            state.right_ankle_y_history.push_back(kpts[Config::RIGHT_ANKLE].y);
            while (state.right_ankle_y_history.size() > static_cast<size_t>(cfg_.history_frames)) {
                state.right_ankle_y_history.pop_front();
            }
        }

        // 1. 姿态检测
        float posture_conf = 0.0f;
        bool is_posture = detectClimbingPosture(det, state, posture_conf);
        if (is_posture) {
            ClimbingEvent evt;
            evt.type = "posture";
            evt.track_id = det.track_id;
            evt.confidence = posture_conf;
            evt.timestamp_ms = frame_counter_;
            evt.detail = "abnormal climbing posture detected";
            analysis.individual_events[det.track_id].push_back(evt);
            analysis.individual_scores[det.track_id] += posture_conf * 0.5f;
            LOG_INFO_FMT("[Climb] Track {} posture detected, conf: {:.2f}, score: {:.2f}",
                         det.track_id, posture_conf, analysis.individual_scores[det.track_id]);
        }

        // 2. 垂直上升检测
        float ascent_conf = 0.0f;
        bool is_ascent = detectVerticalAscent(det, state, ascent_conf);
        if (is_ascent) {
            ClimbingEvent evt;
            evt.type = "ascent";
            evt.track_id = det.track_id;
            evt.confidence = ascent_conf;
            evt.timestamp_ms = frame_counter_;
            evt.detail = "vertical ascent detected";
            analysis.individual_events[det.track_id].push_back(evt);
            analysis.individual_scores[det.track_id] += ascent_conf * 0.4f;
            analysis.has_ascent = true;
            LOG_INFO_FMT("[Climb] Track {} ascent detected, conf: {:.2f}, score: {:.2f}",
                         det.track_id, ascent_conf, analysis.individual_scores[det.track_id]);
        }

        // 3. 肢体不对称检测
        float asym_conf = 0.0f;
        bool is_asym = detectLimbAsymmetry(det, state, asym_conf);
        if (is_asym) {
            ClimbingEvent evt;
            evt.type = "limb_asymmetry";
            evt.track_id = det.track_id;
            evt.confidence = asym_conf;
            evt.timestamp_ms = frame_counter_;
            evt.detail = "alternating limb movement detected";
            analysis.individual_events[det.track_id].push_back(evt);
            analysis.individual_scores[det.track_id] += asym_conf * 0.3f;
            LOG_INFO_FMT("[Climb] Track {} asymmetry detected, conf: {:.2f}, score: {:.2f}",
                         det.track_id, asym_conf, analysis.individual_scores[det.track_id]);
        }

        // 4. 脚部离地检测
        float foot_conf = 0.0f;
        bool is_foot_off = detectFootOffGround(det, foot_conf);
        if (is_foot_off) {
            ClimbingEvent evt;
            evt.type = "foot_off_ground";
            evt.track_id = det.track_id;
            evt.confidence = foot_conf;
            evt.timestamp_ms = frame_counter_;
            evt.detail = "feet off ground detected";
            analysis.individual_events[det.track_id].push_back(evt);
            analysis.individual_scores[det.track_id] += foot_conf * 0.2f;
            LOG_INFO_FMT("[Climb] Track {} foot off ground detected, conf: {:.2f}, score: {:.2f}",
                         det.track_id, foot_conf, analysis.individual_scores[det.track_id]);
        }
    }

    // 个体综合判定
    for (const auto& det : detections) {
        if (det.track_id < 0) continue;

        int tid = det.track_id;
        float score = analysis.individual_scores[tid];
        const auto& evts = analysis.individual_events[tid];

        bool has_posture = false, has_ascent = false, has_asym = false, has_foot_off = false;
        for (const auto& evt : evts) {
            if (evt.type == "posture") has_posture = true;
            if (evt.type == "ascent") has_ascent = true;
            if (evt.type == "limb_asymmetry") has_asym = true;
            if (evt.type == "foot_off_ground") has_foot_off = true;
        }

        LOG_INFO_FMT("[Climb] Track {} has posture: {}, ascent: {}, asymmetry: {}, foot_off: {}",
                     tid, has_posture, has_ascent, has_asym, has_foot_off);

        // 组合规则判定
        bool is_suspicious = false;

        if (has_ascent && has_posture && score > 0.5f) {
            is_suspicious = true;
            LOG_INFO_FMT("[Climb] Track {} triggered rule1: ascent + posture", tid);
        }
        else if (has_ascent && has_foot_off && score > 0.4f) {
            is_suspicious = true;
            LOG_INFO_FMT("[Climb] Track {} triggered rule2: ascent + foot_off_ground", tid);
        }
        else if (has_posture && has_asym && score > 0.4f) {
            is_suspicious = true;
            LOG_INFO_FMT("[Climb] Track {} triggered rule3: posture + asymmetry", tid);
        }
        else if (has_ascent && has_asym && has_foot_off && score > 0.5f) {
            is_suspicious = true;
            LOG_INFO_FMT("[Climb] Track {} triggered rule4: ascent + asymmetry + foot_off", tid);
        }
        else if (has_posture && tracks_[tid].posture_frame_count >= 3 && score > 0.4f) {
            is_suspicious = true;
            LOG_INFO_FMT("[Climb] Track {} triggered rule5: sustained abnormal posture", tid);
        }
        else if (score > 0.6f) {
            is_suspicious = true;
            LOG_INFO_FMT("[Climb] Track {} triggered rule6: high score", tid);
        }

        analysis.is_suspicious[tid] = is_suspicious;
        if (is_suspicious) {
            analysis.suspicious_person_count++;
        }
        analysis.total_score += score;
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

    LOG_INFO_FMT("[Climb] Transition analysis: suspicious_count={}, climbing_person_count={}, total_score={:.2f}",
                 analysis.suspicious_person_count, climbing_person_count, analysis.total_score);

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

// ========== 姿态检测 ==========

bool ClimbingDetector::detectClimbingPosture(
    const core::InferenceResultPacket::BBox& det,
    TrackState& state,
    float& out_confidence) {

    if (!det.has_keypoints) return false;

    const auto& kpts = det.keypoints;
    float conf = 0.0f;
    int feature_count = 0;

    // 1. 躯干角度（偏离垂直方向）
    float torso_angle = calculateTorsoAngle(kpts);
    float torso_deviation = std::abs(std::abs(torso_angle) - 90.0f);
    LOG_INFO_FMT("[Climb] Track {} torso_angle={:.1f}, deviation={:.1f}", det.track_id, torso_angle, torso_deviation);

    if (torso_deviation > cfg_.torso_angle_climb_min && torso_deviation < cfg_.torso_angle_climb_max) {
        conf += std::min(1.0f, torso_deviation / 90.0f) * 0.3f;
        feature_count++;
        LOG_INFO_FMT("[Climb] Track {} torso tilt feature matched", det.track_id);
    }

    // 2. 手臂上举（手腕高于肩膀）
    const auto& ls = kpts[Config::LEFT_SHOULDER];
    const auto& rs = kpts[Config::RIGHT_SHOULDER];
    const auto& lw = kpts[Config::LEFT_WRIST];
    const auto& rw = kpts[Config::RIGHT_WRIST];

    bool left_arm_up = isKeypointValid(ls) && isKeypointValid(lw) && (ls.y - lw.y) > cfg_.arm_raise_threshold;
    bool right_arm_up = isKeypointValid(rs) && isKeypointValid(rw) && (rs.y - rw.y) > cfg_.arm_raise_threshold;

    if (left_arm_up || right_arm_up) {
        conf += 0.25f;
        feature_count++;
        LOG_INFO_FMT("[Climb] Track {} arm raise: left={}, right={}", det.track_id, left_arm_up, right_arm_up);
    }

    // 3. 腿部抬起（脚踝高于膝盖）
    const auto& lk = kpts[Config::LEFT_KNEE];
    const auto& rk = kpts[Config::RIGHT_KNEE];
    const auto& la = kpts[Config::LEFT_ANKLE];
    const auto& ra = kpts[Config::RIGHT_ANKLE];

    bool left_leg_up = isKeypointValid(lk) && isKeypointValid(la) && (lk.y - la.y) > cfg_.leg_raise_threshold;
    bool right_leg_up = isKeypointValid(rk) && isKeypointValid(ra) && (rk.y - ra.y) > cfg_.leg_raise_threshold;

    if (left_leg_up || right_leg_up) {
        conf += 0.25f;
        feature_count++;
        LOG_INFO_FMT("[Climb] Track {} leg raise: left={}, right={}", det.track_id, left_leg_up, right_leg_up);
    }

    // 4. 身体拉伸/压缩（头-踝距离 vs bbox高度）
    float head_y = -1, ankle_y = -1;
    bool has_head = false, has_ankle = false;

    for (int idx : {Config::NOSE, Config::LEFT_EYE, Config::RIGHT_EYE}) {
        if (isKeypointValid(kpts[idx])) {
            if (!has_head || kpts[idx].y < head_y) {
                head_y = kpts[idx].y;
                has_head = true;
            }
        }
    }

    for (int idx : {Config::LEFT_ANKLE, Config::RIGHT_ANKLE}) {
        if (isKeypointValid(kpts[idx])) {
            if (!has_ankle || kpts[idx].y > ankle_y) {
                ankle_y = kpts[idx].y;
                has_ankle = true;
            }
        }
    }

    if (has_head && has_ankle && det.h > 0) {
        float body_length = ankle_y - head_y;
        float stretch_ratio = body_length / det.h;

        if (stretch_ratio > cfg_.body_stretch_ratio) {
            conf += 0.2f;
            feature_count++;
            LOG_INFO_FMT("[Climb] Track {} body stretch ratio: {:.2f}", det.track_id, stretch_ratio);
        } else if (stretch_ratio < cfg_.body_compress_ratio) {
            conf += 0.15f;
            feature_count++;
            LOG_INFO_FMT("[Climb] Track {} body compress ratio: {:.2f}", det.track_id, stretch_ratio);
        }
    }

    // 5. 宽高比（水平姿态/翻越栏杆）
    if (det.w > 0 && det.h > 0) {
        float aspect = det.w / det.h;
        if (aspect > cfg_.aspect_ratio_threshold) {
            conf += 0.15f;
            feature_count++;
            LOG_INFO_FMT("[Climb] Track {} aspect ratio: {:.2f}", det.track_id, aspect);
        }
    }

    // 时序一致性
    if (feature_count >= 2) {
        state.posture_frame_count++;
        if (state.posture_frame_count >= 2) {
            out_confidence = std::min(1.0f, conf);
            return true;
        }
    } else {
        state.posture_frame_count = std::max(0, state.posture_frame_count - 1);
    }

    out_confidence = 0.0f;
    return false;
}

// ========== 垂直上升检测 ==========

bool ClimbingDetector::detectVerticalAscent(
    const core::InferenceResultPacket::BBox& det,
    TrackState& state,
    float& out_confidence) {

    if (state.center_history.size() < static_cast<size_t>(cfg_.ascent_min_frames)) {
        out_confidence = 0.0f;
        return false;
    }

    // 计算Y轴速度（向上为负，故速度 < -threshold 表示上升）
    float speed_y = calculateSpeedY(state.center_history);
    LOG_INFO_FMT("[Climb] Track {} vertical speed_y={:.2f}", det.track_id, speed_y);

    if (speed_y < -cfg_.ascent_speed_threshold) {
        // 检查总上升距离
        float total_ascent = 0.0f;
        for (size_t i = 1; i < state.center_history.size(); ++i) {
            float dy = state.center_history[i-1].y - state.center_history[i].y; // 向上为正
            if (dy > 0) total_ascent += dy;
        }

        LOG_INFO_FMT("[Climb] Track {} total ascent={:.1f} pixels", det.track_id, total_ascent);

        if (total_ascent > cfg_.ascent_min_distance) {
            state.ascent_frame_count++;
            if (state.ascent_frame_count >= 2) {
                out_confidence = std::min(1.0f, total_ascent / (cfg_.ascent_min_distance * 3.0f));
                return true;
            }
        }
    } else {
        state.ascent_frame_count = std::max(0, state.ascent_frame_count - 1);
    }

    out_confidence = 0.0f;
    return false;
}

// ========== 肢体不对称检测 ==========

bool ClimbingDetector::detectLimbAsymmetry(
    const core::InferenceResultPacket::BBox& det,
    TrackState& state,
    float& out_confidence) {

    if (!det.has_keypoints) return false;

    const auto& kpts = det.keypoints;

    // 检查左右手腕高度差
    bool has_lw = isKeypointValid(kpts[Config::LEFT_WRIST]);
    bool has_rw = isKeypointValid(kpts[Config::RIGHT_WRIST]);
    bool has_la = isKeypointValid(kpts[Config::LEFT_ANKLE]);
    bool has_ra = isKeypointValid(kpts[Config::RIGHT_ANKLE]);

    float max_diff = 0.0f;

    if (has_lw && has_rw) {
        float wrist_diff = std::abs(kpts[Config::LEFT_WRIST].y - kpts[Config::RIGHT_WRIST].y);
        max_diff = std::max(max_diff, wrist_diff);
    }

    if (has_la && has_ra) {
        float ankle_diff = std::abs(kpts[Config::LEFT_ANKLE].y - kpts[Config::RIGHT_ANKLE].y);
        max_diff = std::max(max_diff, ankle_diff);
    }

    if (det.h > 0) {
        float diff_ratio = max_diff / det.h;
        LOG_INFO_FMT("[Climb] Track {} limb asymmetry ratio: {:.3f}", det.track_id, diff_ratio);

        if (diff_ratio > cfg_.limb_asymmetry_threshold) {
            out_confidence = std::min(1.0f, diff_ratio / (cfg_.limb_asymmetry_threshold * 3.0f));
            return true;
        }
    }

    out_confidence = 0.0f;
    return false;
}

// ========== 脚部离地检测 ==========

bool ClimbingDetector::detectFootOffGround(
    const core::InferenceResultPacket::BBox& det,
    float& out_confidence) {

    if (!det.has_keypoints) return false;

    const auto& kpts = det.keypoints;
    float bbox_top = det.y;
    float bbox_bottom = det.y + det.h;
    float bbox_mid = bbox_top + det.h * cfg_.foot_off_ground_ratio;

    bool left_off = isKeypointValid(kpts[Config::LEFT_ANKLE]) && kpts[Config::LEFT_ANKLE].y < bbox_mid;
    bool right_off = isKeypointValid(kpts[Config::RIGHT_ANKLE]) && kpts[Config::RIGHT_ANKLE].y < bbox_mid;

    LOG_INFO_FMT("[Climb] Track {} foot off ground: left={}, right={}, mid_y={:.1f}",
                 det.track_id, left_off, right_off, bbox_mid);

    if (left_off || right_off) {
        out_confidence = (left_off && right_off) ? 1.0f : 0.6f;
        return true;
    }

    out_confidence = 0.0f;
    return false;
}

// ========== 辅助函数 ==========

float ClimbingDetector::calculateSpeedY(const std::deque<cv::Point2f>& history) {
    if (history.size() < 2) return 0.0f;

    float total_dy = 0.0f;
    for (size_t i = 1; i < history.size(); ++i) {
        total_dy += history[i].y - history[i-1].y; // Y轴向下为正
    }

    return total_dy / static_cast<float>(history.size() - 1);
}

float ClimbingDetector::calculateTorsoAngle(
    const std::array<core::InferenceResultPacket::KeyPoint, 17>& kpts) {

    const auto& ls = kpts[Config::LEFT_SHOULDER];
    const auto& rs = kpts[Config::RIGHT_SHOULDER];
    const auto& lh = kpts[Config::LEFT_HIP];
    const auto& rh = kpts[Config::RIGHT_HIP];

    if (!isKeypointValid(ls) || !isKeypointValid(rs) ||
        !isKeypointValid(lh) || !isKeypointValid(rh)) {
        return 0.0f;
    }

    float shoulder_x = (ls.x + rs.x) / 2.0f;
    float shoulder_y = (ls.y + rs.y) / 2.0f;
    float hip_x = (lh.x + rh.x) / 2.0f;
    float hip_y = (lh.y + rh.y) / 2.0f;

    return std::atan2(hip_y - shoulder_y, hip_x - shoulder_x) * 180.0f / CV_PI;
}

bool ClimbingDetector::isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp) {
    return kp.visible && kp.confidence > 0.3f;
}

float ClimbingDetector::euclideanDistance(
    const core::InferenceResultPacket::KeyPoint& a,
    const core::InferenceResultPacket::KeyPoint& b) {

    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

float ClimbingDetector::normalizeAngle(float angle) {
    angle = std::fmod(angle, 360.0f);
    if (angle > 180.0f) angle -= 360.0f;
    else if (angle < -180.0f) angle += 360.0f;
    return angle;
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