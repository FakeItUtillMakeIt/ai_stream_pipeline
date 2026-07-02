// src/rules/alert/detector/climbing_detector.cpp

#include "3rd_party/log_mgr/log_mgr.h"
#include "climbing_detector.h"
#include <iostream>
#include <unordered_set>
#include <numeric>

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

// ========== TrackState ==========

void ClimbingDetector::TrackState::reset() {
    center_history.clear();
    left_wrist_y_history.clear();
    right_wrist_y_history.clear();
    left_ankle_y_history.clear();
    right_ankle_y_history.clear();
    avg_center_y = -1.0f;
    posture_frame_count = 0;
    ascent_frame_count = 0;
    last_bbox = cv::Rect2f();
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
        LOG_INFO("[Climb] Entered IDLE state");
    }
}

void ClimbingDetector::onEnterSuspicious() {
    LOG_INFO("[Climb] Entered SUSPICIOUS state");
}

void ClimbingDetector::onEnterClimbing() {
    LOG_INFO("[Climb] Entered CLIMBING state - CLIMBING DETECTED!");
}

void ClimbingDetector::onEnterCooldown() {
    LOG_INFO("[Climb] Entered COOLDOWN state");
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

    LOG_INFO_FMT("[Climb] Frame {}: State={}, Duration={}, Event={}, Climbing={}",
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
            while (state.left_wrist_y_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.left_wrist_y_history.pop_front();
        }
        if (isKeypointValid(kpts[Config::RIGHT_WRIST])) {
            state.right_wrist_y_history.push_back(kpts[Config::RIGHT_WRIST].y);
            while (state.right_wrist_y_history.size() > static_cast<size_t>(cfg_.history_frames))
                state.right_wrist_y_history.pop_front();
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
        float conf = 0.0f;

        if (detectHandAboveShoulder(det, conf)) {
            score += conf * 0.20f;
            analysis.individual_events[det.track_id].push_back(
                {"hand_above_shoulder", det.track_id, conf, {}, "hand above shoulder", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} hand_above_shoulder conf={:.2f} score={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectArmBend(det, conf)) {
            score += conf * 0.20f;
            analysis.individual_events[det.track_id].push_back(
                {"arm_bend", det.track_id, conf, {}, "arm bent", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} arm_bend conf={:.2f} score={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectKneeRaise(det, conf)) {
            score += conf * 0.20f;
            analysis.individual_events[det.track_id].push_back(
                {"knee_raise", det.track_id, conf, {}, "knee raised", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} knee_raise conf={:.2f} score={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectCenterRaise(det, state, conf)) {
            score += conf * 0.15f;
            analysis.individual_events[det.track_id].push_back(
                {"center_raise", det.track_id, conf, {}, "center raised or compressed", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} center_raise conf={:.2f} score={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectBodyTilt(det, conf)) {
            score += conf * 0.15f;
            analysis.individual_events[det.track_id].push_back(
                {"body_tilt", det.track_id, conf, {}, "body tilted", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} body_tilt conf={:.2f} score={:.2f}",
                         det.track_id, conf, score);
        }

        if (detectLimbSpan(det, conf)) {
            score += conf * 0.10f;
            analysis.individual_events[det.track_id].push_back(
                {"limb_span", det.track_id, conf, {}, "limbs spread", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} limb_span conf={:.2f} score={:.2f}",
                         det.track_id, conf, score);
        }

        // ===== 动态特征（多帧） =====
        float dyn_conf = 0.0f;

        bool has_alternation = detectAlternatingLimb(det, state, dyn_conf);
        bool has_ascent = detectOverallAscent(det, state, dyn_conf);

        float dynamic_score = 0.0f;
        if (has_alternation) {
            float alt_conf = 0.0f;
            detectAlternatingLimb(det, state, alt_conf);
            dynamic_score += alt_conf * 0.50f;
            analysis.individual_events[det.track_id].push_back(
                {"alternating_limb", det.track_id, alt_conf, {}, "alternating limb movement", frame_counter_});
            LOG_INFO_FMT("[Climb] Track {} alternating_limb conf={:.2f}", det.track_id, alt_conf);
        }

        float ascent_conf = 0.0f;
        has_ascent = detectOverallAscent(det, state, ascent_conf);
        if (has_ascent) {
            dynamic_score += ascent_conf * 0.50f;
            analysis.individual_events[det.track_id].push_back(
                {"overall_ascent", det.track_id, ascent_conf, {}, "overall upward movement", frame_counter_});
            analysis.has_ascent = true;
            LOG_INFO_FMT("[Climb] Track {} overall_ascent conf={:.2f}", det.track_id, ascent_conf);
        }

        // ===== 综合得分 =====
        float raw_score = score * 0.40f + dynamic_score * 0.60f;

        float penalty = filterByOscillation(state)
                      * filterByLateralMovement(state)
                      * filterByMovementBurst(state);

        float final_score = raw_score * penalty;

        // 硬性约束：必须有持续上升
        if (!has_ascent) {
            final_score = 0.0f;
        }

        analysis.individual_scores[det.track_id] = final_score;
        analysis.total_score += final_score;

        LOG_INFO_FMT("[Climb] Track {} static={:.2f} dynamic={:.2f} raw={:.2f} penalty={:.2f} final={:.2f} ascent={}",
                     det.track_id, score, dynamic_score, raw_score, penalty, final_score, has_ascent);
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
    if (y_history.size() < 3) return 0.0f;

    int direction_changes = 0;
    for (size_t i = 2; i < y_history.size(); i++) {
        float diff1 = y_history[i - 1] - y_history[i - 2];
        float diff2 = y_history[i] - y_history[i - 1];
        if ((diff1 > 0 && diff2 < 0) || (diff1 < 0 && diff2 > 0)) {
            direction_changes++;
        }
    }
    LOG_INFO_FMT("[Climb] oscillation direction_changes={}, size={}", direction_changes, y_history.size());
    return static_cast<float>(direction_changes) / static_cast<float>(y_history.size() - 2);
}

float ClimbingDetector::calculateLateralMovement(const std::deque<cv::Point2f>& center_history) {
    if (center_history.size() < 2) return 0.0f;

    float total_x = 0.0f;
    float total_y = 0.0f;
    for (size_t i = 1; i < center_history.size(); i++) {
        total_x += std::abs(center_history[i].x - center_history[i - 1].x);
        total_y += std::abs(center_history[i].y - center_history[i - 1].y);
    }

    // 两者都几乎不动 → 不是横移
    if (total_x < 1.0f && total_y < 1.0f) return 0.0f;
    // 有横向位移但几乎没纵向位移 → 纯横移
    if (total_y < 1.0f) return 1.0f;

    float ratio = total_x / total_y;
    return std::min(1.0f, ratio);
}

float ClimbingDetector::calculateMovementBurst(const std::deque<cv::Point2f>& center_history) {
    if (center_history.size() < 3) return 0.0f;

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
    LOG_INFO_FMT("[Climb] movement_burst mean:{},variance:{},cv={:.2f}", mean,variance,cv);
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
    bool left_up = isKeypointValid(kpts[Config::LEFT_WRIST]) &&
                   isKeypointValid(kpts[Config::LEFT_SHOULDER]) &&
                   kpts[Config::LEFT_WRIST].y < kpts[Config::LEFT_SHOULDER].y - cfg_.arm_raise_offset;

    bool right_up = isKeypointValid(kpts[Config::RIGHT_WRIST]) &&
                    isKeypointValid(kpts[Config::RIGHT_SHOULDER]) &&
                    kpts[Config::RIGHT_WRIST].y < kpts[Config::RIGHT_SHOULDER].y - cfg_.arm_raise_offset;

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
            float conf = 1.0f - std::abs(angle - 90.0f) / 90.0f;
            best_conf = std::max(best_conf, conf);
            found = true;
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
            float conf = 1.0f - std::abs(angle - 90.0f) / 90.0f;
            best_conf = std::max(best_conf, conf);
            found = true;
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
            float conf = 1.0f - std::abs(angle - 90.0f) / 90.0f;
            best_conf = std::max(best_conf, conf);
            found = true;
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
            float conf = 1.0f - std::abs(angle - 90.0f) / 90.0f;
            best_conf = std::max(best_conf, conf);
            found = true;
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

    // 重心抬高
    bool raised = false;
    if (state.avg_center_y < 0) {
        state.avg_center_y = center_y;
    } else {
        raised = (center_y < state.avg_center_y - cfg_.center_raise_threshold);
        state.avg_center_y = 0.9f * state.avg_center_y + 0.1f * center_y;
    }

    if (raised || compressed) {
        out_conf = compressed ? 0.8f : 0.6f;
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
    } else if (has_lw) {
        limb_span += euclideanDistance(kpts[Config::LEFT_WRIST], kpts[Config::LEFT_WRIST]);
    } else {
        limb_span += euclideanDistance(kpts[Config::RIGHT_WRIST], kpts[Config::RIGHT_WRIST]);
    }

    if (has_la && has_ra) {
        limb_span += euclideanDistance(kpts[Config::LEFT_ANKLE], kpts[Config::RIGHT_ANKLE]);
    } else if (has_la) {
        limb_span += euclideanDistance(kpts[Config::LEFT_ANKLE], kpts[Config::LEFT_ANKLE]);
    } else {
        limb_span += euclideanDistance(kpts[Config::RIGHT_ANKLE], kpts[Config::RIGHT_ANKLE]);
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
        static_cast<int>(state.right_wrist_y_history.size()) < window) {
        out_conf = 0.0f;
        return false;
    }

    int alternation_count = 0;
    int total = 0;

    auto& lw = state.left_wrist_y_history;
    auto& rw = state.right_wrist_y_history;
    int n = std::min(static_cast<int>(lw.size()), static_cast<int>(rw.size()));
    int start = std::max(0, n - window);

    for (int i = start + 1; i < n; i++) {
        float left_dy = lw[i] - lw[i - 1];
        float right_dy = rw[i] - rw[i - 1];
        if (left_dy * right_dy < 0) {
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

    // 线性回归拟合Y轴斜率
    int window = std::min(n, min_frames * 2);
    int start = n - window;

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

    if (slope < cfg_.ascent_slope_threshold) {
        state.ascent_frame_count++;
        if (state.ascent_frame_count >= 2) {
            out_conf = std::min(1.0f, std::abs(slope) / std::abs(cfg_.ascent_slope_threshold));
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
