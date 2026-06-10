// src/rules/alert/detector/fighting_detector.cpp

#include "3rd_party/log_mgr/log_mgr.h"
#include "fighting_detector.h"
#include <iostream>
#include <unordered_set>
#include <set>

// ========== StateMachine实现 ==========

void FightingDetector::StateMachine::reset() {
    current_state = FightingState::IDLE;
    previous_state = FightingState::IDLE;
    state_duration = 0;
    total_fighting_frames = 0;
    suspicious_count = 0;
    normal_count = 0;
    conflict_count = 0;
    cooldown_count = 0;
}

std::string FightingDetector::StateMachine::getStateString() const {
    switch (current_state) {
        case FightingState::IDLE:       return "IDLE";
        case FightingState::SUSPICIOUS: return "SUSPICIOUS";
        case FightingState::CONFLICT:   return "CONFLICT";
        case FightingState::COOLDOWN:     return "COOLDOWN";
        default: return "UNKNOWN";
    }
}

void FightingDetector::StateMachine::update(StateTransitionEvent event, const Config& cfg) {
    previous_state = current_state;
    state_duration++;

    switch (current_state) {
        case FightingState::IDLE: {
            if (event == StateTransitionEvent::INDIVIDUAL_SUSPICIOUS) {
                suspicious_count++;
                normal_count = 0;
            } else if (event == StateTransitionEvent::MULTI_PERSON_SUSPICIOUS) {
                suspicious_count += 2;
                normal_count = 0;
            } else if (event == StateTransitionEvent::CONFIRMED_FIGHTING) {
                suspicious_count += 3;
                normal_count = 0;
            } else {
                suspicious_count = std::max(0, suspicious_count - 1);
                normal_count++;
            }

            if (suspicious_count >= cfg.suspicious_enter_threshold) {
                current_state = FightingState::SUSPICIOUS;
                state_duration = 0;
                suspicious_count = 0;
                LOG_INFO("State transition: IDLE -> SUSPICIOUS");
            }
            break;
        }

        case FightingState::SUSPICIOUS: {
            if (event == StateTransitionEvent::CONFIRMED_FIGHTING) {
                conflict_count++;
                normal_count = 0;
                if (conflict_count >= cfg.conflict_enter_threshold) {
                    current_state = FightingState::CONFLICT;
                    state_duration = 0;
                    conflict_count = 0;
                    LOG_INFO("State transition: SUSPICIOUS -> CONFLICT");
                }
            } else if (event == StateTransitionEvent::MULTI_PERSON_SUSPICIOUS) {
                // 多人可疑加速进入冲突状态
                conflict_count += 2;
                normal_count = 0;
                if (conflict_count >= cfg.conflict_enter_threshold) {
                    current_state = FightingState::CONFLICT;
                    state_duration = 0;
                    conflict_count = 0;
                    LOG_INFO("State transition: SUSPICIOUS -> CONFLICT (accelerated)");
                }
            } else if (event == StateTransitionEvent::NO_EVENT) {
                normal_count++;
                conflict_count = std::max(0, conflict_count - 1);
                if (normal_count >= cfg.suspicious_exit_threshold) {
                    current_state = FightingState::IDLE;
                    state_duration = 0;
                    normal_count = 0;
                    LOG_INFO("State transition: SUSPICIOUS -> IDLE");
                }
            } else {
                // 保持可疑状态
                normal_count = std::max(0, normal_count - 1);
            }
            break;
        }

        case FightingState::CONFLICT: {
            // 在冲突状态中累计总冲突帧数
            total_fighting_frames++;

            if (event == StateTransitionEvent::NO_EVENT || 
                event == StateTransitionEvent::INDIVIDUAL_SUSPICIOUS) {
                normal_count++;
                if (normal_count >= cfg.conflict_exit_threshold) {
                    current_state = FightingState::COOLDOWN;
                    state_duration = 0;
                    normal_count = 0;
                    cooldown_count = 0;
                    LOG_INFO_FMT("State transition: CONFLICT -> COOLDOWN (total_fighting_frames={})", 
                                 total_fighting_frames);
                }
            } else {
                // 持续冲突中，重置正常计数
                normal_count = std::max(0, normal_count - 2);
            }
            break;
        }

        case FightingState::COOLDOWN: {
            cooldown_count++;
            if (event == StateTransitionEvent::CONFIRMED_FIGHTING) {
                // 冷却期间再次检测到冲突，回到冲突状态，累计帧数不清零
                current_state = FightingState::CONFLICT;
                state_duration = 0;
                cooldown_count = 0;
                normal_count = 0;
                LOG_INFO("State transition: COOLDOWN -> CONFLICT (re-trigger)");
            } else if (cooldown_count >= cfg.cooldown_frames) {
                current_state = FightingState::IDLE;
                state_duration = 0;
                cooldown_count = 0;
                total_fighting_frames = 0;  // 冷却结束，重置累计帧数
                LOG_INFO("State transition: COOLDOWN -> IDLE");
            }
            break;
        }
    }
}

// ========== TrackState实现 ==========

void FightingDetector::TrackState::reset() {
    left_wrist_history.clear();
    right_wrist_history.clear();
    center_history.clear();
    punch_frame_count = 0;
    fall_frame_count = 0;
    last_bbox = cv::Rect2f();
}

// ========== FrameAnalysis实现 ==========

void FightingDetector::FrameAnalysis::clear() {
    individual_scores.clear();
    individual_events.clear();
    is_suspicious.clear();
    suspicious_person_count = 0;
    total_fall_count = 0;
    total_score = 0.0f;
    has_interaction = false;
}

// ========== FightingDetector构造函数 ==========

FightingDetector::FightingDetector(const Config& cfg) : cfg_(cfg), frame_counter_(0) {
    fight_history_.clear();
}

// ========== 重置 ==========

void FightingDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tracks_.clear();
    fight_history_.clear();
    state_machine_.reset();
    frame_counter_ = 0;
}

// ========== 状态查询接口 ==========

FightingState FightingDetector::getCurrentState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.current_state;
}

std::string FightingDetector::getCurrentStateString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.getStateString();
}

int FightingDetector::getStateDuration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.state_duration;
}

int FightingDetector::getTotalFightingFrames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_machine_.total_fighting_frames;
}

// ========== 状态行为函数 ==========

void FightingDetector::onEnterIdle() {
    if (state_machine_.previous_state != FightingState::IDLE) {
        LOG_INFO("Entered IDLE state");
    }
}

void FightingDetector::onEnterSuspicious() {
    LOG_INFO("Entered SUSPICIOUS state - potential conflict detected");
}

void FightingDetector::onEnterConflict() {
    LOG_INFO("Entered CONFLICT state - FIGHTING DETECTED!");
}

void FightingDetector::onEnterCooldown() {
    LOG_INFO("Entered COOLDOWN state - conflict ended, monitoring for re-trigger");
}

void FightingDetector::onStateTick(FightingState state) {
    (void)state; // 保留扩展接口
}

// ========== 主处理函数（状态机驱动）==========

FightingResult FightingDetector::process(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    std::lock_guard<std::mutex> lock(mutex_);
    frame_counter_++;

    FightingResult result;

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
        fight_history_.push_back(false);
        if (fight_history_.size() > MAX_HISTORY) fight_history_.pop_front();

        // 空帧也驱动状态机
        state_machine_.update(StateTransitionEvent::NO_EVENT, cfg_);
        result.is_fighting = (state_machine_.current_state == FightingState::CONFLICT);
        result.current_state = state_machine_.getStateString();
        result.fighting_duration_frames = state_machine_.state_duration;
        result.total_fighting_frames = state_machine_.total_fighting_frames;

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
            case FightingState::IDLE:       onEnterIdle(); break;
            case FightingState::SUSPICIOUS: onEnterSuspicious(); break;
            case FightingState::CONFLICT:   onEnterConflict(); break;
            case FightingState::COOLDOWN:   onEnterCooldown(); break;
        }
    }

    // === 阶段5：构建结果 ===
    result = buildResult(analysis, active_ids);
    result.current_state = state_machine_.getStateString();
    result.fighting_duration_frames = state_machine_.state_duration;
    result.total_fighting_frames = state_machine_.total_fighting_frames;
    result.is_fighting = (state_machine_.current_state == FightingState::CONFLICT);

    // === 阶段6：历史平滑与清理 ===
    fight_history_.push_back(result.is_fighting);
    if (fight_history_.size() > MAX_HISTORY) fight_history_.pop_front();

    cleanupInactive(active_ids);
    onStateTick(state_machine_.current_state);

    LOG_INFO_FMT("Frame {}: State={}, Duration={}, TotalFightFrames={}, Event={}, Fighting={}", 
                 frame_counter_, 
                 state_machine_.getStateString(),
                 state_machine_.state_duration,
                 state_machine_.total_fighting_frames,
                 static_cast<int>(transition_event),
                 result.is_fighting);

    return result;
}

// ========== 帧级分析==========

FightingDetector::FrameAnalysis FightingDetector::analyzeFrame(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    FrameAnalysis analysis;

    // 1. 逐人检测 Punch + Fall
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

        // Punch检测
        float punch_conf = 0.0f;
        bool is_punch = detectPunch(det, state, punch_conf, detections);
        if (is_punch) {
            FightingEvent evt;
            evt.type = "punch";
            evt.track_id = det.track_id;
            evt.confidence = punch_conf;
            evt.timestamp_ms = frame_counter_;
            analysis.individual_events[det.track_id].push_back(evt);
            analysis.individual_scores[det.track_id] += punch_conf * 0.5f;
            LOG_INFO_FMT("Track {} punch detected with confidence {:.2f} score:{:.2f}", 
                         det.track_id, punch_conf, analysis.individual_scores[det.track_id]);
        }

        // Fall检测
        bool is_fall = detectFall(det, state);
        if (is_fall) {
            FightingEvent evt;
            evt.type = "fall";
            evt.track_id = det.track_id;
            evt.confidence = 0.8f;
            evt.timestamp_ms = frame_counter_;
            analysis.individual_events[det.track_id].push_back(evt);
            // fall score 使用 max 而非累加，避免多帧持续fall导致分数无限增长
            analysis.individual_scores[det.track_id] = std::max(
                analysis.individual_scores[det.track_id], 0.4f);
            LOG_INFO_FMT("Track {} fall detected with confidence {:.2f} score:{:.2f}", 
                         det.track_id, evt.confidence, analysis.individual_scores[det.track_id]);
        }
    }

    // 2. 交互检测
    auto interactions = detectInteractions(detections);
    for (auto& inter : interactions) {
        for (int tid : inter.involved_track_ids) {
            analysis.individual_events[tid].push_back(inter);
            analysis.individual_scores[tid] += inter.confidence * 0.4f;
            analysis.has_interaction = true;
            LOG_INFO_FMT("Track {} interaction detected with confidence {:.2f} score:{:.2f}", 
                         tid, inter.confidence, analysis.individual_scores[tid]);
        }
    }

    // 3. 个体综合判定（生成is_suspicious标记）
    analysis.total_fall_count = countFalls(detections);

    for (const auto& det : detections) {
        if (det.track_id < 0) continue;

        int tid = det.track_id;
        float score = analysis.individual_scores[tid];
        const auto& evts = analysis.individual_events[tid];

        bool has_punch = false, has_fall = false, has_interaction = false;
        int punch_count = 0, fall_count = 0;
        for (const auto& evt : evts) {
            if (evt.type == "punch") { has_punch = true; punch_count++; }
            if (evt.type == "fall") { has_fall = true; fall_count++; }
            if (evt.type == "interaction") has_interaction = true;
        }

        LOG_INFO_FMT("Track {} has punch event: {}, fall event: {}, interaction event: {}", 
                     tid, has_punch, has_fall, has_interaction);
        if (fall_count >= 2) {
            LOG_INFO_FMT("Track {} has multiple fall events", tid);
        }

        // 组合规则判定
        bool is_suspicious = false;

        if (has_punch && has_interaction && score > 0.4f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule1: punch + interaction", tid);
        }
        else if (has_fall && has_interaction && score > 0.3f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule2: fall + interaction", tid);
        }
        else if (tracks_[tid].punch_frame_count >= 3 && score > 0.5f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule3: consecutive punch", tid);
        }
        else if (has_fall && has_interaction && analysis.total_fall_count >= 2) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule4: multiple falls + interaction (total={})", tid, analysis.total_fall_count);
        }
        else if (fall_count >= 2 && has_interaction && score > 0.3f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule5: multiple falls on same person + interaction", tid);
        }
        else if (score > 0.5f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule6: high score", tid);
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

StateTransitionEvent FightingDetector::determineTransitionEvent(
    const FrameAnalysis& analysis) {

    // 计算满足阈值的可疑人数
    int fighting_person_count = 0;
    for (const auto& [tid, is_suspicious] : analysis.is_suspicious) {
        if (is_suspicious && analysis.individual_scores.at(tid) > cfg_.fight_score_threshold) {
            fighting_person_count++;
        }
    }

    LOG_INFO_FMT("Transition analysis: suspicious_count={}, fighting_person_count={}, total_score={:.2f}",
                 analysis.suspicious_person_count, fighting_person_count, analysis.total_score);

    if (fighting_person_count >= cfg_.min_involved_persons) {
        return StateTransitionEvent::CONFIRMED_FIGHTING;
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

FightingResult FightingDetector::buildResult(
    const FrameAnalysis& analysis,
    const std::vector<int>& active_ids) {

    FightingResult result;
    result.active_track_ids = active_ids;

    // 收集所有事件（interaction按参与人对去重）
    std::set<std::pair<int, int>> seen_interactions;
    for (const auto& [tid, evts] : analysis.individual_events) {
        for (const auto& evt : evts) {
            if (evt.type == "interaction" && evt.involved_track_ids.size() >= 2) {
                int a = evt.involved_track_ids[0];
                int b = evt.involved_track_ids[1];
                if (a > b) std::swap(a, b);
                if (seen_interactions.insert({a, b}).second) {
                    result.events.push_back(evt);
                }
            } else {
                result.events.push_back(evt);
            }
        }
    }

    // 全局分数
    if (!analysis.individual_scores.empty()) {
        result.fight_score = std::min(1.0f, analysis.total_score / analysis.individual_scores.size());
    } else {
        result.fight_score = 0.0f;
    }

    return result;
}

// ========== Punch检测（与最新版本一致）==========

bool FightingDetector::detectPunch(
    const core::InferenceResultPacket::BBox& det, 
    TrackState& state, 
    float& out_confidence,
    const std::vector<core::InferenceResultPacket::BBox>& all_dets) {

    const auto& kpts = det.keypoints;

    const auto& lw = kpts[Config::LEFT_WRIST];
    const auto& rw = kpts[Config::RIGHT_WRIST];
    const auto& ls = kpts[Config::LEFT_SHOULDER];
    const auto& rs = kpts[Config::RIGHT_SHOULDER];
    const auto& le = kpts[Config::LEFT_ELBOW];
    const auto& re = kpts[Config::RIGHT_ELBOW];

    bool left_valid = isKeypointValid(lw) && isKeypointValid(ls) && isKeypointValid(le);
    bool right_valid = isKeypointValid(rw) && isKeypointValid(rs) && isKeypointValid(re);

    if (!left_valid && !right_valid) {
        out_confidence = 0.0f;
        return false;
    }

    // 姿态预筛选：躯干角度
    float torso_angle = calculateTorsoAngle(kpts);
    LOG_INFO_FMT("Track {} torso_angle={:.1f}",det.track_id,torso_angle);
    // 与垂直方向比较
    float deviation = std::abs(std::abs(torso_angle) - 90.0f);

    if (deviation > cfg_.punch_torso_angle_max) {
        return false;
    }

    // 寻找最近的其他人为潜在目标
    float min_target_dist = std::numeric_limits<float>::max();
    cv::Point2f target_dir;
    bool has_target = false;

    for (const auto& other : all_dets) {
        if (other.track_id == det.track_id || other.track_id < 0) continue;
        if (!other.has_keypoints) continue;

        float dx = (other.x + other.w/2) - (det.x + det.w/2);
        float dy = (other.y + other.h/2) - (det.y + det.h/2);
        float dist = std::sqrt(dx*dx + dy*dy);

        if (dist < min_target_dist && dist < det.w * cfg_.punch_target_dist_max 
            && dist > det.w * cfg_.punch_target_dist_min) {
            min_target_dist = dist;
            target_dir = cv::Point2f(dx, dy);
            has_target = true;
        }
    }

    bool is_punch = false;
    float best_conf = 0.0f;

    // 左手检测
    if (left_valid) {
        cv::Point2f curr(lw.x, lw.y);
        state.left_wrist_history.push_back(curr);
        while (state.left_wrist_history.size() > static_cast<size_t>(cfg_.history_frames)) {
            state.left_wrist_history.pop_front();
        }

        if (state.left_wrist_history.size() >= 3) {
            float speed = calculateSpeed(state.left_wrist_history);
            float angle = calculateAngleChange(state.left_wrist_history);

            float arm_ext = calculateArmExtension(ls, le, lw);
            if (arm_ext < cfg_.punch_min_arm_extension) {
                LOG_DEBUG_FMT("Track {} left arm extension {:.2f} too low", det.track_id, arm_ext);
            } else {
                float conf = 0.0f;

                if (speed > cfg_.punch_speed_threshold) {
                    conf = std::min(1.0f, speed / (cfg_.punch_speed_threshold * 2.0f));

                    if (angle < cfg_.punch_straight_angle_max) {
                        conf = std::min(1.0f, conf * 1.2f);
                    }

                    // 方向一致性检查（使用手腕运动速度方向）
                    if (has_target && state.left_wrist_history.size() >= 2) {
                        const auto& prev = state.left_wrist_history[state.left_wrist_history.size() - 2];
                        const auto& back = state.left_wrist_history.back();
                        float vx = back.x - prev.x;
                        float vy = back.y - prev.y;
                        if (std::abs(vx) > 1.0f || std::abs(vy) > 1.0f) {
                            float wrist_angle = std::atan2(vy, vx) * 180.0f / CV_PI;
                            float target_angle = std::atan2(target_dir.y, target_dir.x) * 180.0f / CV_PI;
                            float angle_diff = std::abs(normalizeAngle(wrist_angle - target_angle));

                            if (angle_diff > cfg_.punch_max_angle_diff) {
                                conf *= 0.3f;
                                LOG_DEBUG_FMT("Track {} left wrist angle diff {:.1f} too large", 
                                             det.track_id, angle_diff);
                            } else {
                                conf *= (1.0f + (cfg_.punch_max_angle_diff - angle_diff) 
                                              / cfg_.punch_max_angle_diff * 0.5f);
                            }
                        }
                    }

                    // 速度突变检测（收拳特征）
                    if (state.left_wrist_history.size() >= 5) {
                        float recent_speed = calculateSpeed(state.left_wrist_history, 3, 0);
                        float older_speed = calculateSpeed(state.left_wrist_history, 3, 2);
                        if (older_speed > 0 && recent_speed / older_speed < cfg_.punch_min_speed_drop_ratio) {
                            conf *= 1.2f;
                        }
                    }
                }

                if (conf > 0.5f) {
                    is_punch = true;
                    best_conf = std::max(best_conf, conf);
                }
                LOG_INFO_FMT("Track {} left wrist speed: {:.2f}, angle: {:.2f}, ext: {:.2f}, conf: {:.2f}", 
                             det.track_id, speed, angle, arm_ext, conf);
            }
        }
    }

    // 右手检测
    if (right_valid) {
        cv::Point2f curr(rw.x, rw.y);
        state.right_wrist_history.push_back(curr);
        while (state.right_wrist_history.size() > static_cast<size_t>(cfg_.history_frames)) {
            state.right_wrist_history.pop_front();
        }

        if (state.right_wrist_history.size() >= 3) {
            float speed = calculateSpeed(state.right_wrist_history);
            float angle = calculateAngleChange(state.right_wrist_history);

            float arm_ext = calculateArmExtension(rs, re, rw);
            if (arm_ext < cfg_.punch_min_arm_extension) {
                LOG_DEBUG_FMT("Track {} right arm extension {:.2f} too low", det.track_id, arm_ext);
            } else {
                float conf = 0.0f;

                if (speed > cfg_.punch_speed_threshold) {
                    conf = std::min(1.0f, speed / (cfg_.punch_speed_threshold * 2.0f));
                    if (angle < cfg_.punch_straight_angle_max) {
                        conf = std::min(1.0f, conf * 1.2f);
                    }

                    if (has_target && state.right_wrist_history.size() >= 2) {
                        const auto& prev = state.right_wrist_history[state.right_wrist_history.size() - 2];
                        const auto& back = state.right_wrist_history.back();
                        float vx = back.x - prev.x;
                        float vy = back.y - prev.y;
                        if (std::abs(vx) > 1.0f || std::abs(vy) > 1.0f) {
                            float wrist_angle = std::atan2(vy, vx) * 180.0f / CV_PI;
                            float target_angle = std::atan2(target_dir.y, target_dir.x) * 180.0f / CV_PI;
                            float angle_diff = std::abs(normalizeAngle(wrist_angle - target_angle));

                            if (angle_diff > cfg_.punch_max_angle_diff) {
                                conf *= 0.3f;
                            } else {
                                conf *= (1.0f + (cfg_.punch_max_angle_diff - angle_diff) 
                                              / cfg_.punch_max_angle_diff * 0.5f);
                            }
                        }
                    }

                    if (state.right_wrist_history.size() >= 5) {
                        float recent_speed = calculateSpeed(state.right_wrist_history, 3, 0);
                        float older_speed = calculateSpeed(state.right_wrist_history, 3, 2);
                        if (older_speed > 0 && recent_speed / older_speed < cfg_.punch_min_speed_drop_ratio) {
                            conf *= 1.2f;
                        }
                    }
                }

                if (conf > 0.5f) {
                    is_punch = true;
                    best_conf = std::max(best_conf, conf);
                }
                LOG_INFO_FMT("Track {} right wrist speed: {:.2f}, angle: {:.2f}, ext: {:.2f}, conf: {:.2f}", 
                             det.track_id, speed, angle, arm_ext, conf);
            }
        }
    }

    // 双手互斥：双手同时超高速运动（>2×阈值）→ 跑步/挥手，取消判定
    if (left_valid && right_valid && state.left_wrist_history.size() >= 3 
        && state.right_wrist_history.size() >= 3) {
        float left_speed = calculateSpeed(state.left_wrist_history);
        float right_speed = calculateSpeed(state.right_wrist_history);

        if (left_speed > cfg_.punch_speed_threshold * 2.0f 
            && right_speed > cfg_.punch_speed_threshold * 2.0f) {
            is_punch = false;
            best_conf = 0.0f;
            LOG_INFO_FMT("Track {} both hands very high speed, cancel punch", det.track_id);
        }
    }

    if (is_punch) {
        state.punch_frame_count++;
    } else {
        state.punch_frame_count = std::max(0, state.punch_frame_count - 1);
    }

    out_confidence = best_conf;
    return is_punch;
}

// ========== Fall检测（与最新版本一致）==========

bool FightingDetector::detectFall(
    const core::InferenceResultPacket::BBox& det, 
    TrackState& state) {

    if (!det.has_keypoints) return false;

    const auto& kpts = det.keypoints;
    bool is_fall = false;
    bool has_head = false;

    // 方法1：基于关键点的高度分析
    float head_y = -1, ankle_y = -1;
    bool has_ankle = false;

    for (int idx : {Config::NOSE, Config::LEFT_EYE, Config::RIGHT_EYE, 
                    Config::LEFT_EAR, Config::RIGHT_EAR}) {
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

    if (has_head && has_ankle) {
        float body_height = ankle_y - head_y;
        if (body_height > 0 && body_height < det.h * cfg_.fall_body_height_ratio) {
            is_fall = true;
        }
    }

    // 方法2：肩-臀连线角度
    if (!is_fall) {
        const auto& ls = kpts[Config::LEFT_SHOULDER];
        const auto& rs = kpts[Config::RIGHT_SHOULDER];
        const auto& lh = kpts[Config::LEFT_HIP];
        const auto& rh = kpts[Config::RIGHT_HIP];

        if (isKeypointValid(ls) && isKeypointValid(rs) && 
            isKeypointValid(lh) && isKeypointValid(rh)) {

            float shoulder_y = (ls.y + rs.y) / 2.0f;
            float hip_y = (lh.y + rh.y) / 2.0f;
            float shoulder_x = (ls.x + rs.x) / 2.0f;
            float hip_x = (lh.x + rh.x) / 2.0f;

            float dx = std::abs(hip_x - shoulder_x);
            float dy = std::abs(hip_y - shoulder_y);

            if (dx > 1.0f) {
                float body_angle = std::atan2(dy, dx) * 180.0f / CV_PI;

                if (body_angle < cfg_.fall_torso_angle_threshold && 
                    hip_y < shoulder_y + det.h * 0.3f) {
                    is_fall = true;
                }
            }
        }
    }

    // 方法3：宽高比作为辅助
    if (!is_fall && det.w > 0 && det.h > 0) {
        float aspect = det.w / det.h;
        if (aspect > cfg_.fall_aspect_ratio_threshold && !has_head) {
            is_fall = true;
        }
    }

    // 时序一致性
    if (is_fall) {
        state.fall_frame_count++;
        return state.fall_frame_count >= cfg_.fall_min_frames;
    } else {
        state.fall_frame_count = 0;
        return false;
    }
}

// ========== Interaction检测（与最新版本一致）==========

std::vector<FightingEvent> FightingDetector::detectInteractions(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    std::vector<FightingEvent> events;

    for (size_t i = 0; i < detections.size(); ++i) {
        for (size_t j = i + 1; j < detections.size(); ++j) {
            const auto& d1 = detections[i];
            const auto& d2 = detections[j];

            if (d1.track_id < 0 || d2.track_id < 0) continue;
            if (!d1.has_keypoints || !d2.has_keypoints) continue;

            float center_dx = (d2.x + d2.w/2) - (d1.x + d1.w/2);
            float center_dy = (d2.y + d2.h/2) - (d1.y + d1.h/2);
            float center_dist = std::sqrt(center_dx*center_dx + center_dy*center_dy);

            if (center_dist > cfg_.interaction_distance_threshold * 2.0f) continue;

            auto& s1 = tracks_[d1.track_id];
            auto& s2 = tracks_[d2.track_id];

            cv::Point2f v1 = calculateVelocity(s1);
            cv::Point2f v2 = calculateVelocity(s2);
            cv::Point2f relative_v(v2.x - v1.x, v2.y - v1.y);
            float relative_speed = std::sqrt(relative_v.x*relative_v.x + relative_v.y*relative_v.y);

            float conn_len = std::sqrt(center_dx*center_dx + center_dy*center_dy);
            cv::Point2f connection(0, 0);
            if (conn_len > 0) {
                connection = cv::Point2f(center_dx/conn_len, center_dy/conn_len);
            }

            float dot_product = relative_v.x * connection.x + relative_v.y * connection.y;
            bool approaching = dot_product < cfg_.interaction_approach_dot_threshold;

            bool facing = checkFacingEachOther(d1, d2);

            float effective_threshold = cfg_.interaction_distance_threshold;
            if (approaching && facing) {
                effective_threshold *= cfg_.interaction_near_multiplier;
            } else if (!approaching && !facing) {
                effective_threshold *= cfg_.interaction_far_multiplier;
            }

            const auto& k1 = d1.keypoints;
            const auto& k2 = d2.keypoints;
            float min_dist = std::numeric_limits<float>::max();
            std::vector<std::string> contacts;

            // 手腕距离（含同侧和异侧，提升检测灵敏度）
            const int wrist_pairs[4][2] = {
                {Config::LEFT_WRIST, Config::LEFT_WRIST},
                {Config::RIGHT_WRIST, Config::RIGHT_WRIST},
                {Config::LEFT_WRIST, Config::RIGHT_WRIST},
                {Config::RIGHT_WRIST, Config::LEFT_WRIST}
            };

            for (auto& pair : wrist_pairs) {
                if (isKeypointValid(k1[pair[0]]) && isKeypointValid(k2[pair[1]])) {
                    float d = euclideanDistance(k1[pair[0]], k2[pair[1]]);
                    if (d < min_dist) min_dist = d;
                    if (d < effective_threshold) contacts.push_back("wrist");
                }
            }

            // 头部距离
            if (isKeypointValid(k1[Config::NOSE]) && isKeypointValid(k2[Config::NOSE])) {
                float d = euclideanDistance(k1[Config::NOSE], k2[Config::NOSE]);
                if (d < min_dist) min_dist = d;
                if (d < effective_threshold) contacts.push_back("head");
            }

            LOG_INFO_FMT("Interaction min_dist={:.1f}, threshold={:.1f}",min_dist,effective_threshold);
            // 判定条件
            bool has_contact = (min_dist < effective_threshold);
            bool has_approach = approaching && relative_speed > cfg_.interaction_min_relative_speed;

            if (has_contact && (has_approach || facing)) {
                FightingEvent evt;
                evt.type = "interaction";
                evt.involved_track_ids = {d1.track_id, d2.track_id};
                evt.contact_parts = contacts.empty() ? "near" : contacts[0];
                evt.confidence = (has_contact ? 0.5f : 0.0f) + 
                                (has_approach ? 0.3f : 0.0f) + 
                                (facing ? 0.2f : 0.0f);
                evt.timestamp_ms = frame_counter_;
                LOG_INFO_FMT("Interaction detected between track {} and {}, contact: {}, approach: {}, facing: {}, conf: {:.2f}", 
                             d1.track_id, d2.track_id, evt.contact_parts, has_approach, facing, evt.confidence);
                events.push_back(evt);
            }
        }
    }

    return events;
}

// ========== 辅助函数（与最新版本一致）==========

float FightingDetector::calculateSpeed(const std::deque<cv::Point2f>& history) {
    if (history.size() < 2) return 0.0f;

    float total_dist = 0.0f;
    for (size_t i = 1; i < history.size(); ++i) {
        float dx = history[i].x - history[i-1].x;
        float dy = history[i].y - history[i-1].y;
        total_dist += std::sqrt(dx*dx + dy*dy);
    }

return total_dist / static_cast<float>(history.size() - 1);
}
float FightingDetector::calculateSpeed(const std::deque<cv::Point2f>& history, int recent_n, int offset) {
    if (history.size() < static_cast<size_t>(offset + recent_n)) return 0.0f;

    size_t end = history.size() - offset;
    if (end <= static_cast<size_t>(recent_n)) return 0.0f;
    size_t start = end - recent_n;

    float total_dist = 0.0f;
    for (size_t i = start + 1; i < end; ++i) {
        float dx = history[i].x - history[i-1].x;
        float dy = history[i].y - history[i-1].y;
        total_dist += std::sqrt(dx*dx + dy*dy);
    }

    size_t count = end - start - 1;
    return count > 0 ? total_dist / static_cast<float>(count) : 0.0f;
}

float FightingDetector::calculateAngleChange(const std::deque<cv::Point2f>& history) {
    if (history.size() < 3) return 0.0f;

    const auto& p1 = history[history.size() - 3];
    const auto& p2 = history[history.size() - 2];
    const auto& p3 = history[history.size() - 1];

    float v1x = p2.x - p1.x;
    float v1y = p2.y - p1.y;
    float v2x = p3.x - p2.x;
    float v2y = p3.y - p2.y;

    float dot = v1x * v2x + v1y * v2y;
    float norm1 = std::sqrt(v1x*v1x + v1y*v1y);
    float norm2 = std::sqrt(v2x*v2x + v2y*v2y);

    if (norm1 <= 0 || norm2 <= 0) return 0.0f;

    float cos_angle = dot / (norm1 * norm2);
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
    return std::acos(cos_angle) * 180.0f / CV_PI;
}

float FightingDetector::euclideanDistance(
    const core::InferenceResultPacket::KeyPoint& a, 
    const core::InferenceResultPacket::KeyPoint& b) {

    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

bool FightingDetector::isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp) {
    return kp.visible && kp.confidence > 0.3f;
}

float FightingDetector::calculateArmExtension(
    const core::InferenceResultPacket::KeyPoint& shoulder,
    const core::InferenceResultPacket::KeyPoint& elbow,
    const core::InferenceResultPacket::KeyPoint& wrist) {

    float upper_arm = euclideanDistance(shoulder, elbow);
    float forearm = euclideanDistance(elbow, wrist);
    float full_arm = euclideanDistance(shoulder, wrist);

    if (upper_arm + forearm < 1.0f) return 0.0f;

    return full_arm / (upper_arm + forearm);
}

float FightingDetector::calculateTorsoAngle(
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

cv::Point2f FightingDetector::calculateVelocity(const TrackState& state) {
    if (state.center_history.size() < 2) return cv::Point2f(0, 0);

    const auto& p1 = state.center_history[state.center_history.size() - 2];
    const auto& p2 = state.center_history[state.center_history.size() - 1];

    return cv::Point2f(p2.x - p1.x, p2.y - p1.y);
}

bool FightingDetector::checkFacingEachOther(
    const core::InferenceResultPacket::BBox& d1,
    const core::InferenceResultPacket::BBox& d2) {

    if (!d1.has_keypoints || !d2.has_keypoints) return false;

    const auto& k1 = d1.keypoints;
    const auto& k2 = d2.keypoints;

    if (!isKeypointValid(k1[Config::NOSE]) || !isKeypointValid(k2[Config::NOSE])) return false;

    float c1x = d1.x + d1.w/2, c1y = d1.y + d1.h/2;
    float c2x = d2.x + d2.w/2, c2y = d2.y + d2.h/2;

    float nose_off1_x = k1[Config::NOSE].x - c1x;
    float nose_off1_y = k1[Config::NOSE].y - c1y;
    float nose_off2_x = k2[Config::NOSE].x - c2x;
    float nose_off2_y = k2[Config::NOSE].y - c2y;

    float conn_x = c2x - c1x;
    float conn_y = c2y - c1y;

    float dot1 = nose_off1_x * conn_x + nose_off1_y * conn_y;
    float dot2 = nose_off2_x * (-conn_x) + nose_off2_y * (-conn_y);

    return dot1 > 0 && dot2 > 0;
}

float FightingDetector::normalizeAngle(float angle) {
    angle = std::fmod(angle, 360.0f);
    if (angle > 180.0f) angle -= 360.0f;
    else if (angle < -180.0f) angle += 360.0f;
    return angle;
}

int FightingDetector::countFalls(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    int count = 0;
    for (const auto& det : detections) {
        if (det.track_id < 0) continue;
        auto it = tracks_.find(det.track_id);
        if (it != tracks_.end() && it->second.fall_frame_count >= cfg_.fall_min_frames) {
            count++;
        }
    }
    return count;
}

void FightingDetector::cleanupInactive(const std::vector<int>& active_ids) {
    std::unordered_set<int> active_set(active_ids.begin(), active_ids.end());

    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (active_set.find(it->first) == active_set.end()) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}