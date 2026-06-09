// src/rules/alert/detector/fighting_detector.cpp

#include "3rd_party/log_mgr/log_mgr.h"
#include "fighting_detector.h"
#include <iostream>
#include <unordered_set>

// ========== 构造函数 ==========
FightingDetector::FightingDetector(const Config& cfg) : cfg_(cfg) {
    fight_history_.clear();
}

// ========== 重置 ==========
void FightingDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tracks_.clear();
    fighting_tracks_.clear();
    fight_history_.clear();
}

// ========== 主处理函数 ==========
FightingResult FightingDetector::process(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
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
        return result;
    }

    // === 为每个人独立计算 score ===
    std::unordered_map<int, float> individual_scores;
    std::unordered_map<int, std::vector<FightingEvent>> individual_events;

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

        // Punch检测（传入所有检测用于目标分析）
        float punch_conf = 0.0f;
        bool is_punch = detectPunch(det, state, punch_conf, detections);
        if (is_punch) {
            FightingEvent evt;
            evt.type = "punch";
            evt.track_id = det.track_id;
            evt.confidence = punch_conf;
            individual_events[det.track_id].push_back(evt);
            individual_scores[det.track_id] += punch_conf * 0.5f;
            LOG_INFO_FMT("Track {} punch detected with confidence {:.2f} score:{:.2f}", 
                         det.track_id, punch_conf, individual_scores[det.track_id]);
        }

        // Fall检测
        bool is_fall = detectFall(det, state);
        if (is_fall) {
            FightingEvent evt;
            evt.type = "fall";
            evt.track_id = det.track_id;
            evt.confidence = 0.8f;
            individual_events[det.track_id].push_back(evt);
            individual_scores[det.track_id] += 0.4f;  // 提高 fall 权重
            LOG_INFO_FMT("Track {} fall detected with confidence {:.2f} score:{:.2f}", 
                         det.track_id, evt.confidence, individual_scores[det.track_id]);
        }
    }

    // 2. 交互检测
    auto interactions = detectInteractions(detections);
    for (auto& inter : interactions) {
        for (int tid : inter.involved_track_ids) {
            individual_events[tid].push_back(inter);
            individual_scores[tid] += inter.confidence * 0.4f;  // 提高 interaction 权重
            LOG_INFO_FMT("Track {} interaction detected with confidence {:.2f} score:{:.2f}", 
                         tid, inter.confidence, individual_scores[tid]);
        }
    }

    // 3. 个体综合判定
    std::unordered_map<int, int> new_fighting_counts;
    int total_fall_count = countFalls(detections);  // 先计算全局 fall 数量

    for (const auto& det : detections) {
        if (det.track_id < 0) continue;

        int tid = det.track_id;
        float score = individual_scores[tid];
        const auto& evts = individual_events[tid];

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

        // 规则1：punch + interaction = 高置信度冲突
        if (has_punch && has_interaction && score > 0.4f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule1: punch + interaction", tid);
        }
        // 规则2：fall + interaction = 可能冲突（一人被打倒）
        else if (has_fall && has_interaction && score > 0.3f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule2: fall + interaction", tid);
        }
        // 规则3：连续 punch = 冲突
        else if (tracks_[tid].punch_frame_count >= 3 && score > 0.5f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule3: consecutive punch", tid);
        }
        // 规则4：多人同时 fall = 群体冲突
        else if (has_fall && total_fall_count >= 2) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule4: multiple falls (total={})", tid, total_fall_count);
        }
        // 规则5：单人多次 fall = 可能冲突（被反复击倒）
        else if (fall_count >= 2 && score > 0.3f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule5: multiple falls on same person", tid);
        }
        // 规则6：高 score 直接触发（综合多种弱信号叠加）
        else if (score > 0.5f) {
            is_suspicious = true;
            LOG_INFO_FMT("Track {} triggered rule6: high score", tid);
        }

        auto& count = fighting_tracks_[tid];
        if (is_suspicious) {
            count++;
        } else {
            count = std::max(0, count - 1);
        }
        new_fighting_counts[tid] = count;
    }

    // 4. 全局判定：要求至少两人同时达到阈值
    int fighting_person_count = 0;
    float total_score = 0.0f;
    for (const auto& [tid, count] : new_fighting_counts) {
        LOG_INFO_FMT("Track {} fighting count: {},score: {:.2f}", tid, count, individual_scores[tid]);
        if (count >= cfg_.fight_frames_threshold && 
            individual_scores[tid] > cfg_.fight_score_threshold) {
            fighting_person_count++;
        }
        total_score += individual_scores[tid];
    }

    // 关键：打架需要至少两人参与
    bool is_fighting = (fighting_person_count >= cfg_.min_involved_persons);

    // 收集所有事件
    for (const auto& [tid, evts] : individual_events) {
        result.events.insert(result.events.end(), evts.begin(), evts.end());
    }

    // 全局分数
    if (!individual_scores.empty()) {
        result.fight_score = std::min(1.0f, total_score / individual_scores.size());
    } else {
        result.fight_score = 0.0f;
    }

    result.is_fighting = is_fighting;
    LOG_INFO_FMT("Fighting person count: {}, fight score: {:.2f},individual scores: {},is fighting: {}", 
                 fighting_person_count, result.fight_score, individual_scores.size(),is_fighting);
    // 历史平滑
    fight_history_.push_back(is_fighting);
    if (fight_history_.size() > MAX_HISTORY) fight_history_.pop_front();

    // 清理不活跃
    cleanupInactive(active_ids);

    return result;
}

// ========== Punch检测 ==========
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

    // === 姿态预筛选：躯干角度 ===
    float torso_angle = calculateTorsoAngle(kpts);
    if (std::abs(torso_angle) > 60.0f) {
        out_confidence = 0.0f;
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

        // 目标需在合理范围内（3倍自身宽度）
        if (dist < min_target_dist && dist < det.w * 3.0f && dist > det.w * 0.5f) {
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

            // === 手臂伸展度检查 ===
            float arm_ext = calculateArmExtension(ls, le, lw);
            if (arm_ext < cfg_.punch_min_arm_extension) {
                LOG_DEBUG_FMT("Track {} left arm extension {:.2f} too low", det.track_id, arm_ext);
            } else {
                float conf = 0.0f;

                if (speed > cfg_.punch_speed_threshold) {
                    conf = std::min(1.0f, speed / (cfg_.punch_speed_threshold * 2.0f));

                    // 角度变化小（直线运动）增加置信度
                    if (angle < 45.0f) {
                        conf = std::min(1.0f, conf * 1.2f);
                    }

                    // === 方向一致性检查 ===
                    if (has_target) {
                        cv::Point2f wrist_dir(lw.x - ls.x, lw.y - ls.y);
                        float wrist_angle = std::atan2(wrist_dir.y, wrist_dir.x) * 180.0f / CV_PI;
                        float target_angle = std::atan2(target_dir.y, target_dir.x) * 180.0f / CV_PI;
                        float angle_diff = std::abs(normalizeAngle(wrist_angle - target_angle));

                        if (angle_diff > cfg_.punch_max_angle_diff) {
                            conf *= 0.3f;  // 方向偏差大，大幅降低置信度
                            LOG_DEBUG_FMT("Track {} left wrist angle diff {:.1f} too large", 
                                         det.track_id, angle_diff);
                        } else {
                            conf *= (1.0f + (cfg_.punch_max_angle_diff - angle_diff) 
                                          / cfg_.punch_max_angle_diff * 0.5f);
                        }
                    }

                    // === 速度突变检测（收拳特征）===
                    if (state.left_wrist_history.size() >= 5) {
                        float recent_speed = calculateSpeed(state.left_wrist_history, 3, 0);
                        float older_speed = calculateSpeed(state.left_wrist_history, 3, 2);
                        if (older_speed > 0 && recent_speed / older_speed < cfg_.punch_min_speed_drop_ratio) {
                            conf *= 1.2f;  // 速度下降，可能是收拳，增加置信度
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
                    if (angle < 45.0f) {
                        conf = std::min(1.0f, conf * 1.2f);
                    }

                    if (has_target) {
                        cv::Point2f wrist_dir(rw.x - rs.x, rw.y - rs.y);
                        float wrist_angle = std::atan2(wrist_dir.y, wrist_dir.x) * 180.0f / CV_PI;
                        float target_angle = std::atan2(target_dir.y, target_dir.x) * 180.0f / CV_PI;
                        float angle_diff = std::abs(normalizeAngle(wrist_angle - target_angle));

                        if (angle_diff > cfg_.punch_max_angle_diff) {
                            conf *= 0.3f;
                        } else {
                            conf *= (1.0f + (cfg_.punch_max_angle_diff - angle_diff) 
                                          / cfg_.punch_max_angle_diff * 0.5f);
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

    // === 双手互斥：双手同时高速运动 → 降低置信度（跑步/挥手）===
    if (left_valid && right_valid && state.left_wrist_history.size() >= 3 
        && state.right_wrist_history.size() >= 3) {
        float left_speed = calculateSpeed(state.left_wrist_history);
        float right_speed = calculateSpeed(state.right_wrist_history);

        if (left_speed > cfg_.punch_speed_threshold && right_speed > cfg_.punch_speed_threshold) {
            best_conf *= 0.5f;  // 双手同时高速，大幅降低置信度
            if (best_conf < 0.5f) {
                is_punch = false;
            }
            LOG_INFO_FMT("Track {} both hands high speed, reduce confidence", det.track_id);
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

// ========== Fall检测（基于关键点姿态）==========
bool FightingDetector::detectFall(
    const core::InferenceResultPacket::BBox& det, 
    TrackState& state) {

    if (!det.has_keypoints) return false;

    const auto& kpts = det.keypoints;
    bool is_fall = false;

    // === 方法1：基于关键点的高度分析 ===
    float head_y = -1, ankle_y = -1;
    bool has_head = false, has_ankle = false;

    // 取头部最高点（最小的y值，因为y轴向下）
    for (int idx : {Config::NOSE, Config::LEFT_EYE, Config::RIGHT_EYE, 
                    Config::LEFT_EAR, Config::RIGHT_EAR}) {
        if (isKeypointValid(kpts[idx])) {
            if (!has_head || kpts[idx].y < head_y) {
                head_y = kpts[idx].y;
                has_head = true;
            }
        }
    }

    // 取脚踝最低点（最大的y值）
    for (int idx : {Config::LEFT_ANKLE, Config::RIGHT_ANKLE}) {
        if (isKeypointValid(kpts[idx])) {
            if (!has_ankle || kpts[idx].y > ankle_y) {
                ankle_y = kpts[idx].y;
                has_ankle = true;
            }
        }
    }

    // 关键点身体高度远小于检测框高度 → 身体横躺
    if (has_head && has_ankle) {
        float body_height = ankle_y - head_y;
        if (body_height < det.h * cfg_.fall_body_height_ratio) {
            is_fall = true;
        }
    }

    // === 方法2：肩-臀连线角度 ===
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

            if (dx > 1.0f) {  // 避免除零
                float body_angle = std::atan2(dy, dx) * 180.0f / CV_PI;

                // 身体接近水平（角度小）且臀部不低于肩部太多（不是弯腰）
                if (body_angle < cfg_.fall_torso_angle_threshold && 
                    hip_y < shoulder_y + det.h * 0.3f) {
                    is_fall = true;
                }
            }
        }
    }

    // === 方法3：宽高比作为辅助（仅当缺少上半身关键点时）===
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
        state.fall_frame_count = std::max(0, state.fall_frame_count - 1);
        return false;
    }
}

// ========== Interaction检测 ==========
std::vector<FightingEvent> FightingDetector::detectInteractions(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    std::vector<FightingEvent> events;

    for (size_t i = 0; i < detections.size(); ++i) {
        for (size_t j = i + 1; j < detections.size(); ++j) {
            const auto& d1 = detections[i];
            const auto& d2 = detections[j];

            if (d1.track_id < 0 || d2.track_id < 0) continue;
            if (!d1.has_keypoints || !d2.has_keypoints) continue;

            // 中心距离
            float center_dx = (d2.x + d2.w/2) - (d1.x + d1.w/2);
            float center_dy = (d2.y + d2.h/2) - (d1.y + d1.h/2);
            float center_dist = std::sqrt(center_dx*center_dx + center_dy*center_dy);

            if (center_dist > cfg_.interaction_distance_threshold * 2.0f) continue;

            // 相对运动分析
            auto& s1 = tracks_[d1.track_id];
            auto& s2 = tracks_[d2.track_id];

            cv::Point2f v1 = calculateVelocity(s1);
            cv::Point2f v2 = calculateVelocity(s2);
            cv::Point2f relative_v(v2.x - v1.x, v2.y - v1.y);
            float relative_speed = std::sqrt(relative_v.x*relative_v.x + relative_v.y*relative_v.y);

            // 连线方向归一化
            float conn_len = std::sqrt(center_dx*center_dx + center_dy*center_dy);
            cv::Point2f connection(0, 0);
            if (conn_len > 0) {
                connection = cv::Point2f(center_dx/conn_len, center_dy/conn_len);
            }

            // 相向运动检测
            float dot_product = relative_v.x * connection.x + relative_v.y * connection.y;
            bool approaching = dot_product < cfg_.interaction_approach_dot_threshold;

            // 面对面姿态
            bool facing = checkFacingEachOther(d1, d2);

            // 动态距离阈值
            float effective_threshold = cfg_.interaction_distance_threshold;
            if (approaching && facing) {
                effective_threshold *= 1.2f;
            } else if (!approaching && !facing) {
                effective_threshold *= 0.6f;
            }

            // 关键点接触检测
            const auto& k1 = d1.keypoints;
            const auto& k2 = d2.keypoints;
            float min_dist = std::numeric_limits<float>::max();
            std::vector<std::string> contacts;

            // 手腕距离
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
                LOG_INFO_FMT("Interaction detected between track {} and {}, contact: {}, approach: {}, facing: {}, conf: {:.2f}", 
                             d1.track_id, d2.track_id, evt.contact_parts, has_approach, facing, evt.confidence);
                events.push_back(evt);
            }
        }
    }

    return events;
}


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

// ========== 新增辅助函数 ==========

float FightingDetector::calculateArmExtension(
    const core::InferenceResultPacket::KeyPoint& shoulder,
    const core::InferenceResultPacket::KeyPoint& elbow,
    const core::InferenceResultPacket::KeyPoint& wrist) {

    float upper_arm = euclideanDistance(shoulder, elbow);
    float forearm = euclideanDistance(elbow, wrist);
    float full_arm = euclideanDistance(shoulder, wrist);

    if (upper_arm + forearm < 1.0f) return 0.0f;

    // 伸展度 = 实际肩-腕距离 / (上臂+前臂) 理论最大距离
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

    bool d1_has_nose = isKeypointValid(k1[Config::NOSE]);
    bool d2_has_nose = isKeypointValid(k2[Config::NOSE]);

    if (!d1_has_nose || !d2_has_nose) return false;

    float c1x = d1.x + d1.w/2;
    float c2x = d2.x + d2.w/2;

    bool d1_facing_right = k1[Config::NOSE].x > c1x;
    bool d2_facing_left = k2[Config::NOSE].x < c2x;

    if (c1x < c2x) {
        return d1_facing_right && d2_facing_left;
    } else {
        return !d1_facing_right && !d2_facing_left;
    }
}

float FightingDetector::normalizeAngle(float angle) {
    while (angle > 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
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

    for (auto it = fighting_tracks_.begin(); it != fighting_tracks_.end();) {
        if (active_set.find(it->first) == active_set.end()) {
            it = fighting_tracks_.erase(it);
        } else {
            ++it;
        }
    }
}