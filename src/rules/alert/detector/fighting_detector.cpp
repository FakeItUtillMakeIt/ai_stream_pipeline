// src/rules/alert/detector/fighting_detector.cpp

#include "fighting_detector.h"
#include <iostream>
#include <unordered_set>



// ========== 构造函数 ==========
FightingDetector::FightingDetector(const Config& cfg) : cfg_(cfg) {
    fight_history_.clear();
}

// ========== 重置 ==========
void FightingDetector::reset() {
    tracks_.clear();
    fighting_tracks_.clear();
    fight_history_.clear();
}

// ========== 主处理函数 ==========
FightingResult FightingDetector::process(const std::vector<core::InferenceResultPacket::BBox>& detections) {
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

    float total_score = 0.0f;

    // 1. 逐人检测 Punch + Fall
    for (const auto& det : detections) {
        if (det.track_id < 0) continue;
        if (!det.has_keypoints) continue;

        auto& state = tracks_[det.track_id];
        state.last_bbox = cv::Rect2f(det.x, det.y, det.w, det.h);

        // Punch检测
        float punch_conf = 0.0f;
        bool is_punch = detectPunch(det, state, punch_conf);
        if (is_punch) {
            FightingEvent evt;
            evt.type = "punch";
            evt.track_id = det.track_id;
            evt.confidence = punch_conf;
            result.events.push_back(evt);
            total_score += punch_conf * 0.4f;
        }

        // Fall检测
        bool is_fall = detectFall(det, state);
        if (is_fall) {
            FightingEvent evt;
            evt.type = "fall";
            evt.track_id = det.track_id;
            result.events.push_back(evt);
            total_score += 0.3f;
        }
    }

    // 2. 交互检测
    auto interactions = detectInteractions(detections);
    for (auto& inter : interactions) {
        result.events.push_back(inter);
        if (!inter.contact_parts.empty()) {
            total_score += 0.3f;
        }
    }

    // 3. 综合判定
    result.fight_score = std::min(1.0f, total_score);

    for (const auto& det : detections) {
        if (det.track_id < 0) continue;

        bool has_event = false;
        for (const auto& evt : result.events) {
            if (evt.track_id == det.track_id && 
                (evt.type == "punch" || evt.type == "fall")) {
                has_event = true;
                break;
            }
        }

        bool has_interaction = false;
        for (const auto& evt : result.events) {
            if (evt.type == "interaction") {
                for (int tid : evt.involved_track_ids) {
                    if (tid == det.track_id) {
                        has_interaction = true;
                        break;
                    }
                }
            }
        }

        auto& count = fighting_tracks_[det.track_id];
        if (has_event || has_interaction) {
            count++;
        } else {
            count = std::max(0, count - 1);
        }
    }

    bool is_fighting = result.fight_score > cfg_.fight_score_threshold;
    for (const auto& [tid, count] : fighting_tracks_) {
        if (count >= cfg_.fight_frames_threshold) {
            is_fighting = true;
            break;
        }
    }

    result.is_fighting = is_fighting;

    // 历史平滑
    fight_history_.push_back(is_fighting);
    if (fight_history_.size() > MAX_HISTORY) fight_history_.pop_front();

    // 清理不活跃
    cleanupInactive(active_ids);

    return result;
}

// ========== Punch检测 ==========
bool FightingDetector::detectPunch(const core::InferenceResultPacket::BBox& det, TrackState& state, float& out_confidence) {
    const auto& kpts = det.keypoints;

    // 检查手腕关键点
    const core::InferenceResultPacket::KeyPoint& lw = kpts[Config::LEFT_WRIST];
    const core::InferenceResultPacket::KeyPoint& rw = kpts[Config::RIGHT_WRIST];

    bool left_valid = isKeypointValid(lw);
    bool right_valid = isKeypointValid(rw);

    if (!left_valid && !right_valid) {
        out_confidence = 0.0f;
        return false;
    }

    bool is_punch = false;
    float best_conf = 0.0f;

    // 左手
    if (left_valid) {
        cv::Point2f curr(lw.x, lw.y);
        state.left_wrist_history.push_back(curr);
        while (state.left_wrist_history.size() > static_cast<size_t>(cfg_.history_frames)) {
            state.left_wrist_history.pop_front();
        }

        if (state.left_wrist_history.size() >= 3) {
            float speed = calculateSpeed(state.left_wrist_history);
            float angle = calculateAngleChange(state.left_wrist_history);

            float conf = 0.0f;
            if (speed > cfg_.punch_speed_threshold) {
                conf = std::min(1.0f, speed / (cfg_.punch_speed_threshold * 2.0f));
                if (angle < 45.0f) {
                    conf = std::min(1.0f, conf * 1.2f);
                }
            }

            if (conf > 0.5f) {
                is_punch = true;
                best_conf = std::max(best_conf, conf);
            }
        }
    }

    // 右手
    if (right_valid) {
        cv::Point2f curr(rw.x, rw.y);
        state.right_wrist_history.push_back(curr);
        while (state.right_wrist_history.size() > static_cast<size_t>(cfg_.history_frames)) {
            state.right_wrist_history.pop_front();
        }

        if (state.right_wrist_history.size() >= 3) {
            float speed = calculateSpeed(state.right_wrist_history);
            float angle = calculateAngleChange(state.right_wrist_history);

            float conf = 0.0f;
            if (speed > cfg_.punch_speed_threshold) {
                conf = std::min(1.0f, speed / (cfg_.punch_speed_threshold * 2.0f));
                if (angle < 45.0f) {
                    conf = std::min(1.0f, conf * 1.2f);
                }
            }

            if (conf > 0.5f) {
                is_punch = true;
                best_conf = std::max(best_conf, conf);
            }
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

// ========== Fall检测 ==========
bool FightingDetector::detectFall(const core::InferenceResultPacket::BBox& det, TrackState& state) {
    if (det.w <= 0 || det.h <= 0) return false;

    float aspect = det.w / det.h;
    bool is_fall = aspect > cfg_.fall_aspect_ratio_threshold;

    if (is_fall) {
        state.fall_frame_count++;
    } else {
        state.fall_frame_count = std::max(0, state.fall_frame_count - 1);
    }

    return is_fall;
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

            const auto& k1 = d1.keypoints;
            const auto& k2 = d2.keypoints;

            float min_dist = std::numeric_limits<float>::max();
            std::vector<std::string> contacts;

            // 检查手腕距离
            if (isKeypointValid(k1[Config::LEFT_WRIST]) && isKeypointValid(k2[Config::LEFT_WRIST])) {
                float d = euclideanDistance(k1[Config::LEFT_WRIST], k2[Config::LEFT_WRIST]);
                if (d < min_dist) min_dist = d;
                if (d < cfg_.interaction_distance_threshold) contacts.push_back("wrist");
            }
            if (isKeypointValid(k1[Config::RIGHT_WRIST]) && isKeypointValid(k2[Config::RIGHT_WRIST])) {
                float d = euclideanDistance(k1[Config::RIGHT_WRIST], k2[Config::RIGHT_WRIST]);
                if (d < min_dist) min_dist = d;
                if (d < cfg_.interaction_distance_threshold) contacts.push_back("wrist");
            }
            if (isKeypointValid(k1[Config::LEFT_WRIST]) && isKeypointValid(k2[Config::RIGHT_WRIST])) {
                float d = euclideanDistance(k1[Config::LEFT_WRIST], k2[Config::RIGHT_WRIST]);
                if (d < min_dist) min_dist = d;
                if (d < cfg_.interaction_distance_threshold) contacts.push_back("wrist");
            }
            if (isKeypointValid(k1[Config::RIGHT_WRIST]) && isKeypointValid(k2[Config::LEFT_WRIST])) {
                float d = euclideanDistance(k1[Config::RIGHT_WRIST], k2[Config::LEFT_WRIST]);
                if (d < min_dist) min_dist = d;
                if (d < cfg_.interaction_distance_threshold) contacts.push_back("wrist");
            }

            // 检查头部距离
            if (isKeypointValid(k1[Config::NOSE]) && isKeypointValid(k2[Config::NOSE])) {
                float d = euclideanDistance(k1[Config::NOSE], k2[Config::NOSE]);
                if (d < min_dist) min_dist = d;
                if (d < cfg_.interaction_distance_threshold) contacts.push_back("head");
            }

            if (min_dist < cfg_.interaction_distance_threshold * cfg_.interaction_near_multiplier) {
                FightingEvent evt;
                evt.type = "interaction";
                evt.involved_track_ids = {d1.track_id, d2.track_id};
                evt.contact_parts = contacts.empty() ? "near" : contacts[0];
                events.push_back(evt);
            }
        }
    }

    return events;
}

// ========== 辅助函数 ==========
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

float FightingDetector::euclideanDistance(const core::InferenceResultPacket::KeyPoint& a, const core::InferenceResultPacket::KeyPoint& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

bool FightingDetector::isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp) {
    return kp.visible && kp.confidence > 0.0f;
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
