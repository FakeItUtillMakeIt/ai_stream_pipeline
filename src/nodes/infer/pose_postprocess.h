// src/nodes/infer/pose_postprocess.h
// Pose 节点共享的关键点解码逻辑：从模型原始输出 [num_persons, 8400, 56]
// 解码每人的关键点并写回 InferenceResultPacket。
// 仅供 pose_infer / cuda_pose_infer 内部复用，不属于公开 API。
#pragma once

#include "ai_stream/core/packet.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cmath>
#include <opencv2/core/types.hpp>
#include <vector>

namespace ai_stream {
namespace nodes {
namespace pose_postprocess {

constexpr int NUM_CANDIDATES = 8400;  // 80*80 + 40*40 + 20*20
constexpr int POSE_DIM = 56;          // 4 box + 1 score + 51 kpts
constexpr int NUM_KEYPOINTS = 17;

/**
 * @brief 单帧内多人 batch 关键点解码
 * @param output_host 模型输出 [num_persons, NUM_CANDIDATES, POSE_DIM]
 * @param letterbox_params [scale, pad_x, pad_y] x num_persons
 */
inline void decodeFrame(
    std::shared_ptr<core::InferenceResultPacket> packet,
    const std::vector<int>& person_indices,
    int num_persons,
    const float* output_host,
    const std::vector<float>& letterbox_params,
    float conf_thresh,
    float kpt_conf_thresh,
    const char* log_tag)
{
    packet->pose_results.clear();
    packet->pose_results.reserve(num_persons);

    for (int p = 0; p < num_persons; ++p) {
        int det_idx = person_indices[p];
        auto& det = packet->detections[det_idx];

        float scale = letterbox_params[p * 3 + 0];
        float pad_x = letterbox_params[p * 3 + 1];
        float pad_y = letterbox_params[p * 3 + 2];

        // 该人的输出起始地址: [p, 0, 0]
        const float* person_output =
            output_host + static_cast<size_t>(p) * NUM_CANDIDATES * POSE_DIM;

        // 遍历候选，找 person score 最高的
        float best_score = -1.0f;
        int best_idx = 0;
        for (int i = 0; i < NUM_CANDIDATES; ++i) {
            float score = person_output[i * POSE_DIM + 4];
            score = 1.0f / (1.0f + std::exp(-score));
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_score < conf_thresh) {
            LOG_INFO_FMT("[{}] Person {} best score {:.3f} below threshold {}, skipping",
                         log_tag, det_idx, best_score, conf_thresh);
            continue;
        }

        const float* best_pred = person_output + best_idx * POSE_DIM;

        core::InferenceResultPacket::PoseResult pose;
        pose.person_score = best_score;
        pose.matched_det_idx = det_idx;

        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            float kx = best_pred[5 + k * 3 + 0];
            float ky = best_pred[5 + k * 3 + 1];
            float kconf = best_pred[5 + k * 3 + 2];

            kconf = 1.0f / (1.0f + std::exp(-kconf));

            float orig_kx = (kx - pad_x) / scale + det.x;
            float orig_ky = (ky - pad_y) / scale + det.y;

            pose.keypoints[k] = {orig_kx, orig_ky, kconf, kconf > kpt_conf_thresh};

            det.has_keypoints = true;
            det.keypoints[k] = pose.keypoints[k];
            det.keypoints_conf = kconf;
        }

        pose.person_box = cv::Rect2f(det.x, det.y, det.w, det.h);
        packet->pose_results.push_back(pose);

        LOG_INFO_FMT("[{}] Person {}: score={:.3f}, kpts_visible={}/17, scale={:.3f}, pad=({:.1f},{:.1f})",
                     log_tag, det_idx, best_score,
                     std::count_if(pose.keypoints.begin(), pose.keypoints.end(),
                                   [](const auto& k){ return k.visible; }),
                     scale, pad_x, pad_y);
    }
}

} // namespace pose_postprocess
} // namespace nodes
} // namespace ai_stream
