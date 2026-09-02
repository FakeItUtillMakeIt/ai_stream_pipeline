// src/hal/h264_extradata.h
// H.264 码流格式小工具：AnnexB → AVCC（长度前缀）
// 供 MPP / FFmpeg 编码后端生成容器封装所需的结构化序列头使用。
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace ai_stream {
namespace hal {

// AnnexB（00 00 01 / 00 00 00 01 起始码）→ AVCC（4 字节大端长度前缀）
inline void annexb_to_avcc(const uint8_t* data, size_t size, std::vector<uint8_t>& out) {
    out.clear();
    if (!data || size == 0) return;
    // 收集 NAL：(payload 起点，起始码位置)
    std::vector<std::pair<size_t, size_t>> nals;
    size_t i = 0;
    while (i + 3 <= size) {
        if (i + 4 <= size && data[i] == 0 && data[i + 1] == 0 &&
            data[i + 2] == 0 && data[i + 3] == 1) {
            nals.emplace_back(i + 4, i);
            i += 4;
        } else if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
            nals.emplace_back(i + 3, i);
            i += 3;
        } else {
            ++i;
        }
    }
    for (size_t n = 0; n < nals.size(); ++n) {
        const size_t payload = nals[n].first;
        const size_t end = (n + 1 < nals.size()) ? nals[n + 1].second : size;
        if (end <= payload) continue;
        const size_t len = end - payload;
        out.push_back(static_cast<uint8_t>((len >> 24) & 0xFF));
        out.push_back(static_cast<uint8_t>((len >> 16) & 0xFF));
        out.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        out.push_back(static_cast<uint8_t>(len & 0xFF));
        out.insert(out.end(), data + payload, data + end);
    }
}

// 判断 AnnexB 包是否为关键帧（首个 NAL 类型 5 = IDR）
inline bool annexb_is_keyframe(const uint8_t* d, size_t size) {
    if (!d || size <= 5) return false;
    size_t sc = 0;
    if (d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 1) sc = 4;
    else if (d[0] == 0 && d[1] == 0 && d[2] == 1) sc = 3;
    else return false;
    return (d[sc] & 0x1F) == 5;
}

} // namespace hal
} // namespace ai_stream
