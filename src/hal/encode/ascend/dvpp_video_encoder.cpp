// src/hal/encode/ascend/dvpp_video_encoder.cpp
#include "dvpp_video_encoder.h"
#include "ai_stream/hal/i_video_encoder.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cstring>
#include <unistd.h>

namespace ai_stream {
namespace hal {
namespace {

int alignUp(int v, int a) {
    return (v + a - 1) / a * a;
}

// 查找下一个 AnnexB 起始码；sc.pos 为起始码位置，sc.len 为 3 或 4
struct StartCode {
    size_t pos = 0;
    size_t len = 0;
};

bool findStartCode(const uint8_t* data, size_t size, size_t from, StartCode& sc) {
    for (size_t i = from; i + 3 <= size; ++i) {
        if (i + 4 <= size && data[i] == 0 && data[i + 1] == 0 &&
            data[i + 2] == 0 && data[i + 3] == 1) {
            sc = {i, 4};
            return true;
        }
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
            sc = {i, 3};
            return true;
        }
    }
    return false;
}

} // namespace

DvppVideoEncoder::~DvppVideoEncoder() {
    cleanup();
}

bool DvppVideoEncoder::isAvailable() const {
    return ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

bool DvppVideoEncoder::open(const VideoEncoderConfig& config) {
    if (opened_) return true;

    width_ = config.width;
    height_ = config.height;
    if (width_ <= 0 || height_ <= 0) {
        LOG_ERROR("[DvppVideoEncoder] Invalid frame size");
        return false;
    }
    fps_ = config.fps > 0 ? config.fps : 25;
    gop_ = config.gop > 0 ? config.gop : fps_;
    bitrate_kbps_ = config.bitrate_kbps > 0 ? config.bitrate_kbps : 4000;
    aligned_w_ = alignUp(width_, 16);
    aligned_h_ = alignUp(height_, 16);
    frame_size_ = static_cast<size_t>(aligned_w_) * aligned_h_ * 3 / 2;

    if (!initVenc()) {
        cleanup();
        return false;
    }
    opened_ = true;

    // VENC 的 SPS/PPS 随首个 IDR 输出；先编码一帧灰图提取 extradata，
    // 供 EncoderBase 在写容器头之前使用。该帧丢弃，首个真实帧强制 IDR。
    if (!primeExtradata()) {
        LOG_WARN("[DvppVideoEncoder] Failed to prime SPS/PPS; muxer may lack extradata");
    }

    LOG_INFO_FMT("[DvppVideoEncoder] Opened: {}x{} (aligned {}x{}) @ {}kbps, gop={}",
                 width_, height_, aligned_w_, aligned_h_, bitrate_kbps_, gop_);
    return true;
}

bool DvppVideoEncoder::initVenc() {
    if (!ensureAscendAclInitialized()) {
        LOG_ERROR("[DvppVideoEncoder] aclInit failed");
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateContext(&context_, 0) != ACL_SUCCESS ||
        aclrtSetCurrentContext(context_) != ACL_SUCCESS ||
        aclrtCreateStream(&stream_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoEncoder] create context/stream failed");
        return false;
    }

    report_running_.store(true);
    if (pthread_create(&report_thread_, nullptr, reportThreadEntry, this) != 0) {
        LOG_ERROR("[DvppVideoEncoder] create report thread failed");
        return false;
    }

    channel_ = aclvencCreateChannelDesc();
    if (!channel_) {
        LOG_ERROR("[DvppVideoEncoder] aclvencCreateChannelDesc failed");
        return false;
    }
    aclError ret = aclvencSetChannelDescThreadId(
        channel_, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(report_thread_)));
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescCallback(channel_, &DvppVideoEncoder::vencCallback);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescEnType(channel_, H264_HIGH_LEVEL);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescPicFormat(channel_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescPicWidth(channel_, width_);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescPicHeight(channel_, height_);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescKeyFrameInterval(channel_, gop_);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescRcMode(channel_, 2);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescMaxBitRate(channel_, bitrate_kbps_);
    if (ret == ACL_SUCCESS) ret = aclvencSetChannelDescSrcRate(channel_, fps_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[DvppVideoEncoder] set channel desc failed: {}", static_cast<int>(ret));
        return false;
    }
    if (aclvencCreateChannel(channel_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoEncoder] aclvencCreateChannel failed");
        return false;
    }
    frame_config_ = aclvencCreateFrameConfig();
    if (!frame_config_) {
        LOG_ERROR("[DvppVideoEncoder] aclvencCreateFrameConfig failed");
        return false;
    }

    // 复用同一块输入 device buffer / picDesc（sample 也复用 vpcInputDesc）；
    // 避免每帧 acldvppMalloc，且保证异步回调期间输入内存始终有效。
    if (acldvppMalloc(&input_dev_, frame_size_) != ACL_SUCCESS || !input_dev_) {
        LOG_ERROR("[DvppVideoEncoder] acldvppMalloc input buffer failed");
        return false;
    }
    input_pic_ = acldvppCreatePicDesc();
    if (!input_pic_) {
        LOG_ERROR("[DvppVideoEncoder] acldvppCreatePicDesc failed");
        return false;
    }
    acldvppSetPicDescData(input_pic_, input_dev_);
    acldvppSetPicDescSize(input_pic_, static_cast<uint32_t>(frame_size_));
    acldvppSetPicDescFormat(input_pic_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(input_pic_, width_);
    acldvppSetPicDescHeight(input_pic_, height_);
    acldvppSetPicDescWidthStride(input_pic_, aligned_w_);
    acldvppSetPicDescHeightStride(input_pic_, aligned_h_);
    return true;
}

bool DvppVideoEncoder::primeExtradata() {
    std::vector<uint8_t> gray(static_cast<size_t>(width_) * height_ * 3 / 2, 0);
    std::fill(gray.begin(), gray.begin() + width_ * height_, 16);
    std::fill(gray.begin() + width_ * height_, gray.end(), 128);
    std::vector<EncodedPacket> discard;
    priming_ = true;
    bool ok = encode(gray.data(), gray.size(), 0, discard);
    priming_ = false;
    return ok && !extradata_.empty();
}

void DvppVideoEncoder::fillNv12(const uint8_t* yuv420p,
                                std::vector<uint8_t>& nv12) const {
    nv12.assign(frame_size_, 0);
    const uint8_t* y_plane = yuv420p;
    const uint8_t* u_plane = yuv420p + static_cast<size_t>(width_) * height_;
    const uint8_t* v_plane = u_plane + static_cast<size_t>(width_) * height_ / 4;
    for (int y = 0; y < height_; ++y) {
        std::memcpy(nv12.data() + static_cast<size_t>(y) * aligned_w_,
                    y_plane + static_cast<size_t>(y) * width_, width_);
    }
    uint8_t* uv = nv12.data() + static_cast<size_t>(aligned_w_) * aligned_h_;
    for (int y = 0; y < height_ / 2; ++y) {
        for (int x = 0; x < width_ / 2; ++x) {
            uv[static_cast<size_t>(y) * aligned_w_ + x * 2] =
                u_plane[static_cast<size_t>(y) * width_ / 2 + x];
            uv[static_cast<size_t>(y) * aligned_w_ + x * 2 + 1] =
                v_plane[static_cast<size_t>(y) * width_ / 2 + x];
        }
    }
}

bool DvppVideoEncoder::encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                              std::vector<EncodedPacket>& packets) {
    if (!opened_ || !channel_) return false;
    const size_t need = static_cast<size_t>(width_) * height_ * 3 / 2;
    if (!yuv420p || size < need) return false;
    // 编码器可能在 onStartup 线程 open、在 worker 线程 encode；ACL context 是
    // 线程相关的，必须在使用前切到编码器自己的 context。
    if (aclrtSetCurrentContext(context_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoEncoder] set context failed");
        return false;
    }

    if (!priming_) staging_.clear();

    std::vector<uint8_t> nv12;
    fillNv12(yuv420p, nv12);

    if (aclrtMemcpy(input_dev_, frame_size_, nv12.data(), frame_size_,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoEncoder] H2D failed");
        return false;
    }

    aclvencSetFrameConfigEos(frame_config_, 0);
    aclvencSetFrameConfigForceIFrame(frame_config_,
                                     (priming_ || first_frame_) ? 1 : 0);
    current_pts_.store(pts);

    aclError ret = aclvencSendFrame(channel_, input_pic_, nullptr, frame_config_, this);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[DvppVideoEncoder] aclvencSendFrame failed: {}", static_cast<int>(ret));
        return false;
    }

    std::deque<OutPacket> got;
    {
        std::unique_lock<std::mutex> lock(packet_mutex_);
        if (!packet_cv_.wait_for(lock, std::chrono::milliseconds(1000),
                                 [&] { return !packets_.empty(); })) {
            LOG_WARN("[DvppVideoEncoder] encode callback timeout");
            return false;
        }
        got.swap(packets_);
    }

    for (auto& p : got) {
        staging_.push_back(std::move(p.data));
        auto& buf = staging_.back();
        EncodedPacket out;
        out.data = buf.data();
        out.size = buf.size();
        out.pts = p.pts;
        out.dts = p.pts;
        out.keyframe = p.keyframe;
        packets.push_back(out);
    }
    if (!priming_) first_frame_ = false;
    return true;
}

bool DvppVideoEncoder::flush(std::vector<EncodedPacket>& packets) {
    if (!opened_ || !channel_ || flushed_) return false;
    flushed_ = true;
    staging_.clear();
    if (aclrtSetCurrentContext(context_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoEncoder] set context failed");
        return false;
    }

    aclvencSetFrameConfigForceIFrame(frame_config_, 0);
    aclvencSetFrameConfigEos(frame_config_, 1);
    aclError ret = aclvencSendFrame(channel_, nullptr, nullptr, frame_config_, this);
    if (ret != ACL_SUCCESS) {
        LOG_WARN_FMT("[DvppVideoEncoder] EOS send failed: {}", static_cast<int>(ret));
        return false;
    }

    std::deque<OutPacket> got;
    {
        std::unique_lock<std::mutex> lock(packet_mutex_);
        packet_cv_.wait_for(lock, std::chrono::milliseconds(500),
                            [&] { return !packets_.empty(); });
        got.swap(packets_);
    }
    for (auto& p : got) {
        staging_.push_back(std::move(p.data));
        auto& buf = staging_.back();
        EncodedPacket out;
        out.data = buf.data();
        out.size = buf.size();
        out.pts = p.pts;
        out.dts = p.pts;
        out.keyframe = p.keyframe;
        packets.push_back(out);
    }
    return true;
}

void DvppVideoEncoder::vencCallback(acldvppPicDesc* /*input*/,
                                    acldvppStreamDesc* output, void* userData) {
    auto* self = static_cast<DvppVideoEncoder*>(userData);
    // 输入 picDesc/device buffer 在编码器生命周期内复用
    if (!output) return;

    void* data = acldvppGetStreamDescData(output);
    const uint32_t size = acldvppGetStreamDescSize(output);
    if (!data || size == 0) return;

    OutPacket p;
    p.data.resize(size);
    if (aclrtMemcpy(p.data.data(), size, data, size,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return;
    }
    p.pts = self->current_pts_.load();
    p.keyframe = isIdr(p.data.data(), p.data.size());
    if (self->extradata_.empty()) {
        std::vector<uint8_t> ed;
        if (extractSpsPps(p.data.data(), p.data.size(), ed)) {
            self->extradata_ = std::move(ed);
        }
    }
    {
        std::lock_guard<std::mutex> lock(self->packet_mutex_);
        self->packets_.push_back(std::move(p));
        if (self->packets_.size() > 16) self->packets_.pop_front();
    }
    self->packet_cv_.notify_one();
}

bool DvppVideoEncoder::extractSpsPps(const uint8_t* data, size_t size,
                                     std::vector<uint8_t>& out) {
    if (!data || size < 4) return false;
    bool has_sps = false, has_pps = false;
    std::vector<uint8_t> tmp;
    StartCode sc;
    size_t pos = 0;
    while (findStartCode(data, size, pos, sc)) {
        const size_t nal = sc.pos + sc.len;
        if (nal >= size) break;
        StartCode next;
        const size_t end = findStartCode(data, size, nal, next) ? next.pos : size;
        if (end <= nal) break;
        const uint8_t nal_type = data[nal] & 0x1f;
        if (nal_type == 7 || nal_type == 8) {
            static const uint8_t start[4] = {0, 0, 0, 1};
            tmp.insert(tmp.end(), start, start + 4);
            tmp.insert(tmp.end(), data + nal, data + end);
            if (nal_type == 7) has_sps = true;
            if (nal_type == 8) has_pps = true;
        }
        pos = end;
    }
    if (has_sps && has_pps) {
        out = std::move(tmp);
        return true;
    }
    return false;
}

bool DvppVideoEncoder::isIdr(const uint8_t* data, size_t size) {
    if (!data || size < 5) return false;
    StartCode sc;
    if (!findStartCode(data, size, 0, sc)) return false;
    const size_t nal = sc.pos + sc.len;
    if (nal >= size) return false;
    return (data[nal] & 0x1f) == 5;
}

void* DvppVideoEncoder::reportThreadEntry(void* arg) {
    static_cast<DvppVideoEncoder*>(arg)->reportLoop();
    return nullptr;
}

void DvppVideoEncoder::reportLoop() {
    if (aclrtSetCurrentContext(context_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoEncoder] report thread set context failed");
        return;
    }
    while (report_running_.load()) {
        aclrtProcessReport(1000);
    }
}

void DvppVideoEncoder::cleanup() {
    if (input_pic_) {
        acldvppDestroyPicDesc(input_pic_);
        input_pic_ = nullptr;
    }
    if (input_dev_) {
        acldvppFree(input_dev_);
        input_dev_ = nullptr;
    }
    if (channel_) {
        aclvencDestroyChannel(channel_);
    }
    report_running_.store(false);
    if (report_thread_) {
        pthread_join(report_thread_, nullptr);
        report_thread_ = 0;
    }
    if (frame_config_) {
        aclvencDestroyFrameConfig(frame_config_);
        frame_config_ = nullptr;
    }
    if (channel_) {
        aclvencDestroyChannelDesc(channel_);
        channel_ = nullptr;
    }
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(packet_mutex_);
        packets_.clear();
    }
    staging_.clear();
    extradata_.clear();
    opened_ = false;
    first_frame_ = true;
    flushed_ = false;
}

void DvppVideoEncoder::close() {
    cleanup();
}

REGISTER_VIDEO_ENCODER("dvpp_h264", DvppVideoEncoder)

} // namespace hal
} // namespace ai_stream
