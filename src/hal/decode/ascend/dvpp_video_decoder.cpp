// src/hal/decode/ascend/dvpp_video_decoder.cpp
// 华为 Ascend DVPP VDEC 硬件解码——见头文件说明
#include "dvpp_video_decoder.h"
#include "ai_stream/hal/video_decoder_factory.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unistd.h>

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace ai_stream {
namespace hal {
namespace {

std::atomic<uint32_t> g_next_channel_id{10};

constexpr int kDefaultWidth = 1920;
constexpr int kDefaultHeight = 1080;

bool hasAnnexbStartCode(const uint8_t* d, int size) {
    if (!d || size < 3) return false;
    if (size >= 4 && d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 1) return true;
    return d[0] == 0 && d[1] == 0 && d[2] == 1;
}

int alignUp(int v, int a) {
    return (v + a - 1) / a * a;
}

} // namespace

DvppVideoDecoder::~DvppVideoDecoder() {
    cleanup();
}

bool DvppVideoDecoder::isAvailable() const {
    return ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

bool DvppVideoDecoder::init(const std::string& codec_name,
                            const uint8_t* extradata,
                            int extradata_size) {
    codec_name_ = codec_name;
    if (extradata && extradata_size > 0) {
        extradata_.assign(extradata, extradata + extradata_size);
    }
    LOG_INFO_FMT("[DvppVideoDecoder] Initializing codec: {} (extradata {} bytes)",
                 codec_name_, extradata_.size());
    initialized_ = initDvpp();
    return initialized_;
}

bool DvppVideoDecoder::initDvpp() {
    if (!ensureAscendAclInitialized()) {
        LOG_ERROR("[DvppVideoDecoder] aclInit failed");
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoDecoder] aclrtSetDevice failed");
        return false;
    }
    if (aclrtCreateContext(&context_, 0) != ACL_SUCCESS ||
        aclrtSetCurrentContext(context_) != ACL_SUCCESS ||
        aclrtCreateStream(&stream_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoDecoder] create context/stream failed");
        return false;
    }

    if (codec_name_ == "h265" || codec_name_ == "hevc" || codec_name_ == "H265") {
        en_type_ = H265_MAIN_LEVEL;
    } else {
        en_type_ = H264_HIGH_LEVEL;
    }

    parseExtradata();

    // 输出缓冲按源分辨率分配；未知时用 1080p 兜底。DVPP 的 picDesc 只传
    // data/size/format，宽高由码流 SPS 决定（与官方 vdec sample 一致）。
    out_width_ = source_width_ > 0 ? source_width_ : kDefaultWidth;
    out_height_ = source_height_ > 0 ? source_height_ : kDefaultHeight;
    const int aw = alignUp(out_width_, 16);
    const int ah = alignUp(out_height_, 2);
    output_buffer_size_ = static_cast<size_t>(aw) * ah * 3 / 2;

    report_running_.store(true);
    if (pthread_create(&report_thread_, nullptr, reportThreadEntry, this) != 0) {
        LOG_ERROR("[DvppVideoDecoder] create report thread failed");
        return false;
    }

    channel_ = aclvdecCreateChannelDesc();
    if (!channel_) {
        LOG_ERROR("[DvppVideoDecoder] aclvdecCreateChannelDesc failed");
        return false;
    }
    channel_id_ = g_next_channel_id.fetch_add(1) % 16;
    if (aclvdecSetChannelDescChannelId(channel_, channel_id_) != ACL_SUCCESS ||
        aclvdecSetChannelDescThreadId(channel_,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(report_thread_))) != ACL_SUCCESS ||
        aclvdecSetChannelDescCallback(channel_, &DvppVideoDecoder::vdecCallback) != ACL_SUCCESS ||
        aclvdecSetChannelDescEnType(channel_, en_type_) != ACL_SUCCESS ||
        aclvdecSetChannelDescOutPicFormat(channel_, PIXEL_FORMAT_YUV_SEMIPLANAR_420) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoDecoder] set channel desc failed");
        return false;
    }
    if (aclvdecCreateChannel(channel_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoDecoder] aclvdecCreateChannel failed");
        return false;
    }

    LOG_INFO_FMT("[DvppVideoDecoder] DVPP VDEC ready: channel={} out={}x{} buffer={}B",
                 channel_id_, out_width_, out_height_, output_buffer_size_);
    return true;
}

bool DvppVideoDecoder::parseExtradata() {
    if (extradata_.empty()) return true;

    if (hasAnnexbStartCode(extradata_.data(), static_cast<int>(extradata_.size()))) {
        sps_pps_annexb_ = extradata_;
        return true;
    }

    // AVCDecoderConfigurationRecord（H.264）
    if (extradata_.size() < 7 || extradata_[0] != 1) {
        LOG_WARN("[DvppVideoDecoder] Extradata is not AnnexB/AVCC H.264; relying on in-band SPS/PPS");
        return true;
    }
    nal_length_size_ = (extradata_[4] & 0x03) + 1;
    int num_sps = extradata_[5] & 0x1f;
    size_t off = 6;
    auto appendNal = [&](const uint8_t* p, size_t len) {
        if (!p || len == 0) return;
        static const uint8_t sc[4] = {0, 0, 0, 1};
        sps_pps_annexb_.insert(sps_pps_annexb_.end(), sc, sc + 4);
        sps_pps_annexb_.insert(sps_pps_annexb_.end(), p, p + len);
    };
    for (int i = 0; i < num_sps && off + 2 <= extradata_.size(); ++i) {
        size_t len = (static_cast<size_t>(extradata_[off]) << 8) | extradata_[off + 1];
        off += 2;
        if (off + len > extradata_.size()) return true;
        appendNal(extradata_.data() + off, len);
        off += len;
    }
    if (off < extradata_.size()) {
        int num_pps = extradata_[off++];
        for (int i = 0; i < num_pps && off + 2 <= extradata_.size(); ++i) {
            size_t len = (static_cast<size_t>(extradata_[off]) << 8) | extradata_[off + 1];
            off += 2;
            if (off + len > extradata_.size()) break;
            appendNal(extradata_.data() + off, len);
            off += len;
        }
    }
    LOG_INFO_FMT("[DvppVideoDecoder] Parsed AVCC extradata: nal_len={} sps_pps={}B",
                 nal_length_size_, sps_pps_annexb_.size());
    return true;
}

bool DvppVideoDecoder::toAnnexb(const uint8_t* data, int size,
                                std::vector<uint8_t>& out) const {
    out.clear();
    if (!data || size <= 0) return false;
    if (hasAnnexbStartCode(data, size)) {
        out.assign(data, data + size);
        return true;
    }
    if (nal_length_size_ < 1 || nal_length_size_ > 4) return false;

    size_t pos = 0;
    static const uint8_t sc[4] = {0, 0, 0, 1};
    while (pos + static_cast<size_t>(nal_length_size_) <= static_cast<size_t>(size)) {
        uint32_t len = 0;
        for (int i = 0; i < nal_length_size_; ++i) {
            len = (len << 8) | data[pos++];
        }
        if (len == 0 || pos + len > static_cast<size_t>(size)) break;
        out.insert(out.end(), sc, sc + 4);
        out.insert(out.end(), data + pos, data + pos + len);
        pos += len;
    }
    return !out.empty();
}

bool DvppVideoDecoder::isIdr(const std::vector<uint8_t>& annexb) const {
    if (annexb.size() < 5) return false;
    size_t sc = 0;
    if (annexb[0] == 0 && annexb[1] == 0 && annexb[2] == 0 && annexb[3] == 1) {
        sc = 4;
    } else if (annexb[0] == 0 && annexb[1] == 0 && annexb[2] == 1) {
        sc = 3;
    } else {
        return false;
    }
    if (sc >= annexb.size()) return false;
    if (codec_name_ == "h265" || codec_name_ == "hevc" || codec_name_ == "H265") {
        const int nal_type = (annexb[sc] >> 1) & 0x3f;
        return nal_type == 19 || nal_type == 20;
    }
    return (annexb[sc] & 0x1f) == 5;
}

bool DvppVideoDecoder::decode(const uint8_t* packet_data, int packet_size,
                              DecodedFrame& frame) {
    if (!initialized_ || !channel_) {
        LOG_ERROR("[DvppVideoDecoder] Not initialized");
        return false;
    }
    if (aclrtSetCurrentContext(context_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoDecoder] set context failed");
        return false;
    }

    std::vector<uint8_t> annexb;
    if (!toAnnexb(packet_data, packet_size, annexb)) {
        LOG_WARN("[DvppVideoDecoder] Packet is neither AnnexB nor AVCC");
        return false;
    }

    // IDR 前补 SPS/PPS（AVCC 的 extradata 不在码流里）
    if ((!sps_pps_sent_ || isIdr(annexb)) && !sps_pps_annexb_.empty()) {
        std::vector<uint8_t> with_hdr;
        with_hdr.reserve(sps_pps_annexb_.size() + annexb.size());
        with_hdr.insert(with_hdr.end(), sps_pps_annexb_.begin(), sps_pps_annexb_.end());
        with_hdr.insert(with_hdr.end(), annexb.begin(), annexb.end());
        annexb.swap(with_hdr);
        sps_pps_sent_ = true;
    }

    void* in_dev = nullptr;
    if (acldvppMalloc(&in_dev, annexb.size()) != ACL_SUCCESS || !in_dev) {
        LOG_ERROR("[DvppVideoDecoder] acldvppMalloc input failed");
        return false;
    }
    if (aclrtMemcpy(in_dev, annexb.size(), annexb.data(), annexb.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        acldvppFree(in_dev);
        LOG_ERROR("[DvppVideoDecoder] H2D input failed");
        return false;
    }

    acldvppStreamDesc* in_desc = acldvppCreateStreamDesc();
    acldvppPicDesc* out_desc = acldvppCreatePicDesc();
    void* out_dev = nullptr;
    if (!in_desc || !out_desc ||
        acldvppMalloc(&out_dev, output_buffer_size_) != ACL_SUCCESS || !out_dev) {
        if (in_desc) acldvppDestroyStreamDesc(in_desc);
        if (out_desc) acldvppDestroyPicDesc(out_desc);
        acldvppFree(in_dev);
        if (out_dev) acldvppFree(out_dev);
        LOG_ERROR("[DvppVideoDecoder] create descs/output buffer failed");
        return false;
    }

    acldvppSetStreamDescData(in_desc, in_dev);
    acldvppSetStreamDescSize(in_desc, static_cast<uint32_t>(annexb.size()));
    acldvppSetStreamDescFormat(in_desc, en_type_);
    acldvppSetPicDescData(out_desc, out_dev);
    acldvppSetPicDescSize(out_desc, static_cast<uint32_t>(output_buffer_size_));
    acldvppSetPicDescFormat(out_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    aclError ret = aclvdecSendFrame(channel_, in_desc, out_desc, nullptr, this);
    if (ret != ACL_SUCCESS) {
        acldvppFree(in_dev);
        acldvppFree(out_dev);
        acldvppDestroyStreamDesc(in_desc);
        acldvppDestroyPicDesc(out_desc);
        LOG_ERROR_FMT("[DvppVideoDecoder] aclvdecSendFrame failed: {}", static_cast<int>(ret));
        return false;
    }
    ++sent_packets_;

    std::unique_lock<std::mutex> lock(frame_mutex_);
    if (!frame_cv_.wait_for(lock, std::chrono::milliseconds(500),
                            [&] { return !frames_.empty(); })) {
        ++timeout_count_;
        return false;
    }
    ++recv_frames_;
    frame_buf_ = std::move(frames_.front());
    frames_.pop_front();
    const int w = frame_width_;
    const int h = frame_height_;
    lock.unlock();

    if (frame_buf_.size() < static_cast<size_t>(w) * h * 3 / 2) {
        LOG_ERROR("[DvppVideoDecoder] Decoded frame too small");
        return false;
    }

    frame.data = frame_buf_.data();
    frame.width = w;
    frame.height = h;
    frame.pitch = w;
    frame.format = AV_PIX_FMT_NV12;
    frame.owns_data = false;  // 借用至下次 decode
    frame.data_uv = frame_buf_.data() + static_cast<size_t>(w) * h;
    frame.pitch_uv = w;
    frame.is_gpu = false;
    frame.d_data = nullptr;
    frame.d_data_uv = nullptr;
    return true;
}

void* DvppVideoDecoder::reportThreadEntry(void* arg) {
    static_cast<DvppVideoDecoder*>(arg)->reportLoop();
    return nullptr;
}

void DvppVideoDecoder::reportLoop() {
    if (aclrtSetCurrentContext(context_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppVideoDecoder] report thread set context failed");
        return;
    }
    while (report_running_.load()) {
        aclrtProcessReport(1000);
    }
}

void DvppVideoDecoder::vdecCallback(acldvppStreamDesc* input,
                                    acldvppPicDesc* output, void* userData) {
    auto* self = static_cast<DvppVideoDecoder*>(userData);
    if (input) {
        void* d = acldvppGetStreamDescData(input);
        if (d) acldvppFree(d);
        acldvppDestroyStreamDesc(input);
    }
    if (!output) return;

    const uint32_t ret_code = acldvppGetPicDescRetCode(output);
    if (ret_code != 0) {
        // 非 0 可能只是参考帧/非显示帧告警，仍尝试取图；数据无效时再丢弃
        LOG_DEBUG_FMT("[DvppVideoDecoder] VDEC frame ret={}", ret_code);
    }

    const uint32_t w = acldvppGetPicDescWidth(output);
    const uint32_t h = acldvppGetPicDescHeight(output);
    const uint32_t ws = acldvppGetPicDescWidthStride(output);
    const uint32_t hs = acldvppGetPicDescHeightStride(output);
    const uint32_t size = acldvppGetPicDescSize(output);
    void* data = acldvppGetPicDescData(output);

    if (data && w > 0 && h > 0 && ws >= w && hs >= h) {
        const size_t need = static_cast<size_t>(ws) * hs * 3 / 2;
        const size_t copy_size = std::min<size_t>(need, size);
        std::vector<uint8_t> raw(copy_size);
        if (aclrtMemcpy(raw.data(), copy_size, data, copy_size,
                        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
            std::vector<uint8_t> tight(static_cast<size_t>(w) * h * 3 / 2);
            for (uint32_t y = 0; y < h; ++y) {
                std::memcpy(tight.data() + static_cast<size_t>(y) * w,
                            raw.data() + static_cast<size_t>(y) * ws, w);
            }
            const size_t uv_src = static_cast<size_t>(ws) * hs;
            for (uint32_t y = 0; y < h / 2; ++y) {
                std::memcpy(tight.data() + static_cast<size_t>(w) * h + static_cast<size_t>(y) * w,
                            raw.data() + uv_src + static_cast<size_t>(y) * ws, w);
            }
            {
                std::lock_guard<std::mutex> lk(self->frame_mutex_);
                self->frame_width_ = static_cast<int>(w);
                self->frame_height_ = static_cast<int>(h);
                self->frames_.push_back(std::move(tight));
                if (self->frames_.size() > 32) self->frames_.pop_front();
            }
            self->frame_cv_.notify_one();
        }
    }

    if (data) acldvppFree(data);
    acldvppDestroyPicDesc(output);
}

void DvppVideoDecoder::release() {
    cleanup();
}

void DvppVideoDecoder::cleanup() {
    if (initialized_) {
        size_t queued = 0;
        {
            std::lock_guard<std::mutex> lk(frame_mutex_);
            queued = frames_.size();
        }
        LOG_INFO_FMT("[DvppVideoDecoder] Stats: sent={} recv={} timeouts={} queued={}",
                     sent_packets_, recv_frames_, timeout_count_, queued);
    }
    if (channel_) {
        aclvdecDestroyChannel(channel_);
    }
    report_running_.store(false);
    if (report_thread_) {
        pthread_join(report_thread_, nullptr);
        report_thread_ = 0;
    }
    if (channel_) {
        aclvdecDestroyChannelDesc(channel_);
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
        std::lock_guard<std::mutex> lk(frame_mutex_);
        frames_.clear();
    }
    frame_buf_.clear();
    initialized_ = false;
}

REGISTER_VIDEO_DECODER(VideoDecoderBackend::DVPP, DvppVideoDecoder)

} // namespace hal
} // namespace ai_stream
