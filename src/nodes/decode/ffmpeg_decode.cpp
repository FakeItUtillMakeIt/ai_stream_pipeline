// src/nodes/decode/ffmpeg_decode.cpp
#include "utils/time_util.h"
// 解码节点——使用 VideoCodecFactory，支持多后端（NVDEC/MPP/DVPP/FFmpeg）
#include "ffmpeg_decode.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <sstream>
#include <iomanip>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace fs = std::filesystem;

namespace ai_stream {
namespace nodes {

FFmpegDecodeNode::FFmpegDecodeNode() : core::QueuedNode<IDecodeNode>("FFmpegDecode") {
    LOG_DEBUG_FMT("[FFmpegDecode] Constructor");
}

FFmpegDecodeNode::~FFmpegDecodeNode() {
    stop();
    LOG_DEBUG_FMT("[FFmpegDecode] Destructor");
}

void FFmpegDecodeNode::setDecoderType(const std::string& codec_name) {
    decoder_type_ = codec_name;
}

void FFmpegDecodeNode::setOutputBGR(bool enable) {
    output_bgr_ = enable;
}

size_t FFmpegDecodeNode::getActiveDecoderCount() const {
    return active_decoders_;
}

void FFmpegDecodeNode::setSnapshotEnabled(bool enabled) {
    snapshot_enabled_ = enabled;
    LOG_INFO_FMT("[FFmpegDecode] Snapshot enabled: {}", enabled);
}

void FFmpegDecodeNode::setSnapshotInterval(int interval) {
    snapshot_interval_ = interval > 0 ? interval : 100;
    LOG_INFO_FMT("[FFmpegDecode] Snapshot interval: {} frames", snapshot_interval_.load());
}

void FFmpegDecodeNode::setSnapshotDir(const std::string& dir) {
    snapshot_dir_ = dir;
    LOG_INFO_FMT("[FFmpegDecode] Snapshot directory: {}", dir);
}

void FFmpegDecodeNode::setHwDecodeEnabled(bool enabled) {
    use_hw_ = enabled;
    LOG_INFO_FMT("[FFmpegDecode] HW decode enabled: {}", enabled);
}

bool FFmpegDecodeNode::isHwDecodeEnabled() {
    return use_hw_.load();
}

void FFmpegDecodeNode::setVideoCodecBackend(hal::VideoCodecBackend backend) {
    codec_backend_ = backend;
    LOG_INFO_FMT("[FFmpegDecode] Video codec backend set to: {}", static_cast<int>(backend));
}

bool FFmpegDecodeNode::onStartup() {
    if (snapshot_enabled_) {
        if (!fs::exists(snapshot_dir_)) {
            try {
                fs::create_directories(snapshot_dir_);
                LOG_INFO_FMT("[FFmpegDecode] Created snapshot directory: {}", snapshot_dir_);
            } catch (const std::exception& e) {
                LOG_ERROR_FMT("[FFmpegDecode] Failed to create snapshot directory: {}", e.what());
                snapshot_enabled_ = false;
            }
        }
    }

    LOG_INFO_FMT("[FFmpegDecode] Started (backend: {}, snapshot: {}, interval: {})",
                 static_cast<int>(codec_backend_),
                 snapshot_enabled_.load(), snapshot_interval_.load());
    return true;
}

void FFmpegDecodeNode::onShutdown() {
    // 清理所有解码器
    std::lock_guard<std::mutex> lock(mutex_);
    decoders_.clear();
    active_decoders_ = 0;

    LOG_INFO_FMT("[FFmpegDecode] Stopped (total frames: {}, snapshots saved: {})",
                 frame_count_, snapshot_count_);
}

void FFmpegDecodeNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO("[FFmpegDecode] Stream ended");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 workerLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }

    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    if (packet->type != core::PacketType::RAW_VIDEO) return;
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    auto raw_pkt = std::dynamic_pointer_cast<core::RawVideoPacket>(packet);
    if (!raw_pkt || raw_pkt->data.empty()) return;

    uint32_t stream_id = raw_pkt->stream_id;

    auto decoder_ctx = getOrCreateDecoder(stream_id, raw_pkt->codec_id,
                                          raw_pkt->extradata.data(),
                                          static_cast<int>(raw_pkt->extradata.size()));
    if (!decoder_ctx) {
        LOG_ERROR_FMT("[FFmpegDecode] Failed to get decoder for stream {}", stream_id);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto frame_pkt = decodePacket(raw_pkt, decoder_ctx);
    auto t1 = std::chrono::high_resolution_clock::now();
    LOG_INFO_FMT("[FFmpegDecode] Decoded frame {} ({} ms)", frame_count_,
                 std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    if (frame_pkt) {
        LOG_INFO_FMT("[FFmpegDecode] broadcast frame {} ", frame_count_);
        broadcast(frame_pkt);

        frame_count_++;
        if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) {
            saveSnapshot(frame_pkt, frame_count_);
        }
    }
}

std::shared_ptr<DecoderContext> FFmpegDecodeNode::getOrCreateDecoder(
    uint32_t stream_id, int codec_id,
    const uint8_t* extradata, int extradata_size) {

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = decoders_.find(stream_id);
    if (it != decoders_.end()) {
        return it->second;
    }

    // 创建新的解码器上下文
    auto ctx = std::make_shared<DecoderContext>();

    // 通过 HAL 工厂创建视频编解码器
    ctx->codec = hal::VideoCodecFactory::instance().create(codec_backend_);
    if (!ctx->codec) {
        LOG_ERROR("[FFmpegDecode] Failed to create video codec from factory");
        return nullptr;
    }

    // 初始化解码器
    std::string codec_name = "h264";
    if (codec_id == AV_CODEC_ID_HEVC) {
        codec_name = "hevc";
    }

    if (!ctx->codec->init(codec_name, extradata, extradata_size)) {
        LOG_ERROR_FMT("[FFmpegDecode] Failed to initialize codec for stream {}", stream_id);
        return nullptr;
    }

    ctx->stream_id = stream_id;
    ctx->initialized = true;
    decoders_[stream_id] = ctx;
    active_decoders_++;

    LOG_INFO_FMT("[FFmpegDecode] Created decoder for stream {} (backend: {})",
                 stream_id, ctx->codec->getName());
    return ctx;
}

std::shared_ptr<core::VideoFramePacket> FFmpegDecodeNode::decodePacket(
    std::shared_ptr<core::RawVideoPacket> raw_pkt,
    std::shared_ptr<DecoderContext> ctx) {

    if (!ctx || !ctx->codec) {
        LOG_ERROR("[FFmpegDecode] Invalid decoder context");
        return nullptr;
    }

    bool expected = false;
    if (!ctx->in_use.compare_exchange_strong(expected, true)) {
        LOG_ERROR("[FFmpegDecode] Decoder context is in use");
        return nullptr;
    }

    hal::DecodedFrame decoded;
    bool success = ctx->codec->decode(raw_pkt->data.data(),
                                      static_cast<int>(raw_pkt->data.size()),
                                      decoded);

    if (!success) {
        ctx->in_use = false;
        return nullptr;
    }

    int width = decoded.width;
    int height = decoded.height;

    if (width <= 0 || height <= 0) {
        LOG_ERROR("[FFmpegDecode] Invalid frame size");
        ctx->in_use = false;
        return nullptr;
    }

    ctx->width = width;
    ctx->height = height;

    cv::Mat mat;

    // 检查是否为硬件解码的 NV12 格式，需要转换为 BGR
    bool is_nv12 = (decoded.format == AV_PIX_FMT_NV12 ||
                    decoded.format == AV_PIX_FMT_CUDA);

    if (is_nv12 && output_bgr_) {
        LOG_DEBUG_FMT("[FFmpegDecode] Frame {}: format={}, has_uv={}, pitch={}, pitch_uv={}",
                      frame_count_, decoded.format, decoded.data_uv != nullptr, decoded.pitch, decoded.pitch_uv);

        // 使用 FFmpeg swscale 进行 NV12 -> BGR 转换
        if (!ctx->sws_ctx || ctx->width != width || ctx->height != height) {
            if (ctx->sws_ctx) {
                sws_freeContext(ctx->sws_ctx);
            }

            AVPixelFormat src_format = (decoded.format == AV_PIX_FMT_CUDA) ?
                                       AV_PIX_FMT_NV12 : static_cast<AVPixelFormat>(decoded.format);

            ctx->sws_ctx = sws_getContext(
                width, height, src_format,
                width, height, AV_PIX_FMT_BGR24,
                SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

            if (!ctx->sws_ctx) {
                LOG_ERROR("[FFmpegDecode] Failed to create sws context");
                ctx->in_use = false;
                return nullptr;
            }

            int buffer_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
            if (buffer_size > ctx->buffer_size) {
                uint8_t* new_buffer = (uint8_t*)av_malloc(buffer_size);
                if (!new_buffer) {
                    LOG_ERROR("[FFmpegDecode] Failed to allocate BGR buffer");
                    ctx->in_use = false;
                    return nullptr;
                }
                if (ctx->bgr_buffer) {
                    av_free(ctx->bgr_buffer);
                }
                ctx->bgr_buffer = new_buffer;
                ctx->buffer_size = buffer_size;
            }

            av_image_fill_arrays(ctx->bgr_frame->data, ctx->bgr_frame->linesize,
                                 ctx->bgr_buffer, AV_PIX_FMT_BGR24, width, height, 1);

            LOG_INFO_FMT("[FFmpegDecode] Created sws context for {}x{} NV12->BGR", width, height);
        }

        // 构造 NV12 帧数据
        const uint8_t* src_data[4] = {nullptr};
        int src_linesize[4] = {0};

        if (decoded.data) {
            src_data[0] = decoded.data;
            src_linesize[0] = decoded.pitch;
            // UV 平面（NV12 格式）
            if (decoded.data_uv) {
                src_data[1] = decoded.data_uv;
                src_linesize[1] = decoded.pitch_uv;
            }
        }

        sws_scale(ctx->sws_ctx,
                  src_data, src_linesize, 0, height,
                  ctx->bgr_frame->data, ctx->bgr_frame->linesize);

        mat = cv::Mat(height, width, CV_8UC3, ctx->bgr_frame->data[0], ctx->bgr_frame->linesize[0]).clone();
    } else if (output_bgr_) {
        // 使用 FFmpeg swscale 进行格式转换（非 NV12 格式）
        if (!ctx->sws_ctx || ctx->width != width || ctx->height != height) {
            if (ctx->sws_ctx) {
                sws_freeContext(ctx->sws_ctx);
            }

            AVPixelFormat src_format = static_cast<AVPixelFormat>(decoded.format);
            if (src_format == AV_PIX_FMT_NONE) {
                src_format = AV_PIX_FMT_YUV420P;
            }

            ctx->sws_ctx = sws_getContext(
                width, height, src_format,
                width, height, AV_PIX_FMT_BGR24,
                SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

            if (!ctx->sws_ctx) {
                ctx->in_use = false;
                return nullptr;
            }

            int buffer_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
            if (buffer_size > ctx->buffer_size) {
                uint8_t* new_buffer = (uint8_t*)av_malloc(buffer_size);
                if (!new_buffer) {
                    LOG_ERROR("[FFmpegDecode] Failed to allocate BGR buffer");
                    ctx->in_use = false;
                    return nullptr;
                }
                if (ctx->bgr_buffer) {
                    av_free(ctx->bgr_buffer);
                }
                ctx->bgr_buffer = new_buffer;
                ctx->buffer_size = buffer_size;
            }

            av_image_fill_arrays(ctx->bgr_frame->data, ctx->bgr_frame->linesize,
                                 ctx->bgr_buffer, AV_PIX_FMT_BGR24, width, height, 1);
        }

        const uint8_t* plane_data[4] = {nullptr};
        int plane_linesize[4] = {0};

        if (decoded.data) {
            plane_data[0] = decoded.data;
            plane_linesize[0] = decoded.pitch;
        }

        sws_scale(ctx->sws_ctx,
                  plane_data, plane_linesize, 0, height,
                  ctx->bgr_frame->data, ctx->bgr_frame->linesize);

        mat = cv::Mat(height, width, CV_8UC3, ctx->bgr_frame->data[0], ctx->bgr_frame->linesize[0]).clone();
    } else {
        // 直接输出原始格式
        mat = cv::Mat(height, width, CV_8UC3);
    }

    ctx->in_use = false;

    auto new_frame = std::make_shared<core::VideoFramePacket>();
    new_frame->stream_id = raw_pkt->stream_id;
    new_frame->source_id = raw_pkt->source_id;
    new_frame->timestamp_ms = raw_pkt->timestamp_ms;
    new_frame->mat = std::make_shared<cv::Mat>(std::move(mat));
    new_frame->source_mat = new_frame->mat;
    new_frame->width = width;
    new_frame->height = height;
    new_frame->channels = 3;
    new_frame->frame_id = raw_pkt->frame_id;
    new_frame->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
    new_frame->cost_time_map = raw_pkt->cost_time_map;
    new_frame->cost_time_map.insert({name_, utils::TimeUtil::currentTimeMs() - in_time_ms_});

    return new_frame;
}

void FFmpegDecodeNode::saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num) {
    if (!frame || !frame->mat || frame->mat->empty()) {
        return;
    }

    try {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count() % 1000000;

        std::tm tm_buf = utils::TimeUtil::safeLocaltime(time_t);

        std::stringstream ss;
        ss << snapshot_dir_ << "/"
           << "stream_" << frame->stream_id << "_"
           << "frame_" << std::setfill('0') << std::setw(6) << frame_num << "_"
           << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_"
           << std::setfill('0') << std::setw(6) << us
           << ".jpg";

        std::string filename = ss.str();

        bool success = cv::imwrite(filename, *frame->mat);

        if (success) {
            snapshot_count_++;
            LOG_INFO_FMT("[FFmpegDecode] Snapshot saved: {} (frame #{}, total: {})",
                         filename, frame_num, snapshot_count_);
        }
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[FFmpegDecode] Exception saving snapshot: {}", e.what());
    }
}

REGISTER_NODE("ffmpeg_decode", FFmpegDecodeNode)

} // namespace nodes
} // namespace ai_stream
