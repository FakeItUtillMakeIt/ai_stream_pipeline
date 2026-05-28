// src/nodes/decode/ffmpeg_decode.cpp
#include "hw_cuda_decode.h"
#include "ffmpeg_decode.h"
#include "ai_stream/core/packet.h"
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

FFmpegDecodeNode::FFmpegDecodeNode() : IDecodeNode("FFmpegDecode") {
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
    return pool_.getActiveCount();
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

bool FFmpegDecodeNode::start() {
    running_ = true;
    
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
    
    LOG_INFO_FMT("[FFmpegDecode] Started (snapshot: {}, interval: {})", 
                 snapshot_enabled_.load(), snapshot_interval_.load());
    return true;
}

void FFmpegDecodeNode::stop() {
    running_ = false;
    pool_.clear();
    LOG_INFO_FMT("[FFmpegDecode] Stopped (total frames: {}, snapshots saved: {})", 
                 frame_count_, snapshot_count_);
}

void FFmpegDecodeNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO("[FFmpegDecode] Stream ended");
        stop();
        broadcast(packet);
        return;
    }
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    if (!running_) return;
    if (packet->type != core::PacketType::RAW_VIDEO) return;
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    auto raw_pkt = std::dynamic_pointer_cast<core::RawVideoPacket>(packet);
    if (!raw_pkt || raw_pkt->data.empty()) return;
    
    uint32_t stream_id = raw_pkt->stream_id;

    auto decoder_ctx = pool_.getDecoder(stream_id, raw_pkt->codec_id, use_hw_);
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
        LOG_INFO_FMT("[FFmpegDecode] broadcast frame {} ", frame_count_ );
        broadcast(frame_pkt);
        
        frame_count_++;
        if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) {
            saveSnapshot(frame_pkt, frame_count_);
        }
    }
}

std::shared_ptr<core::VideoFramePacket> FFmpegDecodeNode::decodePacket(
    std::shared_ptr<core::RawVideoPacket> raw_pkt,
    std::shared_ptr<DecoderContext> ctx) {
    
    if (!ctx || !ctx->codec_ctx) {
        LOG_ERROR("[FFmpegDecode] Invalid decoder context");
        return nullptr;
    }
    
    bool expected = false;
    if (!ctx->in_use.compare_exchange_strong(expected, true)) {
        LOG_ERROR("[FFmpegDecode] Decoder context is in use");
        return nullptr;
    }
    
    AVPacket av_pkt;
    av_init_packet(&av_pkt);
    av_pkt.data = const_cast<uint8_t*>(raw_pkt->data.data());
    av_pkt.size = raw_pkt->data.size();
    
    if (raw_pkt->is_key_frame) {
        av_pkt.flags |= AV_PKT_FLAG_KEY;
    }
    
    int ret = avcodec_send_packet(ctx->codec_ctx, &av_pkt);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        
        // 【关键】如果是 extradata 缺失导致的无效数据，降级为 WARN，不中断解码器
        if (ret == AVERROR_INVALIDDATA) {
            LOG_WARN_FMT("[FFmpegDecode] Invalid data (missing SPS/PPS?), skipping packet");
            ctx->in_use = false;
            return nullptr;
        }
        
        LOG_ERROR_FMT("[FFmpegDecode] avcodec_send_packet failed: {}", errbuf);
        ctx->in_use = false;
        return nullptr;
    }
    
    AVFrame* decoded_frame = ctx->use_hw ? ctx->hw_frame : ctx->frame;
    std::shared_ptr<core::VideoFramePacket> frame_pkt = nullptr;
    
    while (true) {
        ret = avcodec_receive_frame(ctx->codec_ctx, decoded_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[FFmpegDecode] avcodec_receive_frame failed: {}, use_hw: {}", errbuf, ctx->use_hw);
            break;
        }
        
        int width = decoded_frame->width;
        int height = decoded_frame->height;
        
        if (width <= 0 || height <= 0) {
            LOG_ERROR("[FFmpegDecode] Invalid frame size");
            av_frame_unref(decoded_frame);
            continue;
        }
        
        ctx->width = width;
        ctx->height = height;

        if (ctx->use_hw) {
            uint8_t* y_ptr = decoded_frame->data[0];
            uint8_t* uv_ptr = decoded_frame->data[1];
            size_t y_pitch = decoded_frame->linesize[0];
            size_t uv_pitch = decoded_frame->linesize[1];
            
            size_t needed = static_cast<size_t>(width) * height * 3;
            if (ctx->d_bgr_buffer_size < needed) {
                if (ctx->d_bgr_buffer) cudaFree(ctx->d_bgr_buffer);
                cudaMalloc(&ctx->d_bgr_buffer, needed);
                ctx->d_bgr_buffer_size = needed;
            }
            
            launchNV12ToBGR(
                y_ptr, uv_ptr, width, height,
                y_pitch, uv_pitch,
                static_cast<uint8_t*>(ctx->d_bgr_buffer),
                static_cast<size_t>(width) * 3,
                cuda_stream_
            );
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                LOG_ERROR_FMT("[FFmpegDecode] NV12ToBGR kernel failed: {}", cudaGetErrorString(err));
                av_frame_unref(decoded_frame);
                continue;
            }
            
            cv::Mat cpu_bgr(height, width, CV_8UC3);
            err = cudaMemcpy(cpu_bgr.data, ctx->d_bgr_buffer,
                             width * height * 3,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                LOG_ERROR_FMT("[FFmpegDecode] D2H copy failed: {}", cudaGetErrorString(err));
                av_frame_unref(decoded_frame);
                continue;
            }
            
            auto new_frame = std::make_shared<core::VideoFramePacket>();
            new_frame->stream_id = raw_pkt->stream_id;
            new_frame->source_id = raw_pkt->source_id;
            new_frame->timestamp_ms = raw_pkt->timestamp_ms;
            new_frame->is_gpu = true;
            new_frame->d_ptr = ctx->d_bgr_buffer;
            new_frame->d_pitch = width * 3;
            new_frame->d_width = width;
            new_frame->d_height = height;
            new_frame->width = width;
            new_frame->height = height;
            new_frame->channels = 3;
            new_frame->frame_id = raw_pkt->frame_id;
            new_frame->mat = std::make_shared<cv::Mat>(std::move(cpu_bgr));
            new_frame->source_mat = new_frame->mat;
            
            av_frame_unref(decoded_frame);
            frame_pkt = new_frame;
        }
        else {
            cv::Mat mat;
            
            if (output_bgr_) {
                if (!ctx->sws_ctx || ctx->width != width || ctx->height != height) {
                    if (ctx->sws_ctx) {
                        sws_freeContext(ctx->sws_ctx);
                    }
                    
                    ctx->sws_ctx = sws_getContext(
                        width, height, (AVPixelFormat)ctx->frame->format,
                        width, height, AV_PIX_FMT_BGR24,
                        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
                    
                    if (!ctx->sws_ctx) {
                        av_frame_unref(decoded_frame);
                        continue;
                    }
                    
                    int buffer_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
                    if (buffer_size > ctx->buffer_size) {
                        if (ctx->bgr_buffer) {
                            av_free(ctx->bgr_buffer);
                        }
                        ctx->bgr_buffer = (uint8_t*)av_malloc(buffer_size);
                        ctx->buffer_size = buffer_size;
                    }
                    
                    av_image_fill_arrays(ctx->bgr_frame->data, ctx->bgr_frame->linesize,
                                         ctx->bgr_buffer, AV_PIX_FMT_BGR24, width, height, 1);
                }
                
                sws_scale(ctx->sws_ctx,
                          ctx->frame->data, ctx->frame->linesize, 0, height,
                          ctx->bgr_frame->data, ctx->bgr_frame->linesize);
                
                mat = cv::Mat(height, width, CV_8UC3, ctx->bgr_frame->data[0], ctx->bgr_frame->linesize[0]).clone();
            } else {
                mat = cv::Mat(height, width, CV_8UC3);
            }
            
            av_frame_unref(decoded_frame);
            
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
            
            frame_pkt = new_frame;
        }
    }
    
    ctx->in_use = false;
    return frame_pkt;
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
        
        std::tm tm_buf;
        localtime_r(&time_t, &tm_buf);
        
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