// src/nodes/source/file_source.cpp
#include "file_source.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/time.h>
}

namespace ai_stream {
namespace nodes {

FileSourceNode::FileSourceNode() : ISourceNode("FileSource") {
    LOG_DEBUG_FMT("[FileSource] Constructor");
}

FileSourceNode::~FileSourceNode() {
    stop();
    LOG_DEBUG_FMT("[FileSource] Destructor");
}

void FileSourceNode::setUrl(const std::string& url) {
    url_ = url;
}

void FileSourceNode::setSourceId(const std::string& id) {
    source_id_ = id;
}

void FileSourceNode::setSkipFrames(int skip_frames) {
    skip_frames_ = skip_frames > 0 ? static_cast<uint8_t>(skip_frames) : 1;
}

std::string FileSourceNode::getUrl() const {
    return url_;
}

bool FileSourceNode::configure(const std::string& node_id, const nlohmann::json& params) {
    if (!ISourceNode::configure(node_id, params)) {
        return false;
    }
    if (params.contains("loop")) {
        loop_enabled_ = params["loop"].get<bool>();
    }
    if (params.contains("realtime")) {
        realtime_ = params["realtime"].get<bool>();
    }
    return true;
}

bool FileSourceNode::start() {
    if (running_) {
        LOG_WARN_FMT("[FileSource] Already running");
        return true;
    }

    if (url_.empty()) {
        LOG_ERROR_FMT("[FileSource] URL (file path) not set");
        return false;
    }

    if (!openInput()) {
        LOG_ERROR_FMT("[FileSource] Failed to open input file: {}", url_);
        return false;
    }

    running_ = true;

    static std::atomic<uint32_t> global_stream_id{1000};
    my_stream_id_ = ++global_stream_id;

    worker_ = std::thread(&FileSourceNode::workerFunc, this);
    LOG_INFO_FMT("[FileSource] Started for file: {} (stream_id={}, loop={}, realtime={})",
                 url_, my_stream_id_, loop_enabled_.load(), realtime_.load());
    return true;
}

void FileSourceNode::stop() {
    if (!running_) return;

    LOG_INFO_FMT("[FileSource] Stopping (stream_id={})", my_stream_id_);
    running_ = false;

    // 文件读取不会长期阻塞（av_read_frame 对本地文件快速返回），直接 join
    if (worker_.joinable()) {
        worker_.join();
    }

    closeInput();
    LOG_INFO_FMT("[FileSource] Stopped (stream_id={}, total frames: {})",
                 my_stream_id_, total_frames_.load());
}

void FileSourceNode::pushData(std::shared_ptr<core::BasePacket> /*packet*/) {
    // 源节点不接受输入
}

bool FileSourceNode::openInput() {
    LOG_INFO_FMT("[FileSource] Opening file: {}", url_);

    int ret = avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[FileSource] Failed to open input: {} ({})", errbuf, ret);
        return false;
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[FileSource] Failed to find stream info: {} ({})", errbuf, ret);
        closeInput();
        return false;
    }

    video_stream_index_ = -1;
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
        AVCodecParameters* codecpar = fmt_ctx_->streams[i]->codecpar;
        if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index_ = i;
            codec_id_ = codecpar->codec_id;

            if (codecpar->extradata && codecpar->extradata_size > 0) {
                extradata_.assign(codecpar->extradata, codecpar->extradata + codecpar->extradata_size);
            }

            LOG_INFO_FMT("[FileSource] Video stream found: index={}, codec={}, {}x{}",
                         i, avcodec_get_name(codecpar->codec_id), codecpar->width, codecpar->height);
            break;
        }
    }

    if (video_stream_index_ == -1) {
        LOG_ERROR_FMT("[FileSource] No video stream found in file");
        closeInput();
        return false;
    }

    return true;
}

void FileSourceNode::closeInput() {
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_index_ = -1;
}

void FileSourceNode::workerFunc() {
    LOG_INFO_FMT("[FileSource] Worker thread started (stream_id={})", my_stream_id_);

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        LOG_ERROR_FMT("[FileSource] Failed to allocate packet");
        return;
    }

    AVRational& time_base = fmt_ctx_->streams[video_stream_index_]->time_base;

    int64_t frame_count = 0;
    int64_t base_pkt_ms = -1;
    auto base_wall = std::chrono::steady_clock::now();

    while (running_) {
        int ret = av_read_frame(fmt_ctx_, pkt);

        if (!running_) break;

        if (ret < 0) {
            // 文件读完或读取错误
            if (!loop_enabled_) {
                LOG_INFO_FMT("[FileSource] End of file reached (stream_id={})", my_stream_id_);
                auto eos_packet = std::make_shared<core::BasePacket>();
                eos_packet->type = core::PacketType::STREAM_END;
                eos_packet->stream_id = my_stream_id_;
                eos_packet->source_id = source_id_;
                eos_packet->timestamp_ms = utils::TimeUtil::currentTimeMs();
                broadcast(eos_packet);
                running_ = false;
                break;
            }

            // 循环播放：回到文件开头，重置 pacing 基准
            LOG_INFO_FMT("[FileSource] Looping file (stream_id={})", my_stream_id_);
            av_seek_frame(fmt_ctx_, video_stream_index_, 0, AVSEEK_FLAG_BACKWARD);
            base_pkt_ms = -1;
            base_wall = std::chrono::steady_clock::now();
            continue;
        }

        if (pkt->stream_index != video_stream_index_) {
            av_packet_unref(pkt);
            continue;
        }

        frame_count++;
        if (frame_count % skip_frames_ != 0) {
            av_packet_unref(pkt);
            continue;
        }

        int64_t pkt_ms = 0;
        if (pkt->pts != AV_NOPTS_VALUE) {
            pkt_ms = av_rescale_q(pkt->pts, time_base, AV_TIME_BASE_Q) / 1000;
        } else if (pkt->dts != AV_NOPTS_VALUE) {
            pkt_ms = av_rescale_q(pkt->dts, time_base, AV_TIME_BASE_Q) / 1000;
        }

        // 按原始帧率节奏推送，避免文件读取过快压垮下游
        if (realtime_) {
            if (base_pkt_ms < 0) {
                base_pkt_ms = pkt_ms;
                base_wall = std::chrono::steady_clock::now();
            } else {
                auto target = base_wall + std::chrono::milliseconds(pkt_ms - base_pkt_ms);
                auto now = std::chrono::steady_clock::now();
                if (now < target) {
                    std::this_thread::sleep_until(target);
                }
            }
        }

        if (!running_) {
            av_packet_unref(pkt);
            break;
        }

        auto raw_pkt = std::make_shared<core::RawVideoPacket>();
        raw_pkt->stream_id = my_stream_id_;
        raw_pkt->source_id = source_id_;
        raw_pkt->frame_id = frame_count;
        raw_pkt->timestamp_ms = utils::TimeUtil::currentTimeMs();
        raw_pkt->is_key_frame = (pkt->flags & AV_PKT_FLAG_KEY);
        raw_pkt->codec_id = codec_id_;
        raw_pkt->data.assign(pkt->data, pkt->data + pkt->size);
        raw_pkt->extradata = extradata_;

        total_frames_ = frame_count;
        broadcast(raw_pkt);

        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    LOG_INFO_FMT("[FileSource] Worker thread ended, total frames: {} (stream_id={})",
                 frame_count, my_stream_id_);
}

REGISTER_NODE("file_source", FileSourceNode)

} // namespace nodes
} // namespace ai_stream
