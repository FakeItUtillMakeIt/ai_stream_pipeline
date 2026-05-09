// src/nodes/source/rtsp_source.cpp
#include "rtsp_source.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <chrono>
#include <sstream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/time.h>
}

namespace ai_stream {
namespace nodes {

// 静态初始化 FFmpeg 网络组件
static struct FFmpegInitializer {
    FFmpegInitializer() {
        avformat_network_init();
        LOG_INFO("[FFmpeg] Network initialized");
    }
    ~FFmpegInitializer() {
        avformat_network_deinit();
        LOG_INFO("[FFmpeg] Network deinitialized");
    }
} g_ffmpeg_init;

RTSPSourceNode::RTSPSourceNode() : ISourceNode("RTSPSource") {
    LOG_DEBUG_FMT("[RTSPSource] Constructor");
}

RTSPSourceNode::~RTSPSourceNode() {
    stop();
    LOG_DEBUG_FMT("[RTSPSource] Destructor");
}

void RTSPSourceNode::setUrl(const std::string& url) {
    url_ = url;
}

void RTSPSourceNode::setSourceId(const std::string& id) {
    source_id_ = id;
}

std::string RTSPSourceNode::getUrl() const {
    return url_;
}

bool RTSPSourceNode::start() {
    if (running_) {
        LOG_WARN_FMT("[RTSPSource] Already running");
        return true;
    }
    
    if (url_.empty()) {
        LOG_ERROR_FMT("[RTSPSource] URL not set");
        return false;
    }
    
    if (!openInput()) {
        LOG_ERROR_FMT("[RTSPSource] Failed to open input");
        return false;
    }
    
    running_ = true;
    
    // 分配唯一的 stream_id
    static std::atomic<uint32_t> global_stream_id{0};
    my_stream_id_ = ++global_stream_id;
    
    worker_ = std::thread(&RTSPSourceNode::workerFunc, this);
    LOG_INFO_FMT("[RTSPSource] Started for URL: {} (stream_id={})", url_, my_stream_id_);
    return true;
}


void RTSPSourceNode::stop() {
    if (!running_) return;
    
    LOG_INFO_FMT("[RTSPSource] Stopping (stream_id={})", my_stream_id_);
    
    // 先设置停止标志
    running_ = false;
    
    // 等待工作线程结束（带超时）
    if (worker_.joinable()) {
        // 使用 timed_join 避免无限等待
        auto start = std::chrono::steady_clock::now();
        while (worker_.joinable() && 
               std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (worker_.joinable()) {
            LOG_WARN_FMT("[RTSPSource] Worker thread not responding, forcing close");
            // 强制关闭输入，触发 av_read_frame 返回错误
            closeInput();
            worker_.join();
        }
    }
    
    closeInput();
    LOG_INFO_FMT("[RTSPSource] Stopped (stream_id={})", my_stream_id_);
}

void RTSPSourceNode::pushData(std::shared_ptr<core::BasePacket> /*packet*/) {
    // 源节点不接受输入
}

// src/nodes/source/rtsp_source.cpp

bool RTSPSourceNode::openInput() {
    LOG_INFO_FMT("[RTSPSource] Opening input: {}", url_);
    
    // 设置 RTSP 选项 - 明确使用客户端模式
    AVDictionary* opts = nullptr;
    
    // 关键：明确指定这是客户端连接，不是服务器
    av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);  // 优先使用 TCP
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);     // 强制使用 TCP
    
    // 设置超时（微秒）
    av_dict_set(&opts, "stimeout", "10000000", 0);      // 10 秒连接超时
    
    // 明确这是输入模式（拉流）
    av_dict_set(&opts, "reconnect", "1", 0);            // 自动重连
    av_dict_set(&opts, "reconnect_streamed", "1", 0);   // 重连流
    av_dict_set(&opts, "reconnect_delay_max", "5", 0);  // 最大重连延迟 5 秒
    
    // 不要使用这些选项，它们会触发服务器模式
    // av_dict_set(&opts, "listen", "0", 0);  // 这会导致服务器模式！
    
    // 分配格式上下文
    fmt_ctx_ = avformat_alloc_context();
    if (!fmt_ctx_) {
        LOG_ERROR_FMT("[RTSPSource] Failed to allocate format context");
        return false;
    }
    
    // 设置格式上下文的标志
    fmt_ctx_->flags |= AVFMT_FLAG_NONBLOCK;  // 非阻塞模式
    
    LOG_INFO_FMT("[RTSPSource] Connecting to RTSP stream as client...");
    
    // 打开输入
    int ret = avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[RTSPSource] Failed to open input: {} ({})", errbuf, ret);
        
        if (ret == -98) {
            LOG_ERROR_FMT("[RTSPSource] Address already in use. This usually means:");
            LOG_ERROR_FMT("  1. The RTSP server is not running");
            LOG_ERROR_FMT("  2. The URL is incorrect");
            LOG_ERROR_FMT("  3. The port is already occupied by another client");
        }
        
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        return false;
    }
    
    // 查找流信息（带超时保护）
    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[RTSPSource] Failed to find stream info: {} ({})", errbuf, ret);
        closeInput();
        return false;
    }
    
    // 打印详细的流信息
    LOG_INFO_FMT("[RTSPSource] Stream information:");
    LOG_INFO_FMT("[RTSPSource]   Format: {}", fmt_ctx_->iformat->name);
    LOG_INFO_FMT("[RTSPSource]   Duration: {} seconds", 
                 fmt_ctx_->duration / AV_TIME_BASE);
    LOG_INFO_FMT("[RTSPSource]   Number of streams: {}", fmt_ctx_->nb_streams);
    
    // 查找视频流
    video_stream_index_ = -1;
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
        AVCodecParameters* codecpar = fmt_ctx_->streams[i]->codecpar;
        
        if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index_ = i;
            codec_id_ = codecpar->codec_id;
            
            LOG_INFO_FMT("[RTSPSource] Video stream found:");
            LOG_INFO_FMT("[RTSPSource]   Index: {}", i);
            LOG_INFO_FMT("[RTSPSource]   Codec: {}", avcodec_get_name(codecpar->codec_id));
            LOG_INFO_FMT("[RTSPSource]   Resolution: {}x{}", codecpar->width, codecpar->height);
            
            AVRational& frame_rate = fmt_ctx_->streams[i]->avg_frame_rate;
            if (frame_rate.num > 0 && frame_rate.den > 0) {
                LOG_INFO_FMT("[RTSPSource]   Frame rate: {}/{} ≈ {:.2f} fps", 
                             frame_rate.num, frame_rate.den,
                             (double)frame_rate.num / frame_rate.den);
            }
            break;
        }
    }
    
    if (video_stream_index_ == -1) {
        LOG_ERROR_FMT("[RTSPSource] No video stream found");
        closeInput();
        return false;
    }
    
    LOG_INFO_FMT("[RTSPSource] Input opened successfully");
    return true;
}



void RTSPSourceNode::closeInput() {
    if (fmt_ctx_) {
        // 创建中断回调，强制中断阻塞的 av_read_frame
        AVIOInterruptCB interrupt_cb = {
            .callback = [](void* ctx) -> int {
                return *(bool*)ctx ? 1 : 0;
            },
            .opaque = &running_
        };
        fmt_ctx_->interrupt_callback = interrupt_cb;
        
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_index_ = -1;
}


void RTSPSourceNode::workerFunc() {
    LOG_INFO_FMT("[RTSPSource] Worker thread started (stream_id={})", my_stream_id_);
    
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        LOG_ERROR_FMT("[RTSPSource] Failed to allocate packet");
        return;
    }
    
    int frame_count = 0;
    int error_count = 0;
    const int MAX_CONSECUTIVE_ERRORS = 10;
    
    while (running_) {
        // 读取数据包（带超时，避免阻塞）
        int ret = av_read_frame(fmt_ctx_, pkt);
        
        if (!running_) {
            // 如果正在停止，立即退出
            break;
        }
        
        if (ret < 0) {
            if (ret == AVERROR(EAGAIN)) {
                // 非阻塞模式下没有数据，稍等再试
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            } else if (ret == AVERROR_EOF) {
                LOG_WARN_FMT("[RTSPSource] End of stream reached");
                break;
            } else {
                error_count++;
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                LOG_WARN_FMT("[RTSPSource] Error reading frame: {} ({})", errbuf, ret);
                
                if (error_count >= MAX_CONSECUTIVE_ERRORS) {
                    LOG_ERROR_FMT("[RTSPSource] Too many errors, stopping");
                    break;
                }
                
                // 等待时检查 running_ 标志
                for (int i = 0; i < 10 && running_; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;
            }
        }
        
        // 重置错误计数
        error_count = 0;
        
        // 只处理视频流
        if (pkt->stream_index == video_stream_index_) {
            frame_count++;
            
            // 每 100 帧打印一次日志
            if (frame_count % 100 == 0) {
                LOG_INFO_FMT("[RTSPSource] Received {} frames (stream_id={})", frame_count, my_stream_id_);
            }
            
            // 创建数据包
            auto raw_pkt = std::make_shared<core::RawVideoPacket>();
            raw_pkt->stream_id = my_stream_id_;
            raw_pkt->source_id = source_id_;
            
            // 计算时间戳
            AVRational& time_base = fmt_ctx_->streams[video_stream_index_]->time_base;
            if (pkt->pts != AV_NOPTS_VALUE) {
                raw_pkt->timestamp_ms = av_rescale_q(pkt->pts, time_base, AV_TIME_BASE_Q) / 1000;
            } else if (pkt->dts != AV_NOPTS_VALUE) {
                raw_pkt->timestamp_ms = av_rescale_q(pkt->dts, time_base, AV_TIME_BASE_Q) / 1000;
            } else {
                raw_pkt->timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            }
            
            raw_pkt->is_key_frame = (pkt->flags & AV_PKT_FLAG_KEY);
            raw_pkt->codec_id = codec_id_;
            raw_pkt->data.assign(pkt->data, pkt->data + pkt->size);
            
            broadcast(raw_pkt);
        }
        
        av_packet_unref(pkt);
    }
    
    av_packet_free(&pkt);
    LOG_INFO_FMT("[RTSPSource] Worker thread ended, total frames: {} (stream_id={})", 
                 frame_count, my_stream_id_);
}

// 工厂注册
REGISTER_NODE("rtsp_source", RTSPSourceNode)

} // namespace nodes
} // namespace ai_stream