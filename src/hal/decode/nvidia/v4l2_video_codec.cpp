// src/hal/decode/nvidia/v4l2_video_codec.cpp
// Jetson V4L2 NVDEC 硬件解码后端实现。
// 设备节点：/dev/v4l2-nvdec（备选 /dev/nvhost-nvdec）。
#include "v4l2_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <linux/videodev2.h>
#include "v4l2_nv_extensions.h"
#include <libavutil/pixfmt.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <unistd.h>
#include <cstring>

namespace ai_stream {
namespace hal {

namespace {
constexpr const char* kDecodeDev   = "/dev/v4l2-nvdec";
constexpr const char* kDecodeDevAlt = "/dev/nvhost-nvdec";
constexpr int kOutputBuffers = 8;
constexpr int kCaptureBuffers = 8;
constexpr size_t kOutputSizeImage = 4 * 1024 * 1024;  // 编码码流缓冲

bool ioctlOk(int fd, unsigned long req, void* arg) {
    return ::ioctl(fd, req, arg) == 0;
}
} // namespace

V4l2VideoCodec::V4l2VideoCodec() {
    LOG_DEBUG("[V4l2VideoCodec] Constructor");
}

V4l2VideoCodec::~V4l2VideoCodec() {
    release();
    LOG_DEBUG("[V4l2VideoCodec] Destroyed");
}

bool V4l2VideoCodec::isAvailable() const {
    // Jetson 平台存在真正的 V4L2 M2M 解码设备才可用。
    // 注意：仅 open() 成功不代表可用（某些虚拟化环境下节点可能是空设备，
    // ioctl 会返回 ENOTTY），必须用 VIDIOC_QUERYCAP 校验。
    int fd = ::open(kDecodeDev, O_RDWR);
    if (fd < 0) {
        fd = ::open(kDecodeDevAlt, O_RDWR);
    }
    if (fd < 0) {
        return false;
    }
    struct v4l2_capability cap;
    bool ok = ::ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0;
    ::close(fd);
    return ok;
}

bool V4l2VideoCodec::openDecoder() {
    fd_ = ::open(kDecodeDev, O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        fd_ = ::open(kDecodeDevAlt, O_RDWR | O_NONBLOCK);
    }
    if (fd_ < 0) {
        LOG_ERROR_FMT("[V4l2VideoCodec] Failed to open decoder device");
        return false;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (!ioctlOk(fd_, VIDIOC_QUERYCAP, &cap)) {
        LOG_ERROR("[V4l2VideoCodec] VIDIOC_QUERYCAP failed");
        return false;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE)) {
        LOG_ERROR_FMT("[V4l2VideoCodec] Device lacks M2M_MPLANE capability: driver={}", cap.driver);
        return false;
    }
    LOG_INFO_FMT("[V4l2VideoCodec] Opened decoder: driver={} card={}", cap.driver, cap.card);

    // 订阅分辨率变更事件（编码流解析后才确定宽高）
    struct v4l2_event_subscription sub;
    memset(&sub, 0, sizeof(sub));
    sub.type = V4L2_EVENT_RESOLUTION_CHANGE;
    ioctlOk(fd_, VIDIOC_SUBSCRIBE_EVENT, &sub);
    return true;
}

bool V4l2VideoCodec::setupOutputPlane(uint32_t coded_fmt, uint32_t sizeimage) {
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    fmt.fmt.pix_mp.pixelformat = coded_fmt;
    fmt.fmt.pix_mp.num_planes = 1;
    fmt.fmt.pix_mp.plane_fmt[0].sizeimage = sizeimage;
    if (!ioctlOk(fd_, VIDIOC_S_FMT, &fmt)) {
        LOG_ERROR("[V4l2VideoCodec] Output plane S_FMT failed");
        return false;
    }

    // 允许输入缓冲不是完整帧（RTSP 分包）
    struct v4l2_ext_control ext_ctrl;
    struct v4l2_ext_controls ext_ctrls;
    memset(&ext_ctrl, 0, sizeof(ext_ctrl));
    memset(&ext_ctrls, 0, sizeof(ext_ctrls));
    ext_ctrl.id = V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT;
    ext_ctrl.value = 1;
    ext_ctrls.which = V4L2_CTRL_WHICH_CUR_VAL;
    ext_ctrls.count = 1;
    ext_ctrls.controls = &ext_ctrl;
    if (ioctl(fd_, VIDIOC_S_EXT_CTRLS, &ext_ctrls) < 0) {
        LOG_WARN("[V4l2VideoCodec] Failed to set DISABLE_COMPLETE_FRAME_INPUT");
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    req.count = kOutputBuffers;
    if (!ioctlOk(fd_, VIDIOC_REQBUFS, &req)) {
        LOG_ERROR("[V4l2VideoCodec] Output REQBUFS failed");
        return false;
    }
    out_count_ = req.count;
    out_mmap_.resize(out_count_);
    out_size_.resize(out_count_);

    for (uint32_t i = 0; i < out_count_; ++i) {
        struct v4l2_plane plane;
        struct v4l2_buffer buf;
        memset(&plane, 0, sizeof(plane));
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = 1;
        buf.m.planes = &plane;
        if (!ioctlOk(fd_, VIDIOC_QUERYBUF, &buf)) {
            LOG_ERROR("[V4l2VideoCodec] Output QUERYBUF failed");
            return false;
        }
        void* mm = ::mmap(nullptr, plane.length, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd_, plane.m.mem_offset);
        if (mm == MAP_FAILED) {
            LOG_ERROR("[V4l2VideoCodec] Output mmap failed");
            return false;
        }
        out_mmap_[i] = mm;
        out_size_[i] = plane.length;
        out_free_.push_back(i);
    }

    uint32_t type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    if (!ioctlOk(fd_, VIDIOC_STREAMON, &type)) {
        LOG_ERROR("[V4l2VideoCodec] Output STREAMON failed");
        return false;
    }
    return true;
}

bool V4l2VideoCodec::init(const std::string& codec_name,
                          const uint8_t* extradata,
                          int extradata_size) {
    codec_name_ = codec_name;
    if (extradata && extradata_size > 0) {
        extradata_.assign(extradata, extradata + extradata_size);
    }

    if (!openDecoder()) {
        return false;
    }

    uint32_t coded_fmt = 0;
    if (codec_name == "h264" || codec_name == "H264") {
        coded_fmt = V4L2_PIX_FMT_H264;
    } else if (codec_name == "hevc" || codec_name == "h265" || codec_name == "H265") {
        coded_fmt = V4L2_PIX_FMT_H265;
    } else if (codec_name == "vp9" || codec_name == "VP9") {
        coded_fmt = V4L2_PIX_FMT_VP9;
    } else if (codec_name == "av1" || codec_name == "AV1") {
        coded_fmt = V4L2_PIX_FMT_AV1;
    }
    if (coded_fmt == 0) {
        LOG_ERROR_FMT("[V4l2VideoCodec] Unsupported codec: {}", codec_name);
        return false;
    }

    if (!setupOutputPlane(coded_fmt, kOutputSizeImage)) {
        return false;
    }

    initialized_ = true;
    LOG_INFO_FMT("[V4l2VideoCodec] Decoder initialized: {}", codec_name);
    return true;
}

bool V4l2VideoCodec::recycleOutputBuffers() {
    for (;;) {
        struct v4l2_plane plane;
        struct v4l2_buffer buf;
        memset(&plane, 0, sizeof(plane));
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.length = 1;
        buf.m.planes = &plane;
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            break;  // EAGAIN：无可用输出缓冲
        }
        out_free_.push_back(buf.index);
    }
    return true;
}

bool V4l2VideoCodec::feedPacket(const uint8_t* data, int size) {
    recycleOutputBuffers();
    if (out_free_.empty()) {
        return false;
    }
    uint32_t idx = out_free_.front();
    out_free_.pop_front();

    const uint8_t* src = data;
    size_t total = static_cast<size_t>(size);
    std::vector<uint8_t> tmp;
    if (!fed_extradata_ && !extradata_.empty()) {
        total += extradata_.size();
        tmp.resize(total);
        std::memcpy(tmp.data(), extradata_.data(), extradata_.size());
        std::memcpy(tmp.data() + extradata_.size(), data, size);
        src = tmp.data();
        fed_extradata_ = true;
    }
    if (total > out_size_[idx]) {
        out_free_.push_back(idx);
        return false;
    }
    std::memcpy(out_mmap_[idx], src, total);

    struct v4l2_plane plane;
    struct v4l2_buffer buf;
    memset(&plane, 0, sizeof(plane));
    memset(&buf, 0, sizeof(buf));
    plane.bytesused = static_cast<uint32_t>(total);
    buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = idx;
    buf.length = 1;
    buf.m.planes = &plane;
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        out_free_.push_back(idx);
        return false;
    }
    return true;
}

bool V4l2VideoCodec::waitResolutionChange(int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = fd_;
    pfd.events = POLLIN | POLLPRI;
    pfd.revents = 0;
    int r = ::poll(&pfd, 1, timeout_ms);
    if (r <= 0) {
        return false;
    }
    for (;;) {
        struct v4l2_event ev;
        memset(&ev, 0, sizeof(ev));
        if (ioctl(fd_, VIDIOC_DQEVENT, &ev) < 0) {
            break;
        }
        if (ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            return true;
        }
    }
    return false;
}

bool V4l2VideoCodec::setupCapturePlane() {
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (!ioctlOk(fd_, VIDIOC_G_FMT, &fmt)) {
        LOG_ERROR("[V4l2VideoCodec] Capture G_FMT failed");
        return false;
    }
    width_ = fmt.fmt.pix_mp.width;
    height_ = fmt.fmt.pix_mp.height;
    const uint32_t num_planes = fmt.fmt.pix_mp.num_planes;
    if (width_ <= 0 || height_ <= 0 || num_planes < 2) {
        LOG_ERROR_FMT("[V4l2VideoCodec] Unexpected capture format {}x{} planes={}",
                      width_, height_, num_planes);
        return false;
    }

    if (!ioctlOk(fd_, VIDIOC_S_FMT, &fmt)) {
        LOG_ERROR("[V4l2VideoCodec] Capture S_FMT failed");
        return false;
    }

    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id = V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;
    uint32_t count = kCaptureBuffers;
    if (ioctl(fd_, VIDIOC_G_CTRL, &ctrl) == 0 && ctrl.value > 0) {
        count = std::max<uint32_t>(kCaptureBuffers, static_cast<uint32_t>(ctrl.value));
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    req.count = count;
    if (!ioctlOk(fd_, VIDIOC_REQBUFS, &req)) {
        LOG_ERROR("[V4l2VideoCodec] Capture REQBUFS failed");
        return false;
    }
    cap_count_ = req.count;
    cap_mmap_.clear();
    cap_size_.clear();
    cap_free_.clear();
    cap_mmap_.resize(cap_count_);
    cap_size_.resize(cap_count_);

    for (uint32_t i = 0; i < cap_count_; ++i) {
        struct v4l2_plane planes[3];
        struct v4l2_buffer buf;
        memset(planes, 0, sizeof(planes));
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = num_planes;
        buf.m.planes = planes;
        if (!ioctlOk(fd_, VIDIOC_QUERYBUF, &buf)) {
            LOG_ERROR("[V4l2VideoCodec] Capture QUERYBUF failed");
            return false;
        }
        for (uint32_t p = 0; p < num_planes; ++p) {
            void* mm = ::mmap(nullptr, planes[p].length, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd_, planes[p].m.mem_offset);
            if (mm == MAP_FAILED) {
                LOG_ERROR("[V4l2VideoCodec] Capture mmap failed");
                return false;
            }
            cap_mmap_[i].push_back(mm);
            cap_size_[i].push_back(planes[p].length);
        }
    }

    uint32_t type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (!ioctlOk(fd_, VIDIOC_STREAMON, &type)) {
        LOG_ERROR("[V4l2VideoCodec] Capture STREAMON failed");
        return false;
    }
    capture_on_ = true;

    // 全部 capture 缓冲入队供解码器写入
    for (uint32_t i = 0; i < cap_count_; ++i) {
        struct v4l2_plane planes[3];
        struct v4l2_buffer buf;
        memset(planes, 0, sizeof(planes));
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = static_cast<uint32_t>(cap_mmap_[i].size());
        buf.m.planes = planes;
        ioctlOk(fd_, VIDIOC_QBUF, &buf);
    }

    LOG_INFO_FMT("[V4l2VideoCodec] Capture plane ready: {}x{} planes={}",
                 width_, height_, num_planes);
    return true;
}

bool V4l2VideoCodec::dequeueCaptureFrame(DecodedFrame& frame) {
    // 处理可能的分辨率变更事件
    for (;;) {
        struct v4l2_event ev;
        memset(&ev, 0, sizeof(ev));
        if (ioctl(fd_, VIDIOC_DQEVENT, &ev) < 0) {
            break;
        }
        if (ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            setupCapturePlane();
        }
    }

    struct pollfd pfd;
    pfd.fd = fd_;
    pfd.events = POLLIN;
    pfd.revents = 0;
    if (::poll(&pfd, 1, 50) <= 0) {
        return false;
    }

    struct v4l2_plane planes[3];
    struct v4l2_buffer buf;
    memset(planes, 0, sizeof(planes));
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = 3;
    buf.m.planes = planes;
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        return false;  // EAGAIN
    }
    uint32_t idx = buf.index;
    const int np = static_cast<int>(cap_mmap_[idx].size());
    if (np >= 2 && width_ > 0 && height_ > 0) {
        const size_t y_size = cap_size_[idx][0];
        const size_t uv_size = cap_size_[idx][1];
        const int stride_y = static_cast<int>(y_size / height_);
        const int stride_uv = static_cast<int>(uv_size / (height_ / 2));

        frame_y_.resize(y_size);
        frame_uv_.resize(uv_size);
        std::memcpy(frame_y_.data(), cap_mmap_[idx][0], y_size);
        std::memcpy(frame_uv_.data(), cap_mmap_[idx][1], uv_size);

        frame.data = frame_y_.data();
        frame.pitch = stride_y;
        frame.data_uv = frame_uv_.data();
        frame.pitch_uv = stride_uv;
        frame.format = AV_PIX_FMT_NV12;
        frame.width = width_;
        frame.height = height_;
        frame.owns_data = false;
        frame.is_gpu = false;
    }

    // 回收 capture 缓冲
    struct v4l2_plane qplanes[3];
    struct v4l2_buffer qbuf;
    memset(qplanes, 0, sizeof(qplanes));
    memset(&qbuf, 0, sizeof(qbuf));
    qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    qbuf.memory = V4L2_MEMORY_MMAP;
    qbuf.index = idx;
    qbuf.length = static_cast<uint32_t>(np);
    qbuf.m.planes = qplanes;
    ioctlOk(fd_, VIDIOC_QBUF, &qbuf);

    return np >= 2;
}

bool V4l2VideoCodec::decode(const uint8_t* packet_data, int packet_size,
                            DecodedFrame& frame) {
    if (!initialized_) {
        return false;
    }

    if (!capture_on_) {
        // 首帧阶段：喂包触发分辨率变更，建立 capture 平面
        if (!feedPacket(packet_data, packet_size)) {
            return false;
        }
        if (waitResolutionChange(300)) {
            if (!setupCapturePlane()) {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if (!feedPacket(packet_data, packet_size)) {
            return false;
        }
    }

    return dequeueCaptureFrame(frame);
}

void V4l2VideoCodec::release() {
    if (fd_ >= 0) {
        if (capture_on_) {
            uint32_t type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            ioctlOk(fd_, VIDIOC_STREAMOFF, &type);
        }
        uint32_t type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        ioctlOk(fd_, VIDIOC_STREAMOFF, &type);
        ::close(fd_);
        fd_ = -1;
    }
    out_mmap_.clear();
    out_size_.clear();
    out_free_.clear();
    out_count_ = 0;
    cap_mmap_.clear();
    cap_size_.clear();
    cap_free_.clear();
    cap_count_ = 0;
    capture_on_ = false;
    initialized_ = false;
    fed_extradata_ = false;
    width_ = 0;
    height_ = 0;
    frame_y_.clear();
    frame_uv_.clear();
    extradata_.clear();
}

// 注册 V4L2 后端到工厂
#ifdef WITH_CUDA
REGISTER_VIDEO_CODEC(VideoCodecBackend::V4L2, V4l2VideoCodec)
#endif

} // namespace hal
} // namespace ai_stream