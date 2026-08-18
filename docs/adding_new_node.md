# 如何新增自定义节点

本文演示如何开发一个自定义节点并接入管道。以"灰度转换节点"为例。

## 1. 选择基类

| 场景 | 基类 |
|---|---|
| 轻处理（透传/查表/分发），不阻塞上游 | `core::Node`（同步 pushData） |
| 重处理（解码/推理/绘制等），需要削峰背压 | `core::QueuedNode<接口>`（输入队列 + worker 线程） |

接口类（`include/ai_stream/nodes/i_*.h`）定义了各类节点的通用配置解析
（`configure()`）与 setter，**优先继承接口**：
`QueuedNode<IDrawNode>`、`QueuedNode<IPreprocessNode>` 等。

## 2. 头文件

```cpp
// src/nodes/gray/gray_node.h
#pragma once
#include "ai_stream/core/queued_node.h"
#include "ai_stream/core/node.h"

namespace ai_stream {
namespace nodes {

class GrayNode : public core::QueuedNode<core::Node> {
public:
    GrayNode() : core::QueuedNode<core::Node>("GrayNode") {}

    // QueuedNode 钩子：原 start/stop 中的资源初始化/释放
    bool onStartup() override;
    void onShutdown() override;

    // 自身参数解析（"queue" 字段已由 QueuedNode 基类处理）
    bool configureImpl(const std::string& node_id, const nlohmann::json& params) override;

protected:
    // worker 线程中串行执行（原 pushData 逻辑）
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;

private:
    bool invert_ = false;
};

} // namespace nodes
} // namespace ai_stream
```

> 同步节点则直接继承 `core::Node`（或接口），实现 `start/stop/pushData/configure`，参考
> `examples/custom_node/`。

## 3. 实现文件

```cpp
// src/nodes/gray/gray_node.cpp
#include "gray_node.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>

namespace ai_stream {
namespace nodes {

bool GrayNode::configureImpl(const std::string& node_id, const nlohmann::json& params) {
    (void)node_id;
    invert_ = params.value("invert", false);
    return true;   // 返回 false 会终止整个管道构建
}

bool GrayNode::onStartup() { return true; }
void GrayNode::onShutdown() {}

void GrayNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    // STREAM_END 标准处理：自停并转发
    if (packet->type == core::PacketType::STREAM_END) {
        stop();
        broadcast(packet);
        return;
    }
    if (packet->type != core::PacketType::DECODED_FRAME) {
        broadcast(packet);   // 非本节点处理的类型直接透传
        return;
    }

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame->mat || frame->mat->empty()) return;

    // ... 处理逻辑：cv::cvtColor 等 ...

    // 耗时打点（broadcast 时自动记入 MetricsCollector）
    packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
    packet->cost_time_map[name_] = packet->cost_ms;
    broadcast(packet);
}

// 注册类型名（JSON 中的 "type"），同文件可注册多个
REGISTER_NODE("gray", GrayNode)

} // namespace nodes
} // namespace ai_stream
```

要点：
- `broadcast()` 推给所有下游，同时记录指标、清理失效下游
- `in_time_ms_` 由 QueuedNode 在出队时自动刷新
- 修改帧数据前先克隆（若该帧还会被其他分支使用）

## 4. 加入构建

`src/nodes/CMakeLists.txt` 的 `NODES_SOURCES` 追加：

```cmake
gray/gray_node.cpp
```

## 5. 管道配置使用

```json
{
  "id": "gray_demo",
  "graph": {
    "nodes": [
      { "id": "src1", "type": "file_source",
        "params": { "url": "file:///path/to/video.mp4", "loop": true } },
      { "id": "decode1", "type": "ffmpeg_decode", "params": { "output_bgr": true } },
      { "id": "gray1", "type": "gray",
        "params": { "invert": false,
                    "queue": { "capacity": 32, "drop_policy": "drop_newest" } } }
    ],
    "edges": [
      { "from": "src1", "to": "decode1" },
      { "from": "decode1", "to": "gray1" }
    ]
  }
}
```

## 6. 检查清单

- [ ] 继承正确的基类/接口，重处理用 QueuedNode
- [ ] `configure/configureImpl` 解析全部参数，非法配置返回 false
- [ ] STREAM_END 处理（stop + broadcast）
- [ ] 非目标类型包透传而非丢弃
- [ ] `REGISTER_NODE` 注册且类型名唯一
- [ ] CMake 源文件列表已添加
- [ ] 重启管道场景：`onStartup` 可重复执行（队列由基类 reset）
