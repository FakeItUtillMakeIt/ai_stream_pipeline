// include/ai_stream/nodes/i_source_node.h
#pragma once

#include "ai_stream/core/node.h"

namespace ai_stream {
namespace nodes {

class ISourceNode : public core::Node {
public:
    using core::Node::Node;
    
    // 设置拉流地址
    virtual void setUrl(const std::string& url) = 0;
    
    // 获取当前 URL
    virtual std::string getUrl() const = 0;
};

} // namespace nodes
} // namespace ai_stream