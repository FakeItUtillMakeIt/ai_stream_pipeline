// src/nodes/track/tracker_factory.h
#pragma once

#include "base_tracker.h"
#include "ai_stream/nodes/i_tracker_node.h"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace ai_stream {
namespace nodes {

using TrackerCreator = std::function<TrackerPtr()>;

class TrackerFactory {
public:
    static TrackerFactory& instance() {
        static TrackerFactory factory;
        return factory;
    }
    
    void registerCreator(TrackerType type, TrackerCreator creator) {
        creators_[type] = std::move(creator);
    }
    
    TrackerPtr create(TrackerType type) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
private:
    TrackerFactory() = default;
    std::map<TrackerType, TrackerCreator> creators_;
};

#define REGISTER_TRACKER(type, class_name) \
    static struct TrackerRegistrar_##class_name { \
        TrackerRegistrar_##class_name() { \
            TrackerFactory::instance().registerCreator( \
                TrackerType::type, \
                []() -> TrackerPtr { return std::make_shared<class_name>(); } \
            ); \
        } \
    } _tracker_registrar_##class_name;

} // namespace nodes
} // namespace ai_stream