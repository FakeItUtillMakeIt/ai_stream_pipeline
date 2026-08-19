// src/core/pipeline_manager.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "ai_stream/core/pipeline.h"
#include "nodes/registry/node_factory.h" // 内部工厂头文件
#include <nlohmann/json.hpp>
#include <queue>
#include <unordered_map>

namespace ai_stream
{
    namespace core
    {

        Pipeline::~Pipeline()
        {
            stop();
        }

        bool Pipeline::buildFromJson(const nlohmann::json &config)
        {
            try
            {
                // 1. 解析节点配置并创建实例
                if (!config.contains("nodes") || !config["nodes"].is_array())
                {
                    LOG_ERROR_FMT("[Pipeline {}] Invalid config: missing 'nodes' array", id_);
                    return false;
                }

                for (const auto &node_cfg : config["nodes"])
                {
                    std::string id = node_cfg.value("id", "");
                    std::string type = node_cfg.value("type", "");
                    nlohmann::json params = node_cfg.value("params", nlohmann::json::object());

                    LOG_INFO_FMT("[Pipeline {}] Node: id:{},type:{}", id_, id, type);

                    if (id.empty() || type.empty())
                    {
                        LOG_ERROR_FMT("[Pipeline {}] Node missing 'id' or 'type'", id_);
                        return false;
                    }

                    if (nodes_.find(id) != nodes_.end())
                    {
                        LOG_ERROR_FMT("[Pipeline {}] Duplicate node id: {}", id_, id);
                        return false;
                    }

                    auto node = NodeFactory::instance().create(type, params);
                    if (!node)
                    {
                        LOG_ERROR_FMT("[Pipeline {}] Failed to create node type: {}", id_, type);
                        return false;
                    }

                    // 节点自行解析参数（模型加载等初始化也在此完成）
                    if (!node->configure(id, params))
                    {
                        LOG_ERROR_FMT("[Pipeline {}] Failed to configure node: {} (type: {})", id_, id, type);
                        return false;
                    }

                    // 节点名统一为配置 id：日志/指标/producer_id 均以此区分同类节点实例
                    node->setName(id);

                    nodes_[id] = node;
                    LOG_INFO_FMT("[Pipeline {}] Created node: {} (type: {})", id_, id, type);
                }

                // 2. 建立边连接（一流多用的关键）
                std::vector<std::pair<std::string, std::string>> edge_list;
                if (config.contains("edges") && config["edges"].is_array())
                {
                    for (const auto &edge : config["edges"])
                    {
                        std::string from = edge.value("from", "");
                        std::string to = edge.value("to", "");

                        auto it_from = nodes_.find(from);
                        auto it_to = nodes_.find(to);
                        if (it_from == nodes_.end() || it_to == nodes_.end())
                        {
                            LOG_ERROR_FMT("[Pipeline {}] Invalid edge: {} -> {}", id_, from, to);
                            return false;
                        }

                        it_from->second->addDownstream(it_to->second);
                        edge_list.emplace_back(from, to);
                        LOG_INFO_FMT("[Pipeline {}] Connected: {} -> {}", id_, from, to);
                    }
                }

                // 3. 计算拓扑序（用于启停顺序），并检测环
                if (!computeTopologicalOrder(edge_list))
                {
                    return false;
                }

                // 4. 为每个节点设置管道弱引用（便于节点访问全局上下文）
                auto weak_self = weak_from_this();
                for (auto &[_, node] : nodes_)
                {
                    node->setPipeline(weak_self);
                }

                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR_FMT("[Pipeline {}] Exception during build: {}", id_, e.what());
                return false;
            }
        }

        bool Pipeline::computeTopologicalOrder(const std::vector<std::pair<std::string, std::string>> &edges)
        {
            std::unordered_map<std::string, int> in_degree;
            std::unordered_map<std::string, std::vector<std::string>> adjacency;
            for (const auto &[name, node] : nodes_)
            {
                in_degree[name] = 0;
            }
            for (const auto &[from, to] : edges)
            {
                adjacency[from].push_back(to);
                in_degree[to]++;
            }

            std::queue<std::string> ready;
            for (const auto &[name, deg] : in_degree)
            {
                if (deg == 0)
                {
                    ready.push(name);
                }
            }

            topo_order_.clear();
            while (!ready.empty())
            {
                std::string name = ready.front();
                ready.pop();
                topo_order_.push_back(name);
                for (const auto &next : adjacency[name])
                {
                    if (--in_degree[next] == 0)
                    {
                        ready.push(next);
                    }
                }
            }

            if (topo_order_.size() != nodes_.size())
            {
                LOG_ERROR_FMT("[Pipeline {}] Cycle detected in pipeline graph", id_);
                return false;
            }
            return true;
        }

        bool Pipeline::start()
        {
            if (running_)
            {
                // 仍有节点在运行 → 视为已启动；
                // 全部节点已停止（如 STREAM_END 级联自停）→ 允许重新启动
                std::lock_guard<std::mutex> lock(mutex_);
                for (const auto &[name, node] : nodes_)
                {
                    if (node->isRunning())
                    {
                        return true;
                    }
                }
                running_ = false;
            }

            // 按逆拓扑序启动：下游（sink/draw）先就绪，source 最后启动，
            // 避免数据到达未启动的节点被丢弃
            bool all_started = true;
            for (auto it = topo_order_.rbegin(); it != topo_order_.rend(); ++it)
            {
                auto &node = nodes_[*it];
                LOG_INFO_FMT("[Pipeline {}] Starting node: {}", id_, *it);
                if (!node->start())
                {
                    LOG_ERROR_FMT("[Pipeline {}] Failed to start node: {}", id_, *it);
                    all_started = false;
                    break;
                }
            }

            if (all_started)
            {
                running_ = true;
                LOG_INFO_FMT("[Pipeline {}] All nodes started successfully", id_);
                return true;
            }

            // 回滚：按拓扑序停止所有节点（stop() 因 running_ 未置位不会生效，需显式执行）
            for (const auto &name : topo_order_)
            {
                auto it = nodes_.find(name);
                if (it != nodes_.end())
                {
                    it->second->stop();
                }
            }
            return false;
        }

        void Pipeline::stop()
        {
            if (!running_)
                return;
            running_ = false;

            // 按拓扑序停止：source 最先停止，阻止新数据进入，下游自然排空
            for (const auto &name : topo_order_)
            {
                auto it = nodes_.find(name);
                if (it == nodes_.end())
                {
                    continue;
                }
                LOG_INFO_FMT("[Pipeline {}] Stopping node: {}", id_, name);
                it->second->stop();
            }
            LOG_INFO_FMT("[Pipeline {}] Pipeline stopped", id_);
        }

        std::shared_ptr<Node> Pipeline::getNode(const std::string &name)
        {
            auto it = nodes_.find(name);
            if (it != nodes_.end())
            {
                return it->second;
            }
            return nullptr;
        }

        bool Pipeline::isRunning() const{
            std::lock_guard<std::mutex> lock(mutex_);
            for(const auto& node : nodes_)
            {
                if(node.second->isRunning())
                {
                    return true;
                }
            }
            return false;
        }

    } // namespace core
} // namespace ai_stream
