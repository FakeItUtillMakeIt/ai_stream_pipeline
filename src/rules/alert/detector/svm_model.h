#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

struct LinearSVMModel {
    std::vector<float> weights;
    float bias = 0.0f;
    int feature_dim = 0;
    std::vector<float> scaler_mean;
    std::vector<float> scaler_scale;
    bool use_scaler = false;

    float predictRaw(const std::vector<float>& features) const {
        if (features.size() != static_cast<size_t>(feature_dim)) return 0.0f;
        std::vector<float> normalized(feature_dim);
        if (use_scaler && scaler_mean.size() == features.size() && scaler_scale.size() == features.size()) {
            for (size_t i = 0; i < features.size(); ++i) {
                float s = scaler_scale[i];
                normalized[i] = (features[i] - scaler_mean[i]) / (s > 1e-6f ? s : 1.0f);
            }
        } else {
            normalized = features;
        }
        float score = bias;
        for (size_t i = 0; i < normalized.size(); ++i) {
            score += weights[i] * normalized[i];
        }
        return score;
    }

    float predictProbability(const std::vector<float>& features) const {
        float raw = predictRaw(features);
        return 1.0f / (1.0f + std::exp(-raw));
    }

    bool load(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        std::string line;
        bool header_found = false;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(iss, token, ',')) {
                token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
                tokens.push_back(token);
            }

            if (tokens.empty()) continue;

            if (tokens[0] == "type" || tokens[0] == "type\r") {
                header_found = true;
                continue;
            }

            if (!header_found) continue;

            std::string type;
            int idx;
            float value;
            if (tokens.size() >= 3) {
                type = tokens[0];
                idx = std::stoi(tokens[1]);
                value = std::stof(tokens[2]);
            } else {
                continue;
            }

            if (type == "dim") {
                feature_dim = idx;
            } else if (type == "weight") {
                if (idx >= static_cast<int>(weights.size())) {
                    weights.resize(idx + 1, 0.0f);
                }
                weights[idx] = value;
            } else if (type == "bias") {
                bias = value;
            }
        }

        return (weights.size() == static_cast<size_t>(feature_dim));
    }

    bool loadScaler(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;
        nlohmann::json j;
        file >> j;
        if (!j.contains("mean") || !j.contains("scale")) return false;
        scaler_mean = j["mean"].get<std::vector<float>>();
        scaler_scale = j["scale"].get<std::vector<float>>();
        if (scaler_mean.size() == static_cast<size_t>(feature_dim) &&
            scaler_scale.size() == static_cast<size_t>(feature_dim)) {
            use_scaler = true;
            return true;
        }
        return false;
    }
};