// src/rules/alert/feature_extractor.h
#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>

namespace ai_stream {
namespace rules {

struct FeatureVector {
    std::vector<float> color_hist;
    std::vector<float> grad_hist;
    float mean[3] = {0};
    float stddev[3] = {0};
    bool valid = false;
};

class FeatureExtractor {
public:
    static FeatureVector extract(const cv::Mat& frame, const std::vector<PixelPoint>& zone) {
        FeatureVector feat;
        if (frame.empty()) return feat;

        if (zone.size() < 3) {
            cv::Mat mask = cv::Mat::ones(frame.rows, frame.cols, CV_8UC1) * 255;
            feat.color_hist = extractColorHist(frame, mask);
            feat.grad_hist = extractGradHist(frame, mask);
            extractPixelStats(frame, mask, feat.mean, feat.stddev);
            feat.valid = true;
            return feat;
        }

        cv::Mat mask = createZoneMask(frame.cols, frame.rows, zone);
        cv::Mat roi;
        frame.copyTo(roi, mask);

        feat.color_hist = extractColorHist(roi, mask);
        feat.grad_hist = extractGradHist(roi, mask);
        extractPixelStats(roi, mask, feat.mean, feat.stddev);
        feat.valid = true;

        return feat;
    }

    static FeatureVector extractFromImage(const cv::Mat& image) {
        FeatureVector feat;
        if (image.empty()) return feat;

        std::vector<PixelPoint> full_zone = {
            {0, 0},
            {static_cast<float>(image.cols), 0},
            {static_cast<float>(image.cols), static_cast<float>(image.rows)},
            {0, static_cast<float>(image.rows)}
        };
        return extract(image, full_zone);
    }

    static float computeSimilarity(const FeatureVector& ref, const FeatureVector& cur) {
        if (!ref.valid || !cur.valid) return 0.0f;

        float color_sim = compareHistograms(ref.color_hist, cur.color_hist);
        float grad_sim = compareHistograms(ref.grad_hist, cur.grad_hist);
        float pixel_sim = comparePixelStats(ref.mean, ref.stddev, cur.mean, cur.stddev);

        return 0.5f * color_sim + 0.3f * grad_sim + 0.2f * pixel_sim;
    }

    static void updateReference(FeatureVector& ref, const FeatureVector& cur, float alpha = 0.05f) {
        if (!ref.valid || !cur.valid) return;

        for (size_t i = 0; i < ref.color_hist.size() && i < cur.color_hist.size(); ++i) {
            ref.color_hist[i] = (1.0f - alpha) * ref.color_hist[i] + alpha * cur.color_hist[i];
        }
        for (size_t i = 0; i < ref.grad_hist.size() && i < cur.grad_hist.size(); ++i) {
            ref.grad_hist[i] = (1.0f - alpha) * ref.grad_hist[i] + alpha * cur.grad_hist[i];
        }
        for (int c = 0; c < 3; ++c) {
            ref.mean[c] = (1.0f - alpha) * ref.mean[c] + alpha * cur.mean[c];
            ref.stddev[c] = (1.0f - alpha) * ref.stddev[c] + alpha * cur.stddev[c];
        }
    }

    static void accumulateReference(FeatureVector& ref, const FeatureVector& cur, int count) {
        if (!cur.valid) return;

        if (!ref.valid) {
            ref = cur;
            ref.valid = true;
            return;
        }

        float alpha = 1.0f / (count + 1);
        for (size_t i = 0; i < ref.color_hist.size() && i < cur.color_hist.size(); ++i) {
            ref.color_hist[i] = (1.0f - alpha) * ref.color_hist[i] + alpha * cur.color_hist[i];
        }
        for (size_t i = 0; i < ref.grad_hist.size() && i < cur.grad_hist.size(); ++i) {
            ref.grad_hist[i] = (1.0f - alpha) * ref.grad_hist[i] + alpha * cur.grad_hist[i];
        }
        for (int c = 0; c < 3; ++c) {
            ref.mean[c] = (1.0f - alpha) * ref.mean[c] + alpha * cur.mean[c];
            ref.stddev[c] = (1.0f - alpha) * ref.stddev[c] + alpha * cur.stddev[c];
        }
    }

private:
    static cv::Mat createZoneMask(int width, int height, const std::vector<PixelPoint>& zone) {
        cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
        std::vector<cv::Point> pts;
        for (const auto& p : zone) {
            pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        }
        cv::fillConvexPoly(mask, pts, cv::Scalar(255));
        return mask;
    }

    static std::vector<float> extractColorHist(const cv::Mat& roi, const cv::Mat& mask) {
        cv::Mat hsv;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

        int h_bins = 16, s_bins = 8;
        int hist_size[] = {h_bins, s_bins};
        float h_ranges[] = {0, 180};
        float s_ranges[] = {0, 256};
        const float* ranges[] = {h_ranges, s_ranges};
        int channels[] = {0, 1};

        cv::Mat hist;
        cv::calcHist(&hsv, 1, channels, mask, hist, 2, hist_size, ranges);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

        std::vector<float> result(h_bins * s_bins);
        for (int h = 0; h < h_bins; ++h) {
            for (int s = 0; s < s_bins; ++s) {
                result[h * s_bins + s] = hist.at<float>(h, s);
            }
        }
        return result;
    }

    static std::vector<float> extractGradHist(const cv::Mat& roi, const cv::Mat& mask) {
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1);

        cv::Mat mag, angle;
        cv::cartToPolar(grad_x, grad_y, mag, angle, true);

        int bins = 9;
        std::vector<float> hist(bins, 0.0f);
        float total = 0.0f;

        for (int y = 0; y < gray.rows; ++y) {
            for (int x = 0; x < gray.cols; ++x) {
                if (mask.at<uchar>(y, x) == 0) continue;
                float a = angle.at<float>(y, x);
                float m = mag.at<float>(y, x);
                int bin = static_cast<int>(a / 20.0f) % bins;
                hist[bin] += m;
                total += m;
            }
        }

        if (total > 0) {
            for (auto& v : hist) v /= total;
        }
        return hist;
    }

    static void extractPixelStats(const cv::Mat& roi, const cv::Mat& mask,
                                   float mean[3], float stddev[3]) {
        cv::Scalar m, s;
        cv::meanStdDev(roi, m, s, mask);
        for (int c = 0; c < 3; ++c) {
            mean[c] = static_cast<float>(m[c]);
            stddev[c] = static_cast<float>(s[c]);
        }
    }

    static float compareHistograms(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-6f || norm_b < 1e-6f) return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    static float comparePixelStats(const float ref_mean[3], const float ref_std[3],
                                    const float cur_mean[3], const float cur_std[3]) {
        float sim = 0.0f;
        for (int c = 0; c < 3; ++c) {
            float mean_diff = std::abs(ref_mean[c] - cur_mean[c]) / 255.0f;
            float std_diff = std::abs(ref_std[c] - cur_std[c]) / 128.0f;
            sim += std::exp(-(mean_diff + std_diff));
        }
        return sim / 3.0f;
    }
};

} // namespace rules
} // namespace ai_stream
