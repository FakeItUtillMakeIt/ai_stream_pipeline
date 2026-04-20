#ifndef _CV_CPP_UTILS_HPP_
#define _CV_CPP_UTILS_HPP_

#include <iostream>
#include <tuple>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <atomic>
//#include "utils.h"
namespace ai
{
    namespace cvUtil
    {
        using namespace std;

        struct PoseBox
        {
            float left, top, right, bottom, confidence;
            int class_label;
            int track_id;

            PoseBox() = default;
            PoseBox(float left, float top, float right, float bottom, float confidence, int class_label, int track_id)
                : left(left),
                  top(top),
                  right(right),
                  bottom(bottom),
                  confidence(confidence),
                  class_label(class_label),
                  track_id(track_id) {}
        };

        typedef std::vector<PoseBox> PoseBoxArray;
        typedef std::vector<PoseBoxArray> BatchPoseBoxArray;

    }
}

#endif // _CV_CPP_UTILS_HPP_