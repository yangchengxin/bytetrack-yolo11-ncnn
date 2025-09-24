/**
 * @author mpj
 * @date 2025/5/7 23:22
 * @version V1.0
 * @since C++11
**/

#ifndef ZHANGCHAO_YOLOV5_VIDEO_H
#define ZHANGCHAO_YOLOV5_VIDEO_H

#include <string>
//#include <android/asset_manager.h>
#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include "common.h"

class Yolov11 {
public:
    Yolov11();

    ~Yolov11();

    bool load_model(const char* param_path, const char* bin_path, int target_size, bool use_gpu,
        unsigned char key1 = 0, unsigned char key2 = 0);

    bool detect(const cv::Mat& bgr, std::vector<Object>& objects, float prob_threshold = 0.25f,
        float nms_threshold = 0.45f, bool is_video = false);

private:
    ncnn::Net net_;
    int input_size_{};
    std::vector<cv::Mat> history_; // 用于存储历史帧
    ncnn::UnlockedPoolAllocator blob_pool_allocator_;
    ncnn::PoolAllocator workspace_pool_allocator_;
};

#endif //ZHANGCHAO_YOLOV5_VIDEO_H
