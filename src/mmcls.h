/**
 * @author mpj
 * @date 2025/5/7 23:53
 * @version V1.0
 * @since C++11
**/

#ifndef ZHANGCHAO_MMCLS_H
#define ZHANGCHAO_MMCLS_H

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/layer.h>
#include "common.h"

class MMCls {
public:
    MMCls();

    ~MMCls();

    bool load_model(const char *param_path, const char *bin_path, int input_size, bool use_gpu,
                    unsigned char key1 = 0, unsigned char key2 = 0);

    bool detect(const cv::Mat &rgb, std::vector<ClassifyOutput> &result);

    bool detect(const cv::Mat &rgb, ClassifyOutput &result);

private:
    ncnn::Net net_;
    const int resize_size_{256};
    int input_size_{};
    ncnn::UnlockedPoolAllocator blob_pool_allocator_;
    ncnn::PoolAllocator workspace_pool_allocator_;
};

#endif //ZHANGCHAO_MMCLS_H
