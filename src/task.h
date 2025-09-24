/**
 * @author mpj
 * @date 25-5-16 下午11:35
 * @version V1.0
 * @since C++11
**/

#ifndef ZHANGCHAO_TASK_H
#define ZHANGCHAO_TASK_H

#include <opencv2/opencv.hpp>


namespace ZhangChao {
    struct ObjectCLs {
        float x{};
        float y{};
        float w{};
        float h{};
        int labelId{};
        float prob{};
        int clsId = -1;
        float clsProb = -1;
    };

    class Task {
    public:
        virtual ~Task() = default;
        /**
         * 进行推理
         * @param rgb 一帧rgb格式的图片
         * @param confidence_threshold 置信度阈值
         * @param nms_threshold nms阈值
         * @param filter 需要保留的类别，空对象不过滤
         * @param objects 返回的检测结果
         * @return 是否成功
         */
        virtual bool infer(cv::Mat &bgr, float confidence_threshold, float nms_threshold, std::vector<int> filter,
                           std::vector<ObjectCLs> &objects) = 0;
    };

    /**
     * 实现RAII格式的掌超任务接口
     * @param yolo_param_path yolo的param文件路径
     * @param yolo_bin_path yolo的bin文件路径
     * @param yolo_input_size yolo的模型输入大小
     * @param yolo_param_key yolo的param加密key
     * @param yolo_bin_key yolo的bin加密key
     * @param cls_param_path 分类器的param文件路径
     * @param cls_bin_path 分类器的bin文件路径
     * @param cls_input_size 分类器的模型输入大小
     * @param cls_param_key 分类器的param加密key
     * @param cls_bin_key 分类器的bin加密key
     * @param isGPU 是否使用GPU，默认使用CPU，在安卓中推荐使用功能CPU，安卓的GPU计算能力远不如PC
     * @return 返回一个Task的智能指针
     */
    std::shared_ptr<Task> load(
            const std::string &yolo_param_path,
            const std::string &yolo_bin_path,
            int yolo_input_size,
            unsigned char yolo_param_key = 0,
            unsigned char yolo_bin_key = 0,
            bool isGPU = false);
}

#endif //ZHANGCHAO_TASK_H
