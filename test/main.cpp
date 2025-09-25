#include <opencv2/core/utils/logger.hpp>
#include "task.h"

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// 定义常用颜色数组 (BGR 格式)
static const cv::Scalar COLOR_PALETTE[] = {
    cv::Scalar(255, 0, 0),     // 蓝色 (Blue)
    cv::Scalar(0, 255, 0),     // 绿色 (Green)
    cv::Scalar(0, 0, 255),     // 红色 (Red)
    cv::Scalar(0, 255, 255),   // 黄色 (Yellow)
    cv::Scalar(255, 255, 0),   // 青色 (Cyan)
    cv::Scalar(255, 0, 255),   // 品红色 (Magenta)
    cv::Scalar(0, 0, 0),       // 黑色 (Black)
    cv::Scalar(255, 255, 255), // 白色 (White)
    cv::Scalar(128, 128, 128), // 灰色 (Gray)
    cv::Scalar(0, 165, 255),   // 橙色 (Orange)
    cv::Scalar(255, 192, 203), // 粉红色 (Pink)
    cv::Scalar(0, 128, 128),   // 深青色 (Teal)
    cv::Scalar(128, 0, 128),   // 紫色 (Purple)
    cv::Scalar(128, 0, 0),     // 深蓝色 (Navy)
    cv::Scalar(0, 128, 0)      // 深绿色 (Dark Green)
};

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    const std::string yolo_param_path = "../assets/yolo11n_ncnn_model/model.ncnn.param";
    const std::string yolo_bin_path = "../assets/yolo11n_ncnn_model/model.ncnn.bin";
    const std::string video_path = "palace.mp4";
    const std::string output_path = "output.mp4";

    auto task = ZhangChao::load(
        yolo_param_path,
        yolo_bin_path,
        640
        );

    if (task == nullptr)
    {
        std::cerr << "load zhang chao task failed" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "cv::VideoCapture " << video_path << " failed" << std::endl;
        return -1;
    }
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    std::cout << "fps: " << fps << std::endl;
    std::cout << "cv::CAP_PROP_FRAME_COUNT " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    std::vector<int> fourcc_list = {
        cv::VideoWriter::fourcc('H', '2', '6', '4'),  // 另一种H.264表示
        cv::VideoWriter::fourcc('X', '2', '6', '4'),  // 另一种H.264
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
    };

    std::vector<ZhangChao::ObjectCLs> objects;
    long long times = 0;
    int count = 0;
    while (true) {
        cv::Mat bgr;
        cap >> bgr;
        if (bgr.empty()) {
            break;
        }
        count++;
        auto start = std::chrono::high_resolution_clock::now();
        task->infer(bgr, 0.25f, 0.45f, {}, objects);
        auto end = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << count << "帧：" << "detect objects: " << objects.size() << " detect time: " << t / 1000.0 << " ms" << std::endl;
        times += t;

        // draw objects
        for (const auto& object : objects) {
            cv::Rect rect(object.x, object.y, object.w, object.h);
            cv::rectangle(bgr, rect, COLOR_PALETTE[object.labelId], 2);
            // 置信度保留2位小数
            std::string label = cv::format("%s %.2f %d", class_names[object.labelId], object.prob, object.trackId);
            cv::putText(bgr, label, cv::Point(object.x, object.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PALETTE[object.labelId], 2);
        }

        // writer << rgb;

        cv::namedWindow("show", cv::WINDOW_NORMAL);
        cv::imshow("show", bgr);
        cv::waitKey(fps);
    }
    if (times > 0) {
        std::cout << "average time: " << times / count / 1000.0 << " ms" << std::endl;
    }

}