/**
 * @author mpj
 * @date 2025/4/26 15:47
 * @version V1.0
 * @since C++11
**/

#ifndef ZHANGCHAO_COMMON_H
#define ZHANGCHAO_COMMON_H

//#include <jni.h>
//#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <ncnn/datareader.h>

// LOGD实现
#define TAG "NativeLib"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct Object {
    cv::Rect_<float> rect;
    int label{};
    float prob{};
};

struct ClassifyOutput {
    int label;
    float score;
};

void draw_objects(cv::Mat &bgr, const std::vector<Object> &objects);

class Memory {
public:
    Memory() = default;

    Memory(const Memory &) = delete;

    Memory &operator=(const Memory &) = delete;

    Memory(Memory &&) = delete;

    Memory &operator=(Memory &&) = delete;

    ~Memory() {
        if (buffer_) {
            delete[] buffer_;
        }
    }

    void alloc(size_t size) {
        if (buffer_) {
            delete[] buffer_;
        }
        buffer_ = new unsigned char[size];
        size_ = size;
    }

    unsigned char *buffer_ = nullptr;
    size_t buffer_offset_ = 0;
    size_t size_ = 0;
};

class MyEncryptedDataReader : public ncnn::DataReader {
public:
    MyEncryptedDataReader(const char *filepath, unsigned char _key, bool is_param = false);

    ~MyEncryptedDataReader() override;

    // 修改为返回实际读取的字节数，与fread一致
    size_t read(void *buf, size_t size) const override;

    int scan(const char *format, void *p) const override;

private:
    FILE *fp;
    unsigned char key;
    // 创建一个unsigned char类型的数组用来存储读取的数据
    Memory *memory = nullptr;
};

#endif //ZHANGCHAO_COMMON_H
