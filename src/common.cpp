/**
 * @author mpj
 * @date 2025/5/8 13:27
 * @version V1.0
 * @since C++11
**/
#include "common.h"

void draw_objects(cv::Mat &bgr, const std::vector<Object> &objects) {
    for (const auto &obj: objects) {
        cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0), 2);

        char text[256];
        sprintf(text, "%d %.1f%%", obj.label, obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

MyEncryptedDataReader::MyEncryptedDataReader(const char *filepath, unsigned char _key,
                                             bool is_param)
        : fp(nullptr), key(_key) {
    fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "open file %s failed\n", filepath);
    }
    if (is_param) {
        // 获取文件大小
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        // 分配内存
        memory = new Memory();
        memory->alloc(file_size);
        // 读取文件内容
        size_t nread = fread(memory->buffer_, 1, file_size, fp);
        if (nread == 0) {
            fprintf(stderr, "read file %s failed\n", filepath);
        }
        // xor decrypt
        for (size_t i = 0; i < nread; i++) {
            memory->buffer_[i] ^= key;
        }
        // 将文件指针移动到文件开头
        fseek(fp, 0, SEEK_SET);
    }
}

MyEncryptedDataReader::~MyEncryptedDataReader() {
    if (fp) {
        fclose(fp);
        fp = nullptr;
    }
    key = 0;
    if (memory) {
        delete memory;
        memory = nullptr;
    }
}

size_t MyEncryptedDataReader::read(void *buf, size_t size) const {
    if (!fp) return 0;

    size_t nread = fread(buf, 1, size, fp);
    if (nread == 0) return 0;

    // xor decrypt
    unsigned char *p = static_cast<unsigned char *>(buf);
    for (size_t i = 0; i < nread; i++) {
        p[i] ^= key;
    }

    return nread;
}

int MyEncryptedDataReader::scan(const char *format, void *p) const {
    // 通过读取buffer中的数据来实现scan功能
    if (!fp || !memory) return 0;

    size_t fmtlen = strlen(format);

    char *format_with_n = new char[fmtlen + 4];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf((const char *) memory->buffer_ + memory->buffer_offset_, format_with_n, p, &nconsumed);
    memory->buffer_offset_ += nconsumed;

    delete[] format_with_n;

    return nconsumed > 0 ? nscan : 0;
}
