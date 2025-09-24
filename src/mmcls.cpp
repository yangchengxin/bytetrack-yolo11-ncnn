/**
 * @author mpj
 * @date 2025/5/7 23:53
 * @version V1.0
 * @since C++11
**/
#include <fstream>
#include <ncnn/datareader.h>
#include <ncnn/cpu.h>
#include "mmcls.h"
#include "common.h"

MMCls::MMCls() {
    blob_pool_allocator_.set_size_compare_ratio(0.f);
    workspace_pool_allocator_.set_size_compare_ratio(0.f);
}

bool MMCls::load_model(const char *param_path, const char *bin_path, int input_size, bool use_gpu,
                       unsigned char key1, unsigned char key2) {
    net_.clear();
    blob_pool_allocator_.clear();
    workspace_pool_allocator_.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net_.opt = ncnn::Option();

#if NCNN_VULKAN
    net_.opt.use_vulkan_compute = use_gpu;
#endif

    net_.opt.num_threads = ncnn::get_big_cpu_count();
    net_.opt.blob_allocator = &blob_pool_allocator_;
    net_.opt.workspace_allocator = &workspace_pool_allocator_;

//	auto ret = net_.load_param(param_path);
//	if (ret != 0) {
//		LOGE("load_param %s ret=%d", param_path, ret);
//		return -1;
//	}
//	LOGD("load_param %s ret=%d", param_path, ret);
//	ret = net_.load_model(bin_path);
//	if (ret != 0) {
//		LOGE("load_model %s ret=%d", bin_path, ret);
//		return -1;
//	}
//	LOGD("load_model %s ret=%d", bin_path, ret);

    MyEncryptedDataReader param_reader(param_path, key1, true);
    auto ret = net_.load_param(param_reader);
    if (ret != 0) {
        //LOGE("load_param %s ret=%d", param_path, ret);
        return -1;
    }
    //LOGD("load_param %s ret=%d", param_path, ret);
    MyEncryptedDataReader model_reader(bin_path, key2);
    ret = net_.load_model(model_reader);
    if (ret != 0) {
        //LOGE("load_model %s ret=%d", bin_path, ret);
        return -1;
    }
    //LOGD("load_model %s ret=%d", bin_path, ret);

    this->input_size_ = input_size;
    return true;
}

bool MMCls::detect(const cv::Mat &rgb, std::vector<ClassifyOutput> &result) {
    result.clear();

    // 根据短边缩放到resize_size_，长边等比例缩放
    float scale = (float) resize_size_ / std::min(rgb.cols, rgb.rows);
    int scale_w = rgb.cols * scale;
    int scale_h = rgb.rows * scale;
    cv::Mat rgb_resized;
    cv::resize(rgb, rgb_resized, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_LINEAR);
    // y1 = max(0, int(round((img_height - crop_height) / 2.)))
    // x1 = max(0, int(round((img_width - crop_width) / 2.)))
    // y2 = min(img_height, y1 + crop_height) - 1
    // x2 = min(img_width, x1 + crop_width) - 1
    int x1 = std::max(0, (int) round((scale_w - input_size_) / 2.));
    int y1 = std::max(0, (int) round((scale_h - input_size_) / 2.));
    int x2 = std::min(scale_w, x1 + input_size_) - 1;
    int y2 = std::min(scale_h, y1 + input_size_) - 1;
    cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    cv::Mat in_roi = rgb_resized(roi);
    // 是否连续
    if (!in_roi.isContinuous()) {
        in_roi = in_roi.clone();
    }
    ncnn::Mat in = ncnn::Mat::from_pixels(in_roi.data, ncnn::Mat::PIXEL_RGB, input_size_, input_size_);
    // mean [123.675, 116.280, 103.530]
    // std [58.395, 57.120, 57.375]
    const float mean_vals[3] = {123.675f, 116.280f, 103.530f};
    const float norm_vals[3] = {1 / 58.395f, 1 / 57.120f, 1 / 57.375f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // inference
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);

    // post process
    std::vector<ClassifyOutput> outputs(out.w);
    for (int i = 0; i < out.w; ++i) {
        ClassifyOutput output{};
        output.label = i;
        output.score = out[i];
        outputs[i] = output;
    }
    // sort
    std::sort(outputs.begin(), outputs.end(), [](const ClassifyOutput &a, const ClassifyOutput &b) {
        return a.score > b.score;
    });
    // 取前5个
    if (outputs.size() > 5) {
        outputs.resize(5);
    }
    for (int i = 0; i < 5; ++i) {
        ClassifyOutput output{};
        output.label = outputs[i].label;
        output.score = outputs[i].score;
        result.push_back(output);
    }

    return true;
}

bool MMCls::detect(const cv::Mat &rgb, ClassifyOutput &result) {
// 根据短边缩放到resize_size_，长边等比例缩放
    float scale = (float) resize_size_ / std::min(rgb.cols, rgb.rows);
    int scale_w = rgb.cols * scale;
    int scale_h = rgb.rows * scale;
    cv::Mat rgb_resized;
    cv::resize(rgb, rgb_resized, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_LINEAR);
    // y1 = max(0, int(round((img_height - crop_height) / 2.)))
    // x1 = max(0, int(round((img_width - crop_width) / 2.)))
    // y2 = min(img_height, y1 + crop_height) - 1
    // x2 = min(img_width, x1 + crop_width) - 1
    int x1 = std::max(0, (int) round((scale_w - input_size_) / 2.));
    int y1 = std::max(0, (int) round((scale_h - input_size_) / 2.));
    int x2 = std::min(scale_w, x1 + input_size_) - 1;
    int y2 = std::min(scale_h, y1 + input_size_) - 1;
    cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    cv::Mat in_roi = rgb_resized(roi);
    // 是否连续
    if (!in_roi.isContinuous()) {
        in_roi = in_roi.clone();
    }
    ncnn::Mat in = ncnn::Mat::from_pixels(in_roi.data, ncnn::Mat::PIXEL_RGB, input_size_, input_size_);
    // mean [123.675, 116.280, 103.530]
    // std [58.395, 57.120, 57.375]
    const float mean_vals[3] = {123.675f, 116.280f, 103.530f};
    const float norm_vals[3] = {1 / 58.395f, 1 / 57.120f, 1 / 57.375f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // inference
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("probs", out);

    // post process
    int max_index = -1;
    float max_score = -1;
    for (int i = 0; i < out.w; ++i) {
        if (out[i] > max_score) {
            max_score = out[i];
            max_index = i;
        }
    }
    result.label = max_index;
    result.score = max_score;

    return true;
}

MMCls::~MMCls() {
    net_.clear();
    blob_pool_allocator_.clear();
    workspace_pool_allocator_.clear();
}
