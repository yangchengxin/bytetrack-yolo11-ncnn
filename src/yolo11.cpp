#include "yolo11.h"

static std::tuple<cv::Mat, std::pair<double, double>, std::pair<double, double>>
letterbox(cv::Mat im, cv::Size new_shape = cv::Size(640, 640),
    cv::Scalar color = cv::Scalar(114, 114, 114),
    bool auto_ = true, bool scaleFill = false, bool scaleup = true, int stride = 32) {
    // Resize and pad image while meeting stride-multiple constraints
    cv::Size shape = im.size();  // current shape [width, height] - Note: OpenCV size is width-first

    if (new_shape.width == 0) new_shape.width = new_shape.height;
    if (new_shape.height == 0) new_shape.height = new_shape.width;

    // Scale ratio (new / old)
    double r = std::min(new_shape.height / static_cast<double>(shape.height),
        new_shape.width / static_cast<double>(shape.width));
    if (!scaleup) {  // only scale down, do not scale up (for better val mAP)
        r = std::min(r, 1.0);
    }

    // Compute padding
    std::pair<double, double> ratio(r, r);  // width, height ratios
    cv::Size new_unpad(static_cast<int>(std::round(shape.width * r)),
        static_cast<int>(std::round(shape.height * r)));

    double dw = new_shape.width - new_unpad.width;
    double dh = new_shape.height - new_unpad.height;

    if (auto_) {  // minimum rectangle
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    }
    else if (scaleFill) {  // stretch
        dw = 0.0;
        dh = 0.0;
        new_unpad = new_shape;
        ratio = { new_shape.width / static_cast<double>(shape.width),
                 new_shape.height / static_cast<double>(shape.height) };
    }

    dw /= 2.0;  // divide padding into 2 sides
    dh /= 2.0;

    if (shape != new_unpad) {  // resize
        cv::resize(im, im, new_unpad, 0, 0, cv::INTER_LINEAR);
    }

    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));

    cv::copyMakeBorder(im, im, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return std::make_tuple(im, ratio, std::make_pair(dw, dh));
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float softmax(const float* src, float* dst, int length)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static float clamp(float val, float min = 0.f, float max = 1280.f)
{
    return val > min ? (val < max ? val : max) : min;
}

static void generate_proposals(
    int stride,
    const ncnn::Mat& feat_blob,
    const float prob_threshold,
    std::vector<Object>& objects
)
{
    const int reg_max = 16;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_class = num_w - 4 * reg_max;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {

            const float* matat = feat_blob.channel(i).row(j);

            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int c = 0; c < num_class; c++)
            {
                float score = matat[4 * reg_max + c];
                if (sigmoid(score) > class_score)
                {
                    class_index = c;
                    class_score = sigmoid(score);
                }
            }

            if (class_score >= prob_threshold)
            {

                float x0 = j + 0.5f - softmax(matat, dst, 16);
                float y0 = i + 0.5f - softmax(matat + 16, dst, 16);
                float x1 = j + 0.5f + softmax(matat + 2 * 16, dst, 16);
                float y1 = i + 0.5f + softmax(matat + 3 * 16, dst, 16);

                x0 *= stride;
                y0 *= stride;
                x1 *= stride;
                y1 *= stride;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = class_score;
                objects.push_back(obj);

            }
        }
    }
}

static void non_max_suppression(
    std::vector<Object>& proposals,
    std::vector<Object>& results,
    int orin_h,
    int orin_w,
    float dh = 0,
    float dw = 0,
    float ratio_h = 1.0f,
    float ratio_w = 1.0f,
    float conf_thres = 0.25f,
    float iou_thres = 0.65f
)
{
    results.clear();
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    for (auto& pro : proposals)
    {
        bboxes.push_back(pro.rect);
        scores.push_back(pro.prob);
        labels.push_back(pro.label);
    }

    cv::dnn::NMSBoxes(
        bboxes,
        scores,
        conf_thres,
        iou_thres,
        indices
    );

    for (auto i : indices)
    {
        auto& bbox = bboxes[i];
        float x0 = bbox.x;
        float y0 = bbox.y;
        float x1 = bbox.x + bbox.width;
        float y1 = bbox.y + bbox.height;
        float& score = scores[i];
        int& label = labels[i];

        x0 = (x0 - dw) / ratio_w;
        y0 = (y0 - dh) / ratio_h;
        x1 = (x1 - dw) / ratio_w;
        y1 = (y1 - dh) / ratio_h;

        x0 = clamp(x0, 0.f, orin_w);
        y0 = clamp(y0, 0.f, orin_h);
        x1 = clamp(x1, 0.f, orin_w);
        y1 = clamp(y1, 0.f, orin_h);

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = score;
        obj.label = label;
        results.push_back(obj);
    }
}

bool Yolov11::load_model(const char* param_path, const char* bin_path, int target_size, bool use_gpu,
    unsigned char key1, unsigned char key2)
{
    net_.clear();
    blob_pool_allocator_.clear();
    workspace_pool_allocator_.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net_.opt = ncnn::Option();

#if NCNN_VULKAN
    net_.opt.use_vulkan_compute = ues_gpu;
#endif

    net_.opt.num_threads = ncnn::get_big_cpu_count();
    net_.opt.blob_allocator = &blob_pool_allocator_;
    net_.opt.workspace_allocator = &workspace_pool_allocator_;

    auto ret = net_.load_param(param_path);
    if (ret != 0)
    {
        std::cerr << "fail to load param!" << std::endl;
        return -1;
    }

    ret = net_.load_model(bin_path);
    if (ret != 0)
    {
        std::cerr << "fail to load bin!" << std::endl;
        return -1;
    }

    this->input_size_ = target_size;
    return true;
}

bool Yolov11::detect(const cv::Mat& bgr, std::vector<Object>& objects, float prob_threshold,
    float nms_threshold, bool is_video)
{
    objects.clear();
    
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)this->input_size_ / w;
        w = this->input_size_;
        h = h * scale;
    }
    else
    {
        scale = (float)this->input_size_ / h;
        h = this->input_size_;
        w = w * scale;
    }

    // bgr2rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // letter box
    int wpad = this->input_size_ - w;
    int hpad = this->input_size_ - h;
    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in,
        in_pad,
        top,
        bottom,
        left,
        right,
        ncnn::BORDER_CONSTANT,
        114.f);

    // normalize
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", in_pad);
    std::vector<Object> proposals;

    // stride 8 
    {
        ncnn::Mat out;
        ex.extract("out0", out);
        std::vector<Object> objects8;
        generate_proposals(8, out, prob_threshold, objects8);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16 
    {
        ncnn::Mat out;
        ex.extract("out1", out);
        std::vector<Object> objects16;
        generate_proposals(16, out, prob_threshold, objects16);
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32 
    {
        ncnn::Mat out;
        ex.extract("out2", out);
        std::vector<Object> objects32;
        generate_proposals(32, out, prob_threshold, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    for (auto& pro : proposals)
    {
        float x0 = pro.rect.x;
        float y0 = pro.rect.y;
        float x1 = pro.rect.x + pro.rect.width;
        float y1 = pro.rect.y + pro.rect.height;
        float& score = pro.prob;
        int& label = pro.label;

        x0 = (x0 - (wpad / 2)) / scale;
        y0 = (y0 - (hpad / 2)) / scale;
        x1 = (x1 - (wpad / 2)) / scale;
        y1 = (y1 - (hpad / 2)) / scale;

        x0 = clamp(x0, 0.f, img_w);
        y0 = clamp(y0, 0.f, img_h);
        x1 = clamp(x1, 0.f, img_w);
        y1 = clamp(y1, 0.f, img_h);

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = score;
        obj.label = label;
        objects.push_back(obj);
    }

    non_max_suppression(proposals, objects,
        img_h, img_w, hpad / 2, wpad / 2,
        scale, scale, prob_threshold, nms_threshold);

    return 0;
}

Yolov11::Yolov11()
{
    blob_pool_allocator_.set_size_compare_ratio(0.f);
    workspace_pool_allocator_.set_size_compare_ratio(0.f);
}

Yolov11::~Yolov11()
{
    net_.clear();
    blob_pool_allocator_.clear();
    workspace_pool_allocator_.clear();
    history_.clear();
}