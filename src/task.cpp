#ifndef PROP_VALUE_MAX
#define PROP_VALUE_MAX 92
#endif

#include "task.h"
#include "yolo11.h"

namespace ZhangChao
{
	class bTask : public Task
	{
	private:
		Yolov11* model = nullptr;

	public:
		~bTask() override
		{
			delete model;
			std::cout << "bTask destructor!" << std::endl;
		}

		bool infer(cv::Mat& bgr, float confidence_threshold, float nms_threshold, std::vector<int> filter,
			std::vector<ObjectCLs>& objects) override
		{
			objects.clear();
			if (!bgr.isContinuous())
			{
				bgr = bgr.clone();
			}
			std::vector<Object> yolo_objects;
			model->detect(bgr, yolo_objects, confidence_threshold, nms_threshold);

			for (auto& yolo_object : yolo_objects)
			{
				ObjectCLs objc;
				objc.x = yolo_object.rect.x;
				objc.y = yolo_object.rect.y;
				objc.w = yolo_object.rect.width;
				objc.h = yolo_object.rect.height;
				objc.labelId = yolo_object.label;
				objc.prob = yolo_object.prob;
				objc.clsId = -1;
				objc.clsProb = -1;
				objects.push_back(objc);
			}

			return true;
		}

		bool load(const std::string& yolo_param_path, const std::string& yolo_bin_path, int yolo_input_size,
			unsigned char yolo_param_key, unsigned char yolo_bin_key, bool isGPU)
		{
			model = new Yolov11();
			if (!model->load_model(yolo_param_path.c_str(), yolo_bin_path.c_str(), yolo_input_size, isGPU,
				yolo_param_key, yolo_bin_key))
			{
				std::cerr << "load YOLOv5 model failed" << std::endl;
				return false;
			}
			std::cout << "load model success!" << std::endl;
			return true;
		}
	};

	std::shared_ptr<Task>
	load(const std::string& yolo_param_path, const std::string& yolo_bin_path, int yolo_input_size,
			unsigned char yolo_param_key, unsigned char yolo_bin_key, bool isGPU)
	{
		auto* task = new bTask();
		if (!task->load(yolo_param_path, yolo_bin_path, yolo_input_size, yolo_param_key, yolo_bin_key, isGPU))
		{
			delete task;
			return nullptr;
		}

		return std::shared_ptr<Task>(task);
	}
}