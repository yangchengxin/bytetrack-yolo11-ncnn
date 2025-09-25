#ifndef PROP_VALUE_MAX
#define PROP_VALUE_MAX 92
#endif

#include "task.h"
#include "yolo11.h"
#include "byte_track/BYTETracker.h"

namespace ZhangChao
{
	class bTask : public Task
	{
	private:
		Yolov11* model = nullptr;
		BYTETracker* tracker = nullptr;

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

			std::vector<STrack> tracks = tracker->update(yolo_objects);

			for (auto& track : tracks)
			{
				ObjectCLs obj;
				obj.x = track.tlwh[0];
				obj.y = track.tlwh[1];
				obj.w = track.tlwh[2];
				obj.h = track.tlwh[3];
				obj.labelId = track.class_id;
				obj.prob = track.score;
				obj.clsId = -1;
				obj.clsProb = -1;
				obj.trackId = track.track_id;
				objects.push_back(obj);
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
			tracker = new BYTETracker(30, 30);

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