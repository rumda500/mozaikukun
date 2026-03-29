#ifndef SORT_H
#define SORT_H

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <set>

#include "ort-model/types.hpp"

class Sort {
public:
	// Constructor
	Sort(size_t maxUnseenFrames = 5);

	// Destructor
	~Sort();

	// Update the tracking with detected objects
	std::vector<Object> update(const std::vector<Object> &detections);

	// Get the current tracked objects and their classes
	std::vector<Object> getTrackedObjects() const;

	// Set Max Unseen Frames
	void setMaxUnseenFrames(size_t maxUnseenFrames_)
	{
		this->maxUnseenFrames = maxUnseenFrames_;
	}

	// Get Max Unseen Frames
	size_t getMaxUnseenFrames() const { return this->maxUnseenFrames; }

private:
	// Private methods for the Kalman filter and other internal workings
	void initializeKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox);
	cv::Rect_<float> predict(cv::KalmanFilter &kf);
	cv::Rect_<float> updateKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox);

	// Data members for tracking
	std::vector<Object> trackedObjects;
	uint64_t nextTrackID;
	size_t maxUnseenFrames;
	static const uint64_t MAX_TRACK_ID = 9;
	std::set<uint64_t> usedIDs;
	uint64_t allocateID();
	void releaseID(uint64_t id);
};

#endif
// Path: src/sort-cpp/Sort.cpp
