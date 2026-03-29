#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include "ort-model/ONNXRuntimeModel.h"
#include "sort/Sort.h"

/**
  * @brief The filter_data struct
  *
  * This struct is used to store the base data needed for ORT filters.
  *
*/
struct filter_data {
	std::string useGPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;

	int minAreaThreshold;
	int objectCategory;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
	int maskingDilateIterations;
	float maskingOverlayAlpha;
	int maskingPatternSize;
	int maskingColor2;
	float maskScale;
	std::string maskShape;
	bool maskInvert;
	// Glitch effect
	float glitchIntensity;
	float totalSeconds;
	// Image stamp
	std::string stampImagePath;
	cv::Mat stampBGRA;
	bool stampKeepAspect;
	// OBS source overlay
	std::string overlaySourceName;
	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	std::string saveDetectionsPath;
	float lastDetectionWriteTime;
	float detectionWriteAccum;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	// create SORT tracker
	Sort tracker;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;
	gs_effect_t *pixelateEffect;
	gs_effect_t *frostedGlassEffect;
	gs_effect_t *patternEffect;
	gs_effect_t *glitchEffect;
	gs_effect_t *colorGradeEffect;
	gs_effect_t *overlayEffect;
	gs_texrender_t *overlayTexrender;

	cv::Mat inputBGRA;
	cv::Mat outputMask;
	cv::Mat outputStampBGRA;
	bool hasStamp;

	bool isDisabled;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;
	std::mutex stampLock;

	std::unique_ptr<ONNXRuntimeModel> onnxruntimemodel;
	std::vector<std::string> classNames;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
