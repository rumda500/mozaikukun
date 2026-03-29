#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif // _WIN32

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "edgeyolo/edgeyolo_onnxruntime.hpp"
#include "yunet/YuNet.h"

#define EXTERNAL_MODEL_SIZE "!!!EXTERNAL_MODEL!!!"
#define FACE_DETECT_MODEL_SIZE "!!!FACE_DETECT!!!"

struct detect_filter : public filter_data {};

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop,
			    const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p,
				     obs_data_t *settings)
{
	const bool enabled = obs_data_get_bool(settings, "advanced");

	for (const char *prop_name :
	     {"threshold", "useGPU", "numThreads", "model_size", "detected_object",
	      "save_detections_path", "crop_group",
	      "min_size_threshold"}) {
		p = obs_properties_get(ppts, prop_name);
		obs_property_set_visible(p, enabled);
	}

	return true;
}

void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		// capitalize the first letter of the class name
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}

	// sort the vector based on the class names
	std::sort(indexed_classes.begin(), indexed_classes.end(),
		  [](const std::pair<size_t, std::string> &a,
		     const std::pair<size_t, std::string> &b) { return a.second < b.second; });

	// clear the object category list
	obs_property_list_clear(object_category);

	// add the sorted classes to the property list
	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	// add the sorted classes to the property list
	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings, struct detect_filter *tf_)
{
	if (model_file == nullptr || model_file[0] == '\0' || strlen(model_file) == 0) {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	// read the '.json' file near the model file to find the class names
	std::string json_file = model_file;
	json_file.replace(json_file.find(".onnx"), 5, ".json");
	std::ifstream file(json_file);
	if (!file.is_open()) {
		obs_data_set_string(settings, "error", "JSON file not found");
		obs_log(LOG_ERROR, "JSON file not found: %s", json_file.c_str());
	} else {
		obs_data_set_string(settings, "error", "");
		// parse the JSON file
		nlohmann::json j;
		file >> j;
		if (j.contains("names")) {
			std::vector<std::string> labels = j["names"];
			set_class_names_on_object_category(
				obs_properties_get(props_, "object_category"), labels);
			tf_->classNames = labels;
		} else {
			obs_data_set_string(settings, "error",
					    "JSON file does not contain 'names' field");
			obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
		}
	}
}

static void update_masking_type_visibility(obs_properties_t *props, obs_data_t *settings,
					   bool enabled)
{
	std::string t = obs_data_get_string(settings, "masking_type");
	obs_property_set_visible(obs_properties_get(props, "masking_color"), false);
	obs_property_set_visible(obs_properties_get(props, "masking_blur_radius"), false);
	obs_property_set_visible(obs_properties_get(props, "masking_overlay_alpha"), false);
	obs_property_set_visible(obs_properties_get(props, "masking_color2"), false);
	obs_property_set_visible(obs_properties_get(props, "masking_pattern_size"), false);
	obs_property_set_visible(obs_properties_get(props, "glitch_intensity"), false);
	obs_property_set_visible(obs_properties_get(props, "stamp_image_path"), false);
	obs_property_set_visible(obs_properties_get(props, "stamp_keep_aspect"), false);
	obs_property_set_visible(obs_properties_get(props, "overlay_source"), false);
	obs_property_set_visible(obs_properties_get(props, "dilation_iterations"), enabled);
	obs_property_set_visible(obs_properties_get(props, "mask_shape"), enabled);
	obs_property_set_visible(obs_properties_get(props, "invert_mask"), enabled);

	if (!enabled)
		return;

	if (t == "solid_color" || t == "eye_bar") {
		obs_property_set_visible(obs_properties_get(props, "masking_color"), true);
	} else if (t == "blur" || t == "pixelate") {
		obs_property_set_visible(obs_properties_get(props, "masking_blur_radius"), true);
	} else if (t == "frosted_glass") {
		obs_property_set_visible(obs_properties_get(props, "masking_blur_radius"), true);
		obs_property_set_visible(obs_properties_get(props, "masking_overlay_alpha"), true);
		obs_property_set_visible(obs_properties_get(props, "masking_color"), true);
	} else if (t == "checker_pattern" || t == "stripe_pattern" || t == "hstripe_pattern" ||
		   t == "dot_pattern") {
		obs_property_set_visible(obs_properties_get(props, "masking_color"), true);
		obs_property_set_visible(obs_properties_get(props, "masking_color2"), true);
		obs_property_set_visible(obs_properties_get(props, "masking_pattern_size"), true);
	} else if (t == "glitch") {
		obs_property_set_visible(obs_properties_get(props, "glitch_intensity"), true);
	} else if (t == "stamp") {
		obs_property_set_visible(obs_properties_get(props, "stamp_image_path"), true);
		obs_property_set_visible(obs_properties_get(props, "stamp_keep_aspect"), true);
	} else if (t == "obs_source") {
		obs_property_set_visible(obs_properties_get(props, "overlay_source"), true);
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	// add dropdown selection for object category selection
	// initialize based on current model to avoid showing COCO classes for face model
	obs_property_t *object_category =
		obs_properties_add_list(props, "object_category", obs_module_text("ObjectCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
		set_class_names_on_object_category(object_category, yunet::FACE_CLASSES);
		tf->classNames = yunet::FACE_CLASSES;
	} else {
		set_class_names_on_object_category(object_category, edgeyolo_cpp::COCO_CLASSES);
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;
	}
	obs_property_set_long_description(object_category, obs_module_text("ObjectCategoryDesc"));


	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);
	obs_property_set_long_description(masking_group_prop, obs_module_text("MaskingGroupDesc"));

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, [](obs_properties_t *props_,
								  obs_property_t *,
								  obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_set_visible(obs_properties_get(props_, "masking_type"), enabled);
		update_masking_type_visibility(props_, settings, enabled);
		return true;
	});

	// add masking options drop down selection: "None", "Solid color", "Blur", "Transparent"
	obs_property_t *masking_type = obs_properties_add_list(masking_group, "masking_type",
							       obs_module_text("MaskingType"),
							       OBS_COMBO_TYPE_LIST,
							       OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(masking_type, obs_module_text("MaskingTypeDesc"));
	obs_property_list_add_string(masking_type, obs_module_text("None"), "none");
	obs_property_list_add_string(masking_type, obs_module_text("Blur"), "blur");
	obs_property_list_add_string(masking_type, obs_module_text("Pixelate"), "pixelate");
	obs_property_list_add_string(masking_type, obs_module_text("FrostedGlass"), "frosted_glass");
	obs_property_list_add_string(masking_type, obs_module_text("Glitch"), "glitch");
	obs_property_list_add_string(masking_type, obs_module_text("Grayscale"), "grayscale");
	obs_property_list_add_string(masking_type, obs_module_text("Thermal"), "thermal");
	obs_property_list_add_string(masking_type, obs_module_text("Sepia"), "sepia");
	obs_property_list_add_string(masking_type, obs_module_text("Negative"), "negative");
	obs_property_list_add_string(masking_type, obs_module_text("CheckerPattern"), "checker_pattern");
	obs_property_list_add_string(masking_type, obs_module_text("StripePattern"), "stripe_pattern");
	obs_property_list_add_string(masking_type, obs_module_text("HStripePattern"), "hstripe_pattern");
	obs_property_list_add_string(masking_type, obs_module_text("DotPattern"), "dot_pattern");
	obs_property_list_add_string(masking_type, obs_module_text("SolidColor"), "solid_color");
	obs_property_list_add_string(masking_type, obs_module_text("EyeBar"), "eye_bar");
	obs_property_list_add_string(masking_type, obs_module_text("Stamp"), "stamp");
	obs_property_list_add_string(masking_type, obs_module_text("OBSSource"), "obs_source");
	obs_property_list_add_string(masking_type, obs_module_text("OutputMask"), "output_mask");
	obs_property_list_add_string(masking_type, obs_module_text("Transparent"), "transparent");

	// add color picker for solid color masking
	obs_properties_add_color(masking_group, "masking_color", obs_module_text("MaskingColor"));

	// add slider for blur radius
	{
		obs_property_t *p = obs_properties_add_int_slider(
			masking_group, "masking_blur_radius",
			obs_module_text("MaskingBlurRadius"), 1, 64, 1);
		obs_property_set_long_description(p, obs_module_text("MaskingBlurRadiusDesc"));
	}

	// add slider for frosted glass overlay alpha
	obs_properties_add_float_slider(masking_group, "masking_overlay_alpha",
					obs_module_text("OverlayAlpha"), 0.0, 1.0, 0.05);

	// add second color picker for pattern masking
	obs_properties_add_color(masking_group, "masking_color2",
				 obs_module_text("MaskingColor2"));

	// add slider for pattern size
	obs_properties_add_int_slider(masking_group, "masking_pattern_size",
				      obs_module_text("PatternSize"), 2, 64, 1);

	// add slider for glitch intensity
	obs_properties_add_float_slider(masking_group, "glitch_intensity",
					obs_module_text("GlitchIntensity"), 0.1, 1.0, 0.05);

	// add file picker for image stamp
	obs_properties_add_path(masking_group, "stamp_image_path",
				obs_module_text("StampImagePath"), OBS_PATH_FILE,
				"PNG Images (*.png);;All Files (*.*)", nullptr);
	obs_properties_add_bool(masking_group, "stamp_keep_aspect",
				obs_module_text("StampKeepAspect"));

	// add editable list for overlay OBS source
	{
		obs_property_t *src_list =
			obs_properties_add_list(masking_group, "overlay_source",
						obs_module_text("OBSSource"),
						OBS_COMBO_TYPE_EDITABLE,
						OBS_COMBO_FORMAT_STRING);
		obs_property_list_add_string(src_list, "", "");
		obs_enum_sources(
			[](void *param, obs_source_t *src) -> bool {
				obs_property_t *list = (obs_property_t *)param;
				const char *name = obs_source_get_name(src);
				obs_property_list_add_string(list, name, name);
				return true;
			},
			src_list);
	}

	// add callback to show/hide blur radius and color picker
	obs_property_set_modified_callback(masking_type, [](obs_properties_t *props_,
							    obs_property_t *,
							    obs_data_t *settings) {
		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
		update_masking_type_visibility(props_, settings, masking_enabled);
		return true;
	});

	// add slider for mask scale (enlarge detection box)
	{
		obs_property_t *p = obs_properties_add_float_slider(
			masking_group, "mask_scale", obs_module_text("MaskScale"), 1.0, 2.0, 0.05);
		obs_property_set_long_description(p, obs_module_text("MaskScaleDesc"));
	}

	// add mask shape dropdown (rect / ellipse)
	obs_property_t *mask_shape = obs_properties_add_list(masking_group, "mask_shape",
							     obs_module_text("MaskShape"),
							     OBS_COMBO_TYPE_LIST,
							     OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(mask_shape, obs_module_text("MaskShapeRect"), "rect");
	obs_property_list_add_string(mask_shape, obs_module_text("MaskShapeEllipse"), "ellipse");
	obs_property_set_long_description(mask_shape, obs_module_text("MaskShapeDesc"));

	// add invert mask checkbox
	{
		obs_property_t *p = obs_properties_add_bool(masking_group, "invert_mask",
							    obs_module_text("InvertMask"));
		obs_property_set_long_description(p, obs_module_text("InvertMaskDesc"));
	}

	// add slider for dilation iterations
	{
		obs_property_t *p = obs_properties_add_int_slider(
			masking_group, "dilation_iterations",
			obs_module_text("DilationIterations"), 0, 20, 1);
		obs_property_set_long_description(p, obs_module_text("DilationIterationsDesc"));
	}

	// add options group for tracking and zoom-follow options
	obs_properties_t *tracking_group_props = obs_properties_create();
	obs_property_t *tracking_group = obs_properties_add_group(
		props, "tracking_group", obs_module_text("TrackingZoomFollowGroup"),
		OBS_GROUP_CHECKABLE, tracking_group_props);

	// add callback to show/hide tracking options
	obs_property_set_modified_callback(tracking_group, [](obs_properties_t *props_,
							      obs_property_t *,
							      obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "tracking_group");
		for (auto prop_name : {"zoom_factor", "zoom_object", "zoom_speed_factor"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add zoom factor slider
	obs_properties_add_float_slider(tracking_group_props, "zoom_factor",
					obs_module_text("ZoomFactor"), 0.0, 1.0, 0.05);

	obs_properties_add_float_slider(tracking_group_props, "zoom_speed_factor",
					obs_module_text("ZoomSpeed"), 0.0, 0.1, 0.01);

	// add object selection for zoom drop down: "Single", "All"
	obs_property_t *zoom_object = obs_properties_add_list(tracking_group_props, "zoom_object",
							      obs_module_text("ZoomObject"),
							      OBS_COMBO_TYPE_LIST,
							      OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(zoom_object, obs_module_text("SingleFirst"), "single");
	obs_property_list_add_string(zoom_object, obs_module_text("Biggest"), "biggest");
	obs_property_list_add_string(zoom_object, obs_module_text("Oldest"), "oldest");
	obs_property_list_add_string(zoom_object, obs_module_text("All"), "all");

	obs_property_t *advanced =
		obs_properties_add_bool(props, "advanced", obs_module_text("Advanced"));

	// If advanced is selected show the advanced settings, otherwise hide them
	obs_property_set_modified_callback(advanced, enable_advanced_settings);

	// add a checkable group for crop region settings
	obs_properties_t *crop_group_props = obs_properties_create();
	obs_property_t *crop_group =
		obs_properties_add_group(props, "crop_group", obs_module_text("CropGroup"),
					 OBS_GROUP_CHECKABLE, crop_group_props);

	// add callback to show/hide crop region options
	obs_property_set_modified_callback(crop_group, [](obs_properties_t *props_,
							  obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "crop_group");
		for (auto prop_name : {"crop_left", "crop_right", "crop_top", "crop_bottom"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add crop region settings
	obs_properties_add_int_slider(crop_group_props, "crop_left", obs_module_text("CropLeft"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_right", obs_module_text("CropRight"),
				      0, 1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_top", obs_module_text("CropTop"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_bottom",
				      obs_module_text("CropBottom"), 0, 1000, 1);

	// add a text input for the currently detected object
	obs_property_t *detected_obj_prop = obs_properties_add_text(
		props, "detected_object", obs_module_text("DetectedObject"), OBS_TEXT_DEFAULT);
	// disable the text input by default
	obs_property_set_enabled(detected_obj_prop, false);

	// add threshold slider
	{
		obs_property_t *p = obs_properties_add_float_slider(
			props, "threshold", obs_module_text("ConfThreshold"), 0.0, 1.0, 0.025);
		obs_property_set_long_description(p, obs_module_text("ConfThresholdDesc"));
	}

	// add minimal size threshold slider
	{
		obs_property_t *p = obs_properties_add_int_slider(
			props, "min_size_threshold", obs_module_text("MinSizeThreshold"), 0, 10000,
			1);
		obs_property_set_long_description(p, obs_module_text("MinSizeThresholdDesc"));
	}

	// add file path for saving detections
	obs_properties_add_path(props, "save_detections_path",
				obs_module_text("SaveDetectionsPath"), OBS_PATH_FILE_SAVE,
				"JSON file (*.json);;All files (*.*)", nullptr);

	/* GPU, CPU and performance Props */
	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(p_use_gpu, obs_module_text("InferenceDeviceDesc"));

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUTensorRT"), USEGPU_TENSORRT);
#endif
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUDirectML"), USEGPU_DML);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 8, 1);

	// add drop down option for model size: Small, Medium, Large
	obs_property_t *model_size =
		obs_properties_add_list(props, "model_size", obs_module_text("ModelSize"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(model_size, obs_module_text("ModelSizeDesc"));
	obs_property_list_add_string(model_size, obs_module_text("SmallFast"), "small");
	obs_property_list_add_string(model_size, obs_module_text("Medium"), "medium");
	obs_property_list_add_string(model_size, obs_module_text("LargeSlow"), "large");
	obs_property_list_add_string(model_size, obs_module_text("FaceDetect"),
				     FACE_DETECT_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("ExternalModel"),
				     EXTERNAL_MODEL_SIZE);

	// add external model file path
	obs_properties_add_path(props, "external_model_file", obs_module_text("ModelPath"),
				OBS_PATH_FILE, "EdgeYOLO onnx files (*.onnx);;all files (*.*)",
				nullptr);

	// add callback to show/hide the external model file path
	obs_property_set_modified_callback2(
		model_size,
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			std::string model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = model_size_value == EXTERNAL_MODEL_SIZE;
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);
			if (!is_external) {
				if (model_size_value == FACE_DETECT_MODEL_SIZE) {
					// set the class names to COCO classes for face detection model
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						yunet::FACE_CLASSES);
					tf_->classNames = yunet::FACE_CLASSES;
				} else {
					// reset the class names to COCO classes for default models
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						edgeyolo_cpp::COCO_CLASSES);
					tf_->classNames = edgeyolo_cpp::COCO_CLASSES;
				}
			} else {
				// if the model path is already set - update the class names
				const char *model_file =
					obs_data_get_string(settings, "external_model_file");
				read_model_config_json_and_set_class_names(model_file, props_,
									   settings, tf_);
			}
			return true;
		},
		tf);

	// add callback on the model file path to check if the file exists
	obs_property_set_modified_callback2(
		obs_properties_get(props, "external_model_file"),
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			const char *model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = strcmp(model_size_value, EXTERNAL_MODEL_SIZE) == 0;
			if (!is_external) {
				return true;
			}
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			const char *model_file =
				obs_data_get_string(settings, "external_model_file");
			read_model_config_json_and_set_class_names(model_file, props_, settings,
								   tf_);
			return true;
		},
		tf);

	// Add a informative text about the plugin
	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "model_size", FACE_DETECT_MODEL_SIZE);
	obs_data_set_default_int(settings, "object_category", 0); // 0 = face for face detection model
	obs_data_set_default_bool(settings, "masking_group", true);
	obs_data_set_default_string(settings, "masking_type", "blur");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_blur_radius", 10);
	obs_data_set_default_double(settings, "mask_scale", 1.3);
	obs_data_set_default_int(settings, "dilation_iterations", 0);
	obs_data_set_default_double(settings, "masking_overlay_alpha", 0.3);
	obs_data_set_default_string(settings, "masking_color2", "#FFFFFF");
	obs_data_set_default_int(settings, "masking_pattern_size", 8);
	obs_data_set_default_double(settings, "glitch_intensity", 0.5);
	obs_data_set_default_string(settings, "stamp_image_path", "");
	obs_data_set_default_bool(settings, "stamp_keep_aspect", false);
	obs_data_set_default_string(settings, "overlay_source", "");
	obs_data_set_default_string(settings, "mask_shape", "rect");
	obs_data_set_default_bool(settings, "invert_mask", false);
	obs_data_set_default_bool(settings, "tracking_group", false);
	obs_data_set_default_double(settings, "zoom_factor", 0.0);
	obs_data_set_default_double(settings, "zoom_speed_factor", 0.05);
	obs_data_set_default_string(settings, "zoom_object", "single");
	obs_data_set_default_string(settings, "save_detections_path", "");
	obs_data_set_default_bool(settings, "crop_group", false);
	obs_data_set_default_int(settings, "crop_left", 0);
	obs_data_set_default_int(settings, "crop_right", 0);
	obs_data_set_default_int(settings, "crop_top", 0);
	obs_data_set_default_int(settings, "crop_bottom", 0);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	// 旧 obs-detect 設定からの自動マイグレーション（初回のみ）
	// masking_group=false, masking_type="none" は旧デフォルト値 → 新デフォルトへ移行
	if (obs_data_get_int(settings, "mozaikukun_version") == 0) {
		bool masking_off = !obs_data_get_bool(settings, "masking_group");
		const char *mtype = obs_data_get_string(settings, "masking_type");
		bool is_old_default =
			masking_off && (mtype != nullptr && strcmp(mtype, "none") == 0);
		if (is_old_default) {
			obs_log(LOG_INFO,
				"Migrating legacy obs-detect settings to モザイク君 defaults");
			obs_data_set_bool(settings, "masking_group", true);
			obs_data_set_string(settings, "masking_type", "blur");
			obs_data_set_string(settings, "model_size", FACE_DETECT_MODEL_SIZE);
			obs_data_set_int(settings, "object_category", 0);
		}
		obs_data_set_int(settings, "mozaikukun_version", 1);
	}

	tf->isDisabled = true;

	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	tf->maskScale = (float)obs_data_get_double(settings, "mask_scale");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "dilation_iterations");
	tf->maskingOverlayAlpha = (float)obs_data_get_double(settings, "masking_overlay_alpha");
	tf->maskingColor2 = (int)obs_data_get_int(settings, "masking_color2");
	tf->maskingPatternSize = (int)obs_data_get_int(settings, "masking_pattern_size");
	tf->glitchIntensity = (float)obs_data_get_double(settings, "glitch_intensity");
	tf->maskShape = obs_data_get_string(settings, "mask_shape");
	tf->maskInvert = obs_data_get_bool(settings, "invert_mask");
	tf->overlaySourceName = obs_data_get_string(settings, "overlay_source");
	tf->stampKeepAspect = obs_data_get_bool(settings, "stamp_keep_aspect");
	// Reload stamp image if path changed
	{
		std::string newPath = obs_data_get_string(settings, "stamp_image_path");
		if (newPath != tf->stampImagePath) {
			tf->stampImagePath = newPath;
			if (!newPath.empty()) {
				cv::Mat loaded;
#ifdef _WIN32
				{
					int wlen = MultiByteToWideChar(CP_UTF8, 0,
						newPath.c_str(), -1, nullptr, 0);
					std::wstring wpath(wlen, 0);
					MultiByteToWideChar(CP_UTF8, 0,
						newPath.c_str(), -1, wpath.data(), wlen);
					IWICImagingFactory *wicFactory = nullptr;
					HRESULT hr = CoCreateInstance(
						CLSID_WICImagingFactory, nullptr,
						CLSCTX_INPROC_SERVER,
						IID_PPV_ARGS(&wicFactory));
					if (SUCCEEDED(hr) && wicFactory) {
						IWICBitmapDecoder *decoder = nullptr;
						hr = wicFactory->CreateDecoderFromFilename(
							wpath.c_str(), nullptr,
							GENERIC_READ,
							WICDecodeMetadataCacheOnDemand,
							&decoder);
						if (SUCCEEDED(hr) && decoder) {
							IWICBitmapFrameDecode *frm = nullptr;
							hr = decoder->GetFrame(0, &frm);
							if (SUCCEEDED(hr) && frm) {
								IWICFormatConverter *conv = nullptr;
								hr = wicFactory->CreateFormatConverter(&conv);
								if (SUCCEEDED(hr) && conv) {
									hr = conv->Initialize(
										frm,
										GUID_WICPixelFormat32bppBGRA,
										WICBitmapDitherTypeNone,
										nullptr, 0.0,
										WICBitmapPaletteTypeCustom);
									if (SUCCEEDED(hr)) {
										UINT w = 0, h = 0;
										conv->GetSize(&w, &h);
										loaded = cv::Mat(h, w, CV_8UC4);
										hr = conv->CopyPixels(
											nullptr, w * 4,
											w * h * 4,
											loaded.data);
										if (FAILED(hr))
											loaded = cv::Mat();
									}
									conv->Release();
								}
								frm->Release();
							}
							decoder->Release();
						}
						wicFactory->Release();
					}
				}
#else
				loaded = cv::imread(newPath, cv::IMREAD_UNCHANGED);
#endif
				if (loaded.empty()) {
					obs_log(LOG_ERROR,
						"[mozaikukun] Failed to load stamp image: %s",
						newPath.c_str());
				}
				cv::Mat converted;
				if (!loaded.empty()) {
					int ch = loaded.channels();
					if (ch == 1)
						cv::cvtColor(loaded, converted, cv::COLOR_GRAY2BGRA);
					else if (ch == 3)
						cv::cvtColor(loaded, converted, cv::COLOR_BGR2BGRA);
					else if (ch == 4)
						converted = loaded;
				}
				std::lock_guard<std::mutex> slock(tf->stampLock);
				tf->stampBGRA = converted;
			} else {
				std::lock_guard<std::mutex> slock(tf->stampLock);
				tf->stampBGRA = cv::Mat();
			}
		}
	}
		bool newTrackingEnabled = obs_data_get_bool(settings, "tracking_group");
	tf->zoomFactor = (float)obs_data_get_double(settings, "zoom_factor");
	tf->zoomSpeedFactor = (float)obs_data_get_double(settings, "zoom_speed_factor");
	tf->zoomObject = obs_data_get_string(settings, "zoom_object");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");
	tf->minAreaThreshold = (int)obs_data_get_int(settings, "min_size_threshold");

	// check if tracking state has changed
	if (tf->trackingEnabled != newTrackingEnabled) {
		tf->trackingEnabled = newTrackingEnabled;
		obs_source_t *parent = obs_filter_get_parent(tf->source);
		if (!parent) {
			obs_log(LOG_WARNING,
				"Parent source not found, deferring tracking setup");
		} else if (tf->trackingEnabled) {
			obs_log(LOG_DEBUG, "Tracking enabled");
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (!crop_pad_filter) {
				crop_pad_filter = obs_source_create(
					"crop_filter", "Detect Tracking", nullptr, nullptr);
				obs_source_filter_add(parent, crop_pad_filter);
			}
			tf->trackingFilter = crop_pad_filter;
		} else {
			obs_log(LOG_DEBUG, "Tracking disabled");
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (crop_pad_filter) {
				obs_source_filter_remove(parent, crop_pad_filter);
			}
			tf->trackingFilter = nullptr;
		}
	}

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		obs_log(LOG_INFO, "Reinitializing model");
		reinitialize = true;

		// lock modelMutex
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;
		if (newModelSize == "small") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_256x416.onnx");
		} else if (newModelSize == "medium") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_480x800.onnx");
		} else if (newModelSize == "large") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_736x1280.onnx");
		} else if (newModelSize == FACE_DETECT_MODEL_SIZE) {
			modelFilepath_rawPtr =
				obs_module_file("models/face_detection_yunet_2023mar.onnx");
		} else if (newModelSize == EXTERNAL_MODEL_SIZE) {
			const char *external_model_file =
				obs_data_get_string(settings, "external_model_file");
			if (external_model_file == nullptr || external_model_file[0] == '\0' ||
			    strlen(external_model_file) == 0) {
				obs_log(LOG_ERROR, "External model file path is empty");
				tf->isDisabled = true;
				return;
			}
			modelFilepath_rawPtr = bstrdup(external_model_file);
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
						    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		// Re-initialize model if it's not already the selected one or switching inference device
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		// parameters
		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;
		float nms_th_ = 0.45f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;

		// If this is an external model - look for the config JSON file
		if (tf->modelSize == EXTERNAL_MODEL_SIZE) {
#ifdef _WIN32
			std::wstring labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(L".onnx"), 5, L".json");
#else
			std::string labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(".onnx"), 5, ".json");
#endif
			std::ifstream labelsFile(labelsFilepath);
			if (labelsFile.is_open()) {
				// Parse the JSON file
				nlohmann::json j;
				labelsFile >> j;
				if (j.contains("names")) {
					std::vector<std::string> labels = j["names"];
					num_classes_ = (int)labels.size();
					tf->classNames = labels;
				} else {
					obs_log(LOG_ERROR,
						"JSON file does not contain 'labels' field");
					tf->isDisabled = true;
					tf->onnxruntimemodel.reset();
					return;
				}
			} else {
				obs_log(LOG_ERROR, "Failed to open JSON file: %s",
					labelsFilepath.c_str());
				tf->isDisabled = true;
				tf->onnxruntimemodel.reset();
				return;
			}
		} else if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
			num_classes_ = 1;
			tf->classNames = yunet::FACE_CLASSES;
		}

		// Load model
		try {
			if (tf->onnxruntimemodel) {
				tf->onnxruntimemodel.reset();
			}
			if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
				tf->onnxruntimemodel = std::make_unique<yunet::YuNetONNX>(
					tf->modelFilepath, tf->numThreads, 50, tf->numThreads,
					tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
			} else {
				tf->onnxruntimemodel =
					std::make_unique<edgeyolo_cpp::EdgeYOLOONNXRuntime>(
						tf->modelFilepath, tf->numThreads, num_classes_,
						tf->numThreads, tf->useGPU, onnxruntime_device_id_,
						onnxruntime_use_parallel_, nms_th_,
						tf->conf_threshold);
			}
			// clear error message
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());
			// disable filter
			tf->isDisabled = true;
			tf->onnxruntimemodel.reset();
			return;
		}
	}

	// update threshold on edgeyolo
	if (tf->onnxruntimemodel) {
		tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		// Log the currently selected options
		obs_log(LOG_INFO, "Detect Filter Options:");
		// name of the source that the filter is attached to
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
		obs_log(LOG_INFO, "  Object Category: %s",
			obs_data_get_string(settings, "object_category"));
		obs_log(LOG_INFO, "  Masking Enabled: %s",
			obs_data_get_bool(settings, "masking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Masking Type: %s",
			obs_data_get_string(settings, "masking_type"));
		obs_log(LOG_INFO, "  Masking Color: %s",
			obs_data_get_string(settings, "masking_color"));
		obs_log(LOG_INFO, "  Masking Blur Radius: %d",
			obs_data_get_int(settings, "masking_blur_radius"));
		obs_log(LOG_INFO, "  Tracking Enabled: %s",
			obs_data_get_bool(settings, "tracking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Zoom Factor: %.2f",
			obs_data_get_double(settings, "zoom_factor"));
		obs_log(LOG_INFO, "  Zoom Object: %s",
			obs_data_get_string(settings, "zoom_object"));
		obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	// enable
	tf->isDisabled = false;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER HELPERS                   */

// Scale a detection rect by maskScale, clamped to frame bounds
static cv::Rect scaleAndClampRect(const cv::Rect2f &rect, float maskScale, int frameCols,
				   int frameRows)
{
	cv::Rect2f scaled = rect;
	if (maskScale > 1.0f) {
		float dw = scaled.width * (maskScale - 1.0f) * 0.5f;
		float dh = scaled.height * (maskScale - 1.0f) * 0.5f;
		scaled.x -= dw;
		scaled.y -= dh;
		scaled.width += dw * 2.0f;
		scaled.height += dh * 2.0f;
		scaled.x = std::max(scaled.x, 0.0f);
		scaled.y = std::max(scaled.y, 0.0f);
		scaled.width = std::min(scaled.width, (float)frameCols - scaled.x);
		scaled.height = std::min(scaled.height, (float)frameRows - scaled.y);
	}
	return cv::Rect(scaled);
}

// Generate a binary mask from detected objects
static cv::Mat generateMask(struct detect_filter *tf, const cv::Mat &frame,
			     const std::vector<Object> &objects)
{
	cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
	for (const Object &obj : objects) {
		cv::Rect clamped = scaleAndClampRect(obj.rect, tf->maskScale, frame.cols, frame.rows);

		if (tf->maskingType == "eye_bar") {
			cv::Rect eyeRect(clamped.x,
					 clamped.y + (int)(clamped.height * 0.20f),
					 clamped.width, (int)(clamped.height * 0.35f));
			eyeRect &= cv::Rect(0, 0, frame.cols, frame.rows);
			if (eyeRect.width > 0 && eyeRect.height > 0)
				cv::rectangle(mask, eyeRect, cv::Scalar(255), -1);
		} else if (tf->maskShape == "ellipse") {
			cv::Point center(clamped.x + clamped.width / 2,
					 clamped.y + clamped.height / 2);
			cv::Size axes(std::max(clamped.width / 2, 1),
				      std::max(clamped.height / 2, 1));
			cv::ellipse(mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
		} else {
			cv::rectangle(mask, clamped, cv::Scalar(255), -1);
		}
	}

	if (tf->maskInvert) {
		cv::bitwise_not(mask, mask);
	}
	if (tf->maskingDilateIterations > 0) {
		cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), tf->maskingDilateIterations);
	}
	return mask;
}

// Composite stamp images onto frame (CPU alpha blending)
static cv::Mat compositeStamps(struct detect_filter *tf, const cv::Mat &frame,
			       const std::vector<Object> &objects, const cv::Mat &stampSnapshot)
{
	cv::Mat frameBGRA;
	cv::cvtColor(frame, frameBGRA, cv::COLOR_BGR2BGRA);
	for (const Object &obj : objects) {
		cv::Rect r = scaleAndClampRect(obj.rect, tf->maskScale, frame.cols, frame.rows);
		r &= cv::Rect(0, 0, frameBGRA.cols, frameBGRA.rows);
		if (r.width <= 0 || r.height <= 0)
			continue;
		cv::Mat stampResized;
		int ox = 0, oy = 0;
		if (tf->stampKeepAspect) {
			float scaleX = (float)r.width / stampSnapshot.cols;
			float scaleY = (float)r.height / stampSnapshot.rows;
			float scale = std::min(scaleX, scaleY);
			int sw = std::max((int)(stampSnapshot.cols * scale), 1);
			int sh = std::max((int)(stampSnapshot.rows * scale), 1);
			cv::resize(stampSnapshot, stampResized, cv::Size(sw, sh));
			ox = (r.width - sw) / 2;
			oy = (r.height - sh) / 2;
		} else {
			cv::resize(stampSnapshot, stampResized, cv::Size(r.width, r.height));
		}
		cv::Mat roi = frameBGRA(r);
		int yStart = std::max(0, -oy);
		int yEnd = std::min(stampResized.rows, r.height - oy);
		int xStart = std::max(0, -ox);
		int xEnd = std::min(stampResized.cols, r.width - ox);
		for (int y = yStart; y < yEnd; y++) {
			const cv::Vec4b *srcRow = stampResized.ptr<cv::Vec4b>(y);
			cv::Vec4b *dstRow = roi.ptr<cv::Vec4b>(y + oy);
			for (int x = xStart; x < xEnd; x++) {
				const cv::Vec4b &src = srcRow[x];
				uint8_t a = src[3];
				if (a == 0)
					continue;
				cv::Vec4b &dst = dstRow[x + ox];
				if (a == 255) {
					dst = src;
					continue;
				}
				uint16_t inv_a = 255 - a;
				dst[0] = (uint8_t)((src[0] * a + dst[0] * inv_a + 127) / 255);
				dst[1] = (uint8_t)((src[1] * a + dst[1] * inv_a + 127) / 255);
				dst[2] = (uint8_t)((src[2] * a + dst[2] * inv_a + 127) / 255);
			}
		}
	}
	return frameBGRA;
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;

	std::vector<std::tuple<const char *, gs_effect_t **>> effects = {
		{KAWASE_BLUR_EFFECT_PATH, &tf->kawaseBlurEffect},
		{MASKING_EFFECT_PATH, &tf->maskingEffect},
		{PIXELATE_EFFECT_PATH, &tf->pixelateEffect},
		{FROSTED_GLASS_EFFECT_PATH, &tf->frostedGlassEffect},
		{PATTERN_EFFECT_PATH, &tf->patternEffect},
		{GLITCH_EFFECT_PATH, &tf->glitchEffect},
		{COLORGRADE_EFFECT_PATH, &tf->colorGradeEffect},
		{OVERLAY_EFFECT_PATH, &tf->overlayEffect},
	};

	for (auto [effectPath, effect] : effects) {
		char *effectPathPtr = obs_module_file(effectPath);
		if (!effectPathPtr) {
			obs_log(LOG_ERROR, "Failed to get effect path: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_enter_graphics();
		*effect = gs_effect_create_from_file(effectPathPtr, nullptr);
		bfree(effectPathPtr);
		if (!*effect) {
			obs_log(LOG_ERROR, "Failed to load effect: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_leave_graphics();
	}

	detect_filter_update(tf, settings);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		gs_effect_destroy(tf->kawaseBlurEffect);
		gs_effect_destroy(tf->maskingEffect);
		gs_effect_destroy(tf->pixelateEffect);
		gs_effect_destroy(tf->frostedGlassEffect);
		gs_effect_destroy(tf->patternEffect);
		gs_effect_destroy(tf->glitchEffect);
		gs_effect_destroy(tf->colorGradeEffect);
		gs_effect_destroy(tf->overlayEffect);
		if (tf->overlayTexrender) {
			gs_texrender_destroy(tf->overlayTexrender);
		}
		obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->totalSeconds += seconds;

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		if (tf->inputBGRA.empty()) {
			return;
		}
		imageBGRA = std::move(tf->inputBGRA);
	}

	cv::Mat inferenceFrame;

	cv::Rect cropRect(0, 0, imageBGRA.cols, imageBGRA.rows);
	if (tf->crop_enabled) {
		int cw = std::max(1, imageBGRA.cols - tf->crop_left - tf->crop_right);
		int ch = std::max(1, imageBGRA.rows - tf->crop_top - tf->crop_bottom);
		int cx = std::clamp(tf->crop_left, 0, imageBGRA.cols - 1);
		int cy = std::clamp(tf->crop_top, 0, imageBGRA.rows - 1);
		cw = std::min(cw, imageBGRA.cols - cx);
		ch = std::min(ch, imageBGRA.rows - cy);
		cropRect = cv::Rect(cx, cy, cw, ch);
		cv::cvtColor(imageBGRA(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
	} else {
		cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
	}

	std::vector<Object> objects;

	try {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		objects = tf->onnxruntimemodel->inference(inferenceFrame);
	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}

	if (tf->crop_enabled) {
		// translate the detected objects to the original frame
		for (Object &obj : objects) {
			obj.rect.x += (float)cropRect.x;
			obj.rect.y += (float)cropRect.y;
		}
	}

	// update the detected object text input
	if (objects.size() > 0) {
		if (tf->lastDetectedObjectId != objects[0].label) {
			tf->lastDetectedObjectId = objects[0].label;
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			if (objects[0].label >= 0 &&
			    (size_t)objects[0].label < tf->classNames.size()) {
				obs_data_set_string(
					source_settings, "detected_object",
					tf->classNames[objects[0].label].c_str());
			}
			obs_data_release(source_settings);
		}
	} else {
		if (tf->lastDetectedObjectId != -1) {
			tf->lastDetectedObjectId = -1;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object", "");
			// release the source settings
			obs_data_release(source_settings);
		}
	}

	if (tf->minAreaThreshold > 0) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.rect.area() > (float)tf->minAreaThreshold) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	if (tf->objectCategory != -1) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.label == tf->objectCategory) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	// Always run SORT tracker for stable effects
	objects = tf->tracker.update(objects);

	// Remove objects not currently visible
	objects.erase(
		std::remove_if(objects.begin(), objects.end(),
			       [](const Object &obj) { return obj.unseenFrames > 0; }),
		objects.end());

	if (!tf->saveDetectionsPath.empty()) {
		// Throttle: write at most once per second to avoid disk thrashing
		tf->detectionWriteAccum += tf->totalSeconds - tf->lastDetectionWriteTime;
		if (tf->detectionWriteAccum >= 1.0f) {
			tf->detectionWriteAccum = 0.0f;
			tf->lastDetectionWriteTime = tf->totalSeconds;
			std::ofstream detectionsFile(tf->saveDetectionsPath);
			if (detectionsFile.is_open()) {
				nlohmann::json j;
				for (const Object &obj : objects) {
					nlohmann::json obj_json;
					obj_json["label"] = obj.label;
					obj_json["confidence"] = obj.prob;
					obj_json["rect"] = {{"x", obj.rect.x},
							    {"y", obj.rect.y},
							    {"width", obj.rect.width},
							    {"height", obj.rect.height}};
					obj_json["id"] = obj.id;
					j.push_back(obj_json);
				}
				detectionsFile << j.dump(4);
			} else {
				obs_log(LOG_ERROR,
					"Failed to open file for writing detections: %s",
					tf->saveDetectionsPath.c_str());
			}
		}
	}

	if (tf->maskingEnabled) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		cv::Mat mask = generateMask(tf, frame, objects);
		{
			std::lock_guard<std::mutex> lock(tf->outputLock);
			mask.copyTo(tf->outputMask);
			tf->hasStamp = false;
		}

		// Image stamp: composite PNG onto frame on CPU
		cv::Mat stampSnapshot;
		{
			std::lock_guard<std::mutex> slock(tf->stampLock);
			stampSnapshot = tf->stampBGRA;
		}
		if (tf->maskingType == "stamp" && !stampSnapshot.empty()) {
			cv::Mat frameBGRA = compositeStamps(tf, frame, objects, stampSnapshot);
			std::lock_guard<std::mutex> lock(tf->outputLock);
			tf->outputStampBGRA = std::move(frameBGRA);
			tf->hasStamp = true;
			return;
		}
	}

	if (tf->trackingEnabled && tf->trackingFilter) {
		const int width = imageBGRA.cols;
		const int height = imageBGRA.rows;

		cv::Rect2f boundingBox = cv::Rect2f(0, 0, (float)width, (float)height);
		// get location of the objects
		if (tf->zoomObject == "single") {
			if (objects.size() > 0) {
				// find first visible object
				for (const Object &obj : objects) {
					if (obj.unseenFrames == 0) {
						boundingBox = obj.rect;
						break;
					}
				}
			}
		} else if (tf->zoomObject == "biggest") {
			// get the bounding box of the biggest object
			if (objects.size() > 0) {
				float maxArea = 0;
				for (const Object &obj : objects) {
					const float area = obj.rect.width * obj.rect.height;
					if (area > maxArea) {
						maxArea = area;
						boundingBox = obj.rect;
					}
				}
			}
		} else if (tf->zoomObject == "oldest") {
			// get the object with the oldest id that's visible currently
			if (objects.size() > 0) {
				uint64_t oldestId = UINT64_MAX;
				for (const Object &obj : objects) {
					if (obj.unseenFrames == 0 && obj.id < oldestId) {
						oldestId = obj.id;
						boundingBox = obj.rect;
					}
				}
			}
		} else {
			// get the bounding box of all objects
			if (objects.size() > 0) {
				boundingBox = objects[0].rect;
				for (const Object &obj : objects) {
					if (obj.unseenFrames > 0) {
						continue;
					}
					boundingBox |= obj.rect;
				}
			}
		}
		bool lostTracking = objects.size() == 0;
		// the zooming box should maintain the aspect ratio of the image
		// with the tf->zoomFactor controlling the effective buffer around the bounding box
		// the bounding box is the center of the zooming box
		float frameAspectRatio = (float)width / (float)height;
		// calculate an aspect ratio box around the object using its height
		float boxHeight = boundingBox.height;
		// calculate the zooming box size
		float dh = (float)height - boxHeight;
		float buffer = dh * (1.0f - tf->zoomFactor);
		float zh = boxHeight + buffer;
		float zw = zh * frameAspectRatio;
		// calculate the top left corner of the zooming box
		float zx = boundingBox.x - (zw - boundingBox.width) / 2.0f;
		float zy = boundingBox.y - (zh - boundingBox.height) / 2.0f;

		if (tf->trackingRect.width == 0) {
			// initialize the trackingRect
			tf->trackingRect = cv::Rect2f(zx, zy, zw, zh);
		} else {
			// interpolate the zooming box to tf->trackingRect
			float factor = tf->zoomSpeedFactor * (lostTracking ? 0.2f : 1.0f);
			tf->trackingRect.x =
				tf->trackingRect.x + factor * (zx - tf->trackingRect.x);
			tf->trackingRect.y =
				tf->trackingRect.y + factor * (zy - tf->trackingRect.y);
			tf->trackingRect.width =
				tf->trackingRect.width + factor * (zw - tf->trackingRect.width);
			tf->trackingRect.height =
				tf->trackingRect.height + factor * (zh - tf->trackingRect.height);
		}

		// get the settings of the crop/pad filter
		obs_data_t *crop_pad_settings = obs_source_get_settings(tf->trackingFilter);
		obs_data_set_int(crop_pad_settings, "left", (int)tf->trackingRect.x);
		obs_data_set_int(crop_pad_settings, "top", (int)tf->trackingRect.y);
		// right = image width - (zx + zw)
		obs_data_set_int(
			crop_pad_settings, "right",
			(int)((float)width - (tf->trackingRect.x + tf->trackingRect.width)));
		// bottom = image height - (zy + zh)
		obs_data_set_int(
			crop_pad_settings, "bottom",
			(int)((float)height - (tf->trackingRect.y + tf->trackingRect.height)));
		// apply the settings
		obs_source_update(tf->trackingFilter, crop_pad_settings);
		obs_data_release(crop_pad_settings);
	}
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	if (tf->maskingEnabled) {
		cv::Mat outputBGRA, outputMask;
		bool hasStamp = false;
		{
			std::lock_guard<std::mutex> lock(tf->outputLock);
			hasStamp = tf->hasStamp;
			if (hasStamp) {
				if (tf->outputStampBGRA.empty() ||
				    (uint32_t)tf->outputStampBGRA.cols != width ||
				    (uint32_t)tf->outputStampBGRA.rows != height) {
					obs_source_skip_video_filter(tf->source);
					return;
				}
				outputBGRA = tf->outputStampBGRA.clone();
			}
			outputMask = tf->outputMask.clone();
		}

		// Stamp: already composited in video_tick, just render
		if (hasStamp) {
			gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
							      (const uint8_t **)&outputBGRA.data, 0);
			if (tex) {
				gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
				gs_effect_set_texture(imageParam, tex);
				while (gs_effect_loop(tf->maskingEffect, "Draw"))
					gs_draw_sprite(tex, 0, width, height);
				gs_texture_destroy(tex);
			}
			return;
		}

		// Apply GPU masking effects — copy source texture directly (no CPU roundtrip)
		gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
		gs_copy_texture(tex, gs_texrender_get_texture(tf->texrender));
		gs_texture_t *maskTexture = nullptr;
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

		maskTexture = gs_texture_create(width, height, GS_R8, 1,
						(const uint8_t **)&outputMask.data, 0);
		gs_effect_set_texture(maskParam, maskTexture);
		if (tf->maskingType == "output_mask") {
			technique_name = "DrawMask";
		} else if (tf->maskingType == "blur") {
			gs_texture_destroy(tex);
			tex = blur_image(tf, width, height, maskTexture);
		} else if (tf->maskingType == "frosted_glass") {
			gs_texture_destroy(tex);
			tex = blur_image(tf, width, height, maskTexture);
		} else if (tf->maskingType == "pixelate") {
			gs_texture_destroy(tex);
			tex = pixelate_image(tf, width, height, maskTexture,
					     (float)tf->maskingBlurRadius);
		} else if (tf->maskingType == "transparent") {
			technique_name = "DrawSolidColor";
			gs_effect_set_color(maskColorParam, 0);
		} else if (tf->maskingType == "solid_color" ||
			   tf->maskingType == "eye_bar") {
			technique_name = "DrawSolidColor";
			gs_effect_set_color(maskColorParam, tf->maskingColor);
		} else if (tf->maskingType == "stamp") {
			technique_name = "Draw";
		}

		// Frosted glass and pattern types use their own effects for final composite
		if (tf->maskingType == "frosted_glass" &&
		    tf->frostedGlassEffect) {
			gs_eparam_t *fgImage =
				gs_effect_get_param_by_name(tf->frostedGlassEffect, "image");
			gs_eparam_t *fgMask =
				gs_effect_get_param_by_name(tf->frostedGlassEffect, "focalmask");
			gs_eparam_t *fgAlpha = gs_effect_get_param_by_name(
				tf->frostedGlassEffect, "overlay_alpha");
			gs_eparam_t *fgTintColor = gs_effect_get_param_by_name(
				tf->frostedGlassEffect, "tint_color");
			gs_effect_set_texture(fgImage, tex);
			gs_effect_set_texture(fgMask, maskTexture);
			gs_effect_set_float(fgAlpha, tf->maskingOverlayAlpha);
			gs_effect_set_color(fgTintColor, tf->maskingColor);
			while (gs_effect_loop(tf->frostedGlassEffect, "Draw")) {
				gs_draw_sprite(tex, 0, 0, 0);
			}
		} else if ((tf->maskingType == "checker_pattern" ||
			    tf->maskingType == "stripe_pattern" ||
			    tf->maskingType == "hstripe_pattern" ||
			    tf->maskingType == "dot_pattern") &&
			   tf->patternEffect) {
			gs_eparam_t *patImage =
				gs_effect_get_param_by_name(tf->patternEffect, "image");
			gs_eparam_t *patMask =
				gs_effect_get_param_by_name(tf->patternEffect, "focalmask");
			gs_eparam_t *patColor =
				gs_effect_get_param_by_name(tf->patternEffect, "color");
			gs_eparam_t *patColor2 =
				gs_effect_get_param_by_name(tf->patternEffect, "color2");
			gs_eparam_t *patTexSize =
				gs_effect_get_param_by_name(tf->patternEffect, "tex_size");
			gs_eparam_t *patSize =
				gs_effect_get_param_by_name(tf->patternEffect, "pattern_size");
			gs_effect_set_texture(patImage, tex);
			gs_effect_set_texture(patMask, maskTexture);
			gs_effect_set_color(patColor, tf->maskingColor);
			gs_effect_set_color(patColor2, tf->maskingColor2);
			vec2 texsize_vec;
			vec2_set(&texsize_vec, (float)width, (float)height);
			gs_effect_set_vec2(patTexSize, &texsize_vec);
			gs_effect_set_float(patSize, (float)tf->maskingPatternSize);
			const char *pat_technique = "DrawChecker";
			if (tf->maskingType == "stripe_pattern")
				pat_technique = "DrawStripe";
			else if (tf->maskingType == "hstripe_pattern")
				pat_technique = "DrawHStripe";
			else if (tf->maskingType == "dot_pattern")
				pat_technique = "DrawDot";
			while (gs_effect_loop(tf->patternEffect, pat_technique)) {
				gs_draw_sprite(tex, 0, 0, 0);
			}
		} else if (tf->maskingType == "glitch" &&
			   tf->glitchEffect) {
			gs_eparam_t *glImage =
				gs_effect_get_param_by_name(tf->glitchEffect, "image");
			gs_eparam_t *glMask =
				gs_effect_get_param_by_name(tf->glitchEffect, "focalmask");
			gs_eparam_t *glTime =
				gs_effect_get_param_by_name(tf->glitchEffect, "time");
			gs_eparam_t *glIntensity =
				gs_effect_get_param_by_name(tf->glitchEffect, "intensity");
			gs_eparam_t *glTexSize =
				gs_effect_get_param_by_name(tf->glitchEffect, "tex_size");
			gs_effect_set_texture(glImage, tex);
			gs_effect_set_texture(glMask, maskTexture);
			gs_effect_set_float(glTime, tf->totalSeconds);
			gs_effect_set_float(glIntensity, tf->glitchIntensity);
			vec2 texsize_gl;
			vec2_set(&texsize_gl, (float)width, (float)height);
			gs_effect_set_vec2(glTexSize, &texsize_gl);
			while (gs_effect_loop(tf->glitchEffect, "Draw")) {
				gs_draw_sprite(tex, 0, 0, 0);
			}
		} else if ((tf->maskingType == "grayscale" || tf->maskingType == "thermal" ||
			    tf->maskingType == "sepia" || tf->maskingType == "negative") &&
			   tf->colorGradeEffect) {
			gs_eparam_t *cgImage =
				gs_effect_get_param_by_name(tf->colorGradeEffect, "image");
			gs_eparam_t *cgMask =
				gs_effect_get_param_by_name(tf->colorGradeEffect, "focalmask");
			gs_effect_set_texture(cgImage, tex);
			gs_effect_set_texture(cgMask, maskTexture);
			const char *cg_technique = "DrawGrayscale";
			if (tf->maskingType == "thermal")
				cg_technique = "DrawThermal";
			else if (tf->maskingType == "sepia")
				cg_technique = "DrawSepia";
			else if (tf->maskingType == "negative")
				cg_technique = "DrawNegative";
			while (gs_effect_loop(tf->colorGradeEffect, cg_technique)) {
				gs_draw_sprite(tex, 0, 0, 0);
			}
		} else if (tf->maskingType == "obs_source" &&
			   !tf->overlaySourceName.empty() && tf->overlayEffect) {
			obs_source_t *overlaySource =
				obs_get_source_by_name(tf->overlaySourceName.c_str());
			if (overlaySource) {
				uint32_t ow = obs_source_get_width(overlaySource);
				uint32_t oh = obs_source_get_height(overlaySource);
				if (!tf->overlayTexrender) {
					tf->overlayTexrender =
						gs_texrender_create(GS_BGRA, GS_ZS_NONE);
				}
				bool rendered = false;
				gs_texrender_reset(tf->overlayTexrender);
				// Render overlay source at video frame size to avoid any
				// size-mismatch copies
				if (ow > 0 && oh > 0 &&
				    gs_texrender_begin(tf->overlayTexrender, width, height)) {
					struct vec4 bg;
					vec4_zero(&bg);
					gs_clear(GS_CLEAR_COLOR, &bg, 0.0f, 0);
					gs_ortho(0.0f, (float)ow, 0.0f, (float)oh, -100.0f,
						 100.0f);
					obs_source_video_render(overlaySource);
					gs_texrender_end(tf->overlayTexrender);
					rendered = true;
				}
				obs_source_release(overlaySource);

				if (rendered) {
					gs_texture_t *overlayTex =
						gs_texrender_get_texture(tf->overlayTexrender);
					gs_eparam_t *ovImg = gs_effect_get_param_by_name(
						tf->overlayEffect, "image");
					gs_eparam_t *ovMask = gs_effect_get_param_by_name(
						tf->overlayEffect, "focalmask");
					gs_eparam_t *ovOverlay = gs_effect_get_param_by_name(
						tf->overlayEffect, "overlay");
					gs_effect_set_texture(ovImg, tex);
					gs_effect_set_texture(ovMask, maskTexture);
					gs_effect_set_texture(ovOverlay, overlayTex);
					while (gs_effect_loop(tf->overlayEffect, "Draw")) {
						gs_draw_sprite(tex, 0, 0, 0);
					}
				} else {
					gs_effect_set_texture(imageParam, tex);
					while (gs_effect_loop(tf->maskingEffect, "Draw")) {
						gs_draw_sprite(tex, 0, 0, 0);
					}
				}
			} else {
				gs_effect_set_texture(imageParam, tex);
				while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
					gs_draw_sprite(tex, 0, 0, 0);
				}
			}
		} else {
			gs_effect_set_texture(imageParam, tex);
			while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
				gs_draw_sprite(tex, 0, 0, 0);
			}
		}

		gs_texture_destroy(tex);
		gs_texture_destroy(maskTexture);
	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
