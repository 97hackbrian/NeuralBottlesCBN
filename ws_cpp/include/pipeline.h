#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// ============================================================================
// State Representation Struct
// ============================================================================
struct AppSettings {
    int camera_width = 1280;
    int camera_height = 720;

    int roi_x = 100;
    int roi_y = 100;
    int roi_width = 400;
    int roi_height = 400;

    int resize_enable = 0;
    int resize_width = 640;
    int resize_height = 480;

    int grayscale_enable = 0;

    int blur_enable = 0;
    int blur_kernel_size = 3;

    int clahe_enable = 0;
    int clahe_clip_limit = 40;

    int hist_eq_enable = 0;

    int burst_recording_enable = 0;
};

// Global application settings state
extern AppSettings g_settings;

// ============================================================================
// Persistence Functions (YAML with Native OpenCV cv::FileStorage)
// ============================================================================
void guardarPresets(const std::string& filename);
bool cargarPresets(const std::string& filename);

// ============================================================================
// Core Vision Pipeline
// ============================================================================
cv::Mat processImage(const cv::Mat& input, const AppSettings& config, cv::Rect& applied_roi);

// ============================================================================
// Batch Processing
// ============================================================================
void processDataset(const AppSettings& config, int camera_id);
