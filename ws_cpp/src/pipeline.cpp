#include "pipeline.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

// Definitions
AppSettings g_settings;

void guardarPresets(const std::string& filename) {
    try {
        std::filesystem::path p(filename);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "[ERROR] guardarPresets: No se pudo abrir el archivo " << filename << " para escritura." << std::endl;
            return;
        }

        fs << "camera_width" << g_settings.camera_width;
        fs << "camera_height" << g_settings.camera_height;
        fs << "roi_x" << g_settings.roi_x;
        fs << "roi_y" << g_settings.roi_y;
        fs << "roi_width" << g_settings.roi_width;
        fs << "roi_height" << g_settings.roi_height;
        fs << "resize_enable" << g_settings.resize_enable;
        fs << "resize_width" << g_settings.resize_width;
        fs << "resize_height" << g_settings.resize_height;
        fs << "grayscale_enable" << g_settings.grayscale_enable;
        fs << "blur_enable" << g_settings.blur_enable;
        fs << "blur_kernel_size" << g_settings.blur_kernel_size;
        fs << "clahe_enable" << g_settings.clahe_enable;
        fs << "clahe_clip_limit" << g_settings.clahe_clip_limit;
        fs << "hist_eq_enable" << g_settings.hist_eq_enable;
        fs << "burst_recording_enable" << g_settings.burst_recording_enable;

        fs.release();
        std::cout << "[INFO] Presets guardados exitosamente en: " << filename << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Excepcion OpenCV al guardar presets: " << e.what() << std::endl;
    }
}

bool cargarPresets(const std::string& filename) {
    try {
        if (!std::filesystem::exists(filename)) {
            std::cout << "[INFO] cargarPresets: El archivo '" << filename 
                      << "' no existe. Se utilizaran los valores por defecto." << std::endl;
            return false;
        }

        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ERROR] cargarPresets: No se pudo abrir el archivo " << filename << " para lectura." << std::endl;
            return false;
        }

        fs["camera_width"] >> g_settings.camera_width;
        fs["camera_height"] >> g_settings.camera_height;
        fs["roi_x"] >> g_settings.roi_x;
        fs["roi_y"] >> g_settings.roi_y;
        fs["roi_width"] >> g_settings.roi_width;
        fs["roi_height"] >> g_settings.roi_height;
        fs["resize_enable"] >> g_settings.resize_enable;
        fs["resize_width"] >> g_settings.resize_width;
        fs["resize_height"] >> g_settings.resize_height;
        fs["grayscale_enable"] >> g_settings.grayscale_enable;
        fs["blur_enable"] >> g_settings.blur_enable;
        fs["blur_kernel_size"] >> g_settings.blur_kernel_size;
        fs["clahe_enable"] >> g_settings.clahe_enable;
        fs["clahe_clip_limit"] >> g_settings.clahe_clip_limit;
        fs["hist_eq_enable"] >> g_settings.hist_eq_enable;
        fs["burst_recording_enable"] >> g_settings.burst_recording_enable;

        fs.release();
        std::cout << "[INFO] Presets cargados exitosamente desde: " << filename << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Excepcion OpenCV al cargar presets: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat processImage(const cv::Mat& input, const AppSettings& config, cv::Rect& applied_roi) {
    if (input.empty()) return input;

    // 1. ROI es puramente visual, no recortamos la imagen.
    // Los valores en config ahora se interpretan como porcentajes (0 a 100)
    int x_pct = std::clamp(config.roi_x, 0, 100);
    int y_pct = std::clamp(config.roi_y, 0, 100);
    int w_pct = std::clamp(config.roi_width, 1, 100);
    int h_pct = std::clamp(config.roi_height, 1, 100);

    int safe_x = (x_pct * input.cols) / 100;
    int safe_y = (y_pct * input.rows) / 100;
    
    // Evitamos desbordamientos y garantizamos al menos 1 pixel
    safe_x = std::clamp(safe_x, 0, input.cols - 1);
    safe_y = std::clamp(safe_y, 0, input.rows - 1);
    
    int safe_w = (w_pct * input.cols) / 100;
    int safe_h = (h_pct * input.rows) / 100;
    
    safe_w = std::clamp(safe_w, 1, input.cols - safe_x);
    safe_h = std::clamp(safe_h, 1, input.rows - safe_y);

    applied_roi = cv::Rect(safe_x, safe_y, safe_w, safe_h);
    
    // Clonamos toda la imagen para aplicarle los filtros al 100% de los píxeles
    cv::Mat processed = input.clone();

    // 2. Resize
    if (config.resize_enable == 1) {
        int r_w = std::max(1, config.resize_width);
        int r_h = std::max(1, config.resize_height);
        cv::resize(processed, processed, cv::Size(r_w, r_h));
        
        // Escalar las coordenadas del ROI verde visual si hay resize en la imagen global
        float scale_x = (float)r_w / input.cols;
        float scale_y = (float)r_h / input.rows;
        applied_roi.x = static_cast<int>(applied_roi.x * scale_x);
        applied_roi.y = static_cast<int>(applied_roi.y * scale_y);
        applied_roi.width = static_cast<int>(applied_roi.width * scale_x);
        applied_roi.height = static_cast<int>(applied_roi.height * scale_y);
    }

    // 3. Grayscale
    if (config.grayscale_enable == 1) {
        if (processed.channels() == 3) {
            cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
        }
    }

    // 4. Blur
    if (config.blur_enable == 1) {
        int k_size = config.blur_kernel_size;
        if (k_size % 2 == 0) k_size += 1;
        cv::GaussianBlur(processed, processed, cv::Size(k_size, k_size), 0);
    }

    // 5. CLAHE / Hist
    if (config.clahe_enable == 1) {
        double clip = config.clahe_clip_limit / 10.0;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip, cv::Size(8, 8));
        
        if (processed.channels() == 3) {
            cv::Mat hsv;
            cv::cvtColor(processed, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            clahe->apply(channels[2], channels[2]);
            cv::merge(channels, hsv);
            cv::cvtColor(hsv, processed, cv::COLOR_HSV2BGR);
        } else {
            clahe->apply(processed, processed);
        }
    } else if (config.hist_eq_enable == 1) {
        if (processed.channels() == 3) {
            cv::Mat hsv;
            cv::cvtColor(processed, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            cv::equalizeHist(channels[2], channels[2]);
            cv::merge(channels, hsv);
            cv::cvtColor(hsv, processed, cv::COLOR_HSV2BGR);
        } else {
            cv::equalizeHist(processed, processed);
        }
    }

    return processed;
}

void processDataset(const AppSettings& config, int camera_id) {
    std::string input_dir = "../data/raw/cam" + std::to_string(camera_id) + "/";
    std::string output_dir = "../data/processed/cam" + std::to_string(camera_id) + "/";

    if (!std::filesystem::exists(input_dir)) {
        std::cerr << "[ERROR] La carpeta de entrada no existe: " << input_dir << std::endl;
        std::cout << "[INFO] Por favor, crea la carpeta y coloca tus imagenes crudas en: " << input_dir << std::endl;
        return;
    }
    std::filesystem::create_directories(output_dir);

    std::vector<cv::String> filenames;
    std::vector<cv::String> exts = { "*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG" };
    
    for (const auto& ext : exts) {
        std::vector<cv::String> temp;
        cv::glob(input_dir + ext, temp, false);
        filenames.insert(filenames.end(), temp.begin(), temp.end());
    }

    if (filenames.empty()) {
        std::cout << "[INFO] No se encontraron imagenes compatibles en " << input_dir << std::endl;
        return;
    }

    std::cout << "\n[INFO] Iniciando procesamiento por lotes de " << filenames.size() << " imagenes..." << std::endl;
    size_t count = 0;
    
    for (const auto& file : filenames) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "[WARNING] No se pudo leer la imagen: " << file << std::endl;
            continue;
        }

        cv::Rect applied_roi;
        cv::Mat result = processImage(img, config, applied_roi);

        std::filesystem::path p(file);
        std::string out_name = output_dir + "proc_" + p.filename().string();
        cv::imwrite(out_name, result);

        count++;
        if (count % 10 == 0 || count == filenames.size()) {
            std::cout << "[INFO] Procesadas " << count << "/" << filenames.size() << " imagenes." << std::endl;
        }
    }
    std::cout << "[INFO] Procesamiento del dataset finalizado exitosamente.\n" << std::endl;
}
