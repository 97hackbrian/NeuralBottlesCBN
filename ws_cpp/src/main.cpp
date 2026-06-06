#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include "pipeline.h"
#include "cbn_detector_inference.hpp"
#include "cbn_detector.hpp"

// Helper para argumentos CLI
bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    std::cout << "[INFO] Iniciando Inferencia CBN Node...\n";
    bool show_ui = hasFlag(argc, argv, "--show");

    // 1. Cargar configuraciones del pipeline (procesamiento de imagen)
    std::string config_cam_path = getWorkspacePath("../config/config_cam0.yaml");
    if (!cargarPresets(config_cam_path)) {
        std::cerr << "[WARN] No se pudo cargar " << config_cam_path << " usando defaults.\n";
    }

    // 2. Cargar configuraciones de inferencia YOLO26
    std::string inf_config_path = getWorkspacePath("../config/inference.yaml");
    cv::FileStorage fs(inf_config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ERROR] Fallo al cargar " << inf_config_path << "\n";
        return 1;
    }

    std::string model_path = (std::string)fs["model_path"];
    std::string labels_path = (std::string)fs["labels_path"];
    int input_w = (int)fs["input_width"];
    int input_h = (int)fs["input_height"];
    float conf_threshold = (float)fs["conf_threshold"];
    float nms_threshold = (float)fs["nms_threshold"];
    int target_total = (int)fs["target_total_slots"];
    int target_cap = (int)fs["target_cap_count"];
    fs.release();

    model_path = getWorkspacePath(model_path);
    labels_path = getWorkspacePath(labels_path);

    // 3. Inicializar Detector OpenVINO
    CBNDetectorInference detector;
    if (!detector.init(model_path, labels_path, input_w, input_h)) {
        std::cerr << "[ERROR] No se pudo inicializar el modelo OpenVINO.\n";
        return 1;
    }

    std::cout << "[DEBUG] Abriendo camara...\n";
    // 4. Inicializar Cámara
    cv::VideoCapture capture(0, cv::CAP_ANY);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] No se pudo abrir la cámara (ej: /dev/video0).\n";
        return 1;
    }
    std::cout << "[DEBUG] Camara abierta.\n";
    capture.set(cv::CAP_PROP_FRAME_WIDTH, g_settings.camera_width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, g_settings.camera_height);

    if (show_ui) {
        cv::namedWindow("CBN Inference Node", cv::WINDOW_NORMAL);
    }

    cv::Mat frame;
    cv::Rect applied_roi;
    std::cout << "[DEBUG] Entrando al loop...\n";

    while (true) {
        if (!capture.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        std::cout << "[DEBUG] Frame capturado. Procesando...\n";

        // Aplicar pipeline de procesamiento visual (crop, blur, eq, etc.)
        cv::Mat processed = processImage(frame, g_settings, applied_roi);

        std::cout << "[DEBUG] Frame procesado. Iniciando inferencia...\n";
        // Inferencia usando YOLO26
        auto detections = detector.infer(processed, conf_threshold, nms_threshold);

        std::cout << "[DEBUG] Inferencia terminada. Detecciones: " << detections.size() << "\n";
        // Validar lógica de negocio de los casilleros
        auto resultado = CBNDetector::validarCasillero(detections, target_total, target_cap);

        if (show_ui) {
            // Si la imagen es en escala de grises, la convertimos a BGR para ver colores en cajas y texto
            cv::Mat display;
            if (processed.channels() == 1) {
                cv::cvtColor(processed, display, cv::COLOR_GRAY2BGR);
            } else {
                display = processed.clone();
            }

            for (const auto& d : detections) {
                cv::Scalar color = (d.class_id == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::rectangle(display, d.box, color, 2);
                cv::putText(display, d.class_name + " " + std::to_string(d.confidence).substr(0,4), 
                            cv::Point(d.box.x, d.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }

            // Texto de veredicto
            if (resultado.has_value()) {
                std::string text_veredicto;
                cv::Scalar color_veredicto;
                if (resultado.value()) {
                    text_veredicto = "PASA (OK)";
                    color_veredicto = cv::Scalar(0, 255, 0); // Verde
                } else {
                    text_veredicto = "RECHAZADO";
                    color_veredicto = cv::Scalar(0, 0, 255); // Rojo
                }
                cv::putText(display, text_veredicto, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, color_veredicto, 3);
            }
            cv::imshow("CBN Inference Node", display);

            if (cv::waitKey(1) == 27) { // ESC para salir
                break;
            }
        } else {
            // Modo consola (Headless)
            if (resultado.has_value()) {
                if (resultado.value()) {
                    std::cout << "[VEREDICTO] PASA - 12 botellas correctas\n";
                } else {
                    std::cout << "[VEREDICTO] RECHAZADO - Error en el casillero\n";
                }
                // Prevenir spam de log en la terminal si se queda la caja estacionada (opcional)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }

    return 0;
}
