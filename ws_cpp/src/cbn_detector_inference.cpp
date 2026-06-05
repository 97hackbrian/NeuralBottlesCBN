#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace {
constexpr float kConfThreshold = 0.5f; // <-- Umbral de confianza
}

int main(int argc, char** argv) {
    std::cout << "[PRODUCTION EDGE NODE] Iniciando Inferencia Headless...\n";

    // 1. Inicializar OpenVINO
    ov::Core core;
    // En producción (contenedor edge), el working_dir es /app
    std::string model_path = "models/cbn_model.xml";
    std::string labels_path = "config/labels.yaml";
    std::shared_ptr<ov::Model> model;
    
    try {
        model = core.read_model(model_path);
        std::cout << "[INFO] Modelo cargado exitosamente desde " << model_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Fallo al cargar modelo: " << e.what() << "\n";
        return 1;
    }

    // Configurar el preprocesamiento
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0f);
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    // Cargar Labels
    std::vector<std::string> class_names = {"Clase 0", "Clase 1"};
    cv::FileStorage fs(labels_path, cv::FileStorage::READ);
    if (fs.isOpened()) {
        cv::FileNode names_node = fs["names"];
        if (names_node.type() == cv::FileNode::SEQ) {
            class_names.clear();
            for (auto it = names_node.begin(); it != names_node.end(); ++it) {
                class_names.push_back((std::string)*it);
            }
        }
        fs.release();
    } else {
        std::cerr << "[WARN] No se pudo abrir " << labels_path << ", usando genéricos.\n";
    }

    // Compilar para CPU
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 2. Conectar a cámara industrial
    cv::VideoCapture capture(0, cv::CAP_ANY);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] No se pudo conectar a /dev/video0.\n";
        return 1;
    }

    std::cout << "[INFO] Entrando en loop infinito de inferencia...\n";
    cv::Mat frame;

    // Loop infinito de producción (Headless)
    while (true) {
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "[WARN] Frame vacío, reintentando...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // Pre-proceso manual básico
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 640));

        // Inyectar al tensor
        ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, resized.data);
        infer_request.set_input_tensor(input_tensor);

        // Inferencia
        auto start = std::chrono::high_resolution_clock::now();
        infer_request.infer();
        auto end = std::chrono::high_resolution_clock::now();

        // Extraer salidas (en producción se emitirán señales/GPIO según esto)
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<const float>();
        
        int num_boxes = output_tensor.get_shape()[1];
        int detecciones_validas = 0;
        
        std::map<std::string, int> detection_counts;
        for (const auto& name : class_names) {
            detection_counts[name] = 0;
        }

        for (int i = 0; i < num_boxes; i++) {
            float conf = output_data[i * 6 + 4];
            float class_id = output_data[i * 6 + 5];
            if (conf > kConfThreshold) {
                detecciones_validas++;
                int class_idx = static_cast<int>(class_id);
                std::string class_name = (class_idx >= 0 && class_idx < class_names.size()) ? class_names[class_idx] : "Obj";
                detection_counts[class_name]++;
            }
        }

        double latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Frame procesado | Latencia: " << latency_ms << " ms | Total detecciones: " << detecciones_validas << " [";
        for (const auto& pair : detection_counts) {
            if (pair.second > 0) {
                std::cout << pair.first << ": " << pair.second << " ";
            }
        }
        std::cout << "]\n";

        // Pequeña pausa para no saturar 100% CPU en iteraciones ultra rápidas si la cámara no limita FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return 0;
}
