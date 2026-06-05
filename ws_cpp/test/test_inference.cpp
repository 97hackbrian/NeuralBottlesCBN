#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {
constexpr int kDefaultCameraIndex = 0;
const std::string kModelPath = "../models/cbn_model.xml";
const std::string kLabelsPath = "../config/labels.yaml";
constexpr float kConfThreshold = 0.01f; // <-- Umbral de confianza
}

int main(int argc, char** argv) {
    int cameraIndex = kDefaultCameraIndex;
    if (argc > 1) {
        try {
            cameraIndex = std::stoi(argv[1]);
        } catch (const std::exception&) {
            std::cerr << "Índice de cámara inválido: " << argv[1] << '\n';
            return 1;
        }
    }

    // 1. Inicializar OpenVINO
    ov::Core core;
    std::cout << "Inicializando OpenVINO...\n";
    std::shared_ptr<ov::Model> model;
    try {
        model = core.read_model(kModelPath);
    } catch (const std::exception& e) {
        std::cerr << "Error al cargar el modelo " << kModelPath << ": " << e.what() << "\n";
        return 1;
    }

    // Configurar preprocesamiento si es necesario
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
    cv::FileStorage fs(kLabelsPath, cv::FileStorage::READ);
    if (fs.isOpened()) {
        cv::FileNode names_node = fs["names"];
        if (names_node.type() == cv::FileNode::SEQ) {
            class_names.clear();
            for (auto it = names_node.begin(); it != names_node.end(); ++it) {
                class_names.push_back((std::string)*it);
            }
        }
        fs.release();
        std::cout << "Labels cargados: " << class_names.size() << "\n";
    } else {
        std::cerr << "[WARN] No se pudo abrir " << kLabelsPath << ", usando genéricos.\n";
    }

    std::cout << "Compilando modelo para CPU...\n";
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 2. Iniciar Cámara
    cv::VideoCapture capture(cameraIndex, cv::CAP_ANY);
    if (!capture.isOpened()) {
        std::cerr << "No se pudo abrir la cámara " << cameraIndex << '\n';
        return 1;
    }

    const bool has_display = (std::getenv("DISPLAY") != nullptr);
    cv::Mat frame;

    if (has_display) {
        std::cout << "Display detectado. Iniciando inferencia en cámara " << cameraIndex << " con GUI...\n";
        const char* kWindowName = "NeuralBottlesCBN Inference Test";
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowName, 1280, 720);
        std::cout << "Presiona 'q' o ESC para salir.\n";

        while (true) {
            auto start_time = std::chrono::high_resolution_clock::now();

            if (!capture.read(frame) || frame.empty()) {
                std::cerr << "No se pudo leer un frame de la cámara.\n";
                break;
            }

            // Redimensionar para la inferencia (640x640)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(640, 640));

            // Crear tensor de entrada OpenVINO
            ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, resized.data);
            infer_request.set_input_tensor(input_tensor);

            // Inferencia sincrona
            infer_request.infer();

            // Procesar salida
            ov::Tensor output_tensor = infer_request.get_output_tensor();
            const float* output_data = output_tensor.data<const float>();
            ov::Shape output_shape = output_tensor.get_shape(); // Normalmente [1, 300, 6] para tu YOLO26

            int num_boxes = output_shape[1]; // 300
            float rx = static_cast<float>(frame.cols) / 640.0f;
            float ry = static_cast<float>(frame.rows) / 640.0f;

            for (int i = 0; i < num_boxes; i++) {
                float x_min = output_data[i * 6 + 0];
                float y_min = output_data[i * 6 + 1];
                float x_max = output_data[i * 6 + 2];
                float y_max = output_data[i * 6 + 3];
                float conf = output_data[i * 6 + 4];
                float class_id = output_data[i * 6 + 5];

                if (conf > kConfThreshold) { // Usamos la constante definida arriba
                    int left = static_cast<int>(x_min * rx);
                    int top = static_cast<int>(y_min * ry);
                    int width = static_cast<int>((x_max - x_min) * rx);
                    int height = static_cast<int>((y_max - y_min) * ry);

                    int class_idx = static_cast<int>(class_id);
                    std::string class_name = (class_idx >= 0 && class_idx < class_names.size()) ? class_names[class_idx] : "Obj";
                    
                    cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
                    std::string label = class_name + ": " + std::to_string(conf).substr(0, 4);
                    cv::putText(frame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 5), cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

            cv::imshow(kWindowName, frame);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }
        }
        cv::destroyAllWindows();
    } else {
        std::cout << "Iniciando prueba de inferencia HEADLESS...\n";
        for (int i = 0; i < 10; ++i) {
            if (capture.read(frame) && !frame.empty()) {
                cv::Mat resized;
                cv::resize(frame, resized, cv::Size(640, 640));
                ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, resized.data);
                infer_request.set_input_tensor(input_tensor);
                
                auto start = std::chrono::high_resolution_clock::now();
                infer_request.infer();
                auto end = std::chrono::high_resolution_clock::now();
                
                std::cout << "Inferencia " << i+1 << " completada en " 
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                          << " ms.\n";
            }
        }
        std::cout << "Prueba headless exitosa.\n";
    }
    return 0;
}
