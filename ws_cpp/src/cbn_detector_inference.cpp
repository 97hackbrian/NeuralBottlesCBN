#include "cbn_detector_inference.hpp"
#include <iostream>

CBNDetectorInference::CBNDetectorInference() : input_width_(640), input_height_(640) {}

CBNDetectorInference::~CBNDetectorInference() {}

bool CBNDetectorInference::init(const std::string& model_path, const std::string& labels_path, int input_w, int input_h) {
    input_width_ = input_w;
    input_height_ = input_h;

    try {
        model_ = core_.read_model(model_path);
        std::cout << "[INFO] Modelo cargado exitosamente desde " << model_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Fallo al cargar modelo: " << e.what() << "\n";
        return false;
    }

    std::cout << "[DEBUG] Configurando PrePostProcessor...\n";
    try {
        ov::preprocess::PrePostProcessor ppp(model_);
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
        model_ = ppp.build();
        std::cout << "[DEBUG] PrePostProcessor configurado correctamente.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Fallo en PrePostProcessor: " << e.what() << "\n";
    }

    std::cout << "[DEBUG] Cargando labels...\n";
    // Cargar Labels
    class_names_ = {"cap", "empty_slot"}; // Default en caso de fallo
    try {
        cv::FileStorage fs(labels_path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            cv::FileNode names_node = fs["names"];
            if (names_node.type() == cv::FileNode::SEQ) {
                class_names_.clear();
                for (auto it = names_node.begin(); it != names_node.end(); ++it) {
                    class_names_.push_back((std::string)*it);
                }
            } else if (names_node.type() == cv::FileNode::MAP) {
                class_names_.clear();
                for (auto it = names_node.begin(); it != names_node.end(); ++it) {
                    class_names_.push_back((std::string)*it);
                }
            }
            fs.release();
        } else {
            std::cerr << "[WARN] No se pudo abrir " << labels_path << ", usando labels por defecto.\n";
        }
    } catch (...) {
        std::cerr << "[WARN] Exception leyendo labels.yaml.\n";
    }

    std::cout << "[DEBUG] Compilando modelo...\n";
    try {
        compiled_model_ = core_.compile_model(model_, "CPU");
        infer_request_ = compiled_model_.create_infer_request();
        std::cout << "[DEBUG] Modelo compilado exitosamente.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Fallo al compilar el modelo o crear la request: " << e.what() << "\n";
        return false;
    }

    return true;
}

std::vector<Detection> CBNDetectorInference::infer(const cv::Mat& image, float conf_thresh, float nms_thresh) {
    std::cout << "[DEBUG] infer(): Inicio.\n";
    if (image.empty()) return {};

    // 1. Preparar el input tensor
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(input_width_, input_height_));
    std::cout << "[DEBUG] infer(): Imagen redimensionada.\n";

    // Llenar el tensor de OpenVINO con los datos de la imagen BGR usando zero-copy
    ov::Tensor input_tensor(ov::element::u8, {1, (size_t)input_height_, (size_t)input_width_, 3}, resized_img.data);
    infer_request_.set_input_tensor(input_tensor);
    std::cout << "[DEBUG] infer(): Tensor de entrada configurado.\n";

    // 2. Ejecutar inferencia
    std::cout << "[DEBUG] infer(): Ejecutando inferencia OpenVINO...\n";
    try {
        infer_request_.infer();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception en infer_request_.infer(): " << e.what() << "\n";
    }
    std::cout << "[DEBUG] infer(): Inferencia OpenVINO terminada.\n";

    // 3. Procesar salida
    ov::Tensor output_tensor = infer_request_.get_output_tensor();
    std::cout << "[DEBUG] infer(): Tensor de salida obtenido.\n";
    const float* output_data = output_tensor.data<const float>();
    ov::Shape output_shape = output_tensor.get_shape();

    // YOLO26 output can be [1, channels, 8400] OR [1, 300, 6]
    if (output_shape.size() != 3) {
        std::cerr << "[ERROR] Output shape no tiene 3 dimensiones. Size: " << output_shape.size() << "\n";
        return {};
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float rx = (float)image.cols / (float)input_width_;
    float ry = (float)image.rows / (float)input_height_;

    if (output_shape[1] == 300 && output_shape[2] == 6) {
        // Formato con NMS embebido (YOLO26 / export nms=True): [1, 300, 6]
        // 6 campos: [xmin, ymin, xmax, ymax, conf, class_id]
        std::cout << "[DEBUG] infer(): Parseando formato [1, 300, 6]\n";
        for (size_t i = 0; i < 300; ++i) {
            float xmin = output_data[i * 6 + 0];
            float ymin = output_data[i * 6 + 1];
            float xmax = output_data[i * 6 + 2];
            float ymax = output_data[i * 6 + 3];
            float conf = output_data[i * 6 + 4];
            int class_id = static_cast<int>(output_data[i * 6 + 5]);

            if (conf > conf_thresh) {
                int left = static_cast<int>(xmin * rx);
                int top = static_cast<int>(ymin * ry);
                int width = static_cast<int>((xmax - xmin) * rx);
                int height = static_cast<int>((ymax - ymin) * ry);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(conf);
                class_ids.push_back(class_id);
            }
        }
    } else {
        // Formato estándar YOLO26: [1, 4 + num_clases, 8400]
        size_t num_channels = output_shape[1];
        size_t num_anchors = output_shape[2];
        size_t num_classes = num_channels - 4;
        std::cout << "[DEBUG] infer(): Parseando formato estándar. classes=" << num_classes << ", anchors=" << num_anchors << "\n";

        for (size_t i = 0; i < num_anchors; ++i) {
            float max_conf = 0.0f;
            int class_id = -1;

            for (size_t c = 0; c < num_classes; ++c) {
                float conf = output_data[(4 + c) * num_anchors + i];
                if (conf > max_conf) {
                    max_conf = conf;
                    class_id = (int)c;
                }
            }

            if (max_conf > conf_thresh) {
                float xc = output_data[0 * num_anchors + i];
                float yc = output_data[1 * num_anchors + i];
                float w  = output_data[2 * num_anchors + i];
                float h  = output_data[3 * num_anchors + i];

                int left = static_cast<int>((xc - w / 2) * rx);
                int top = static_cast<int>((yc - h / 2) * ry);
                int width = static_cast<int>(w * rx);
                int height = static_cast<int>(h * ry);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(max_conf);
                class_ids.push_back(class_id);
            }
        }
    }

    std::cout << "[DEBUG] infer(): Anchors filtrados. Boxes=" << boxes.size() << "\n";

    std::vector<Detection> results;

    if (output_shape[1] == 300 && output_shape[2] == 6) {
        // YOLO26 ya tiene NMS embebido, así que devolvemos las cajas directamente
        std::cout << "[DEBUG] infer(): Omitiendo NMS extra para formato YOLO26.\n";
        for (size_t i = 0; i < boxes.size(); ++i) {
            Detection d;
            d.box = boxes[i];
            d.confidence = confidences[i];
            d.class_id = class_ids[i];
            if (d.class_id >= 0 && d.class_id < class_names_.size()) {
                d.class_name = class_names_[d.class_id];
            } else {
                d.class_name = "unknown";
            }
            results.push_back(d);
        }
    } else {
        // 4. Non-Maximum Suppression (NMS) para YOLO estándar
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, nms_thresh, indices);
        std::cout << "[DEBUG] infer(): NMS aplicado. Resultantes=" << indices.size() << "\n";

        for (int idx : indices) {
            Detection d;
            d.box = boxes[idx];
            d.confidence = confidences[idx];
            d.class_id = class_ids[idx];
            if (d.class_id >= 0 && d.class_id < class_names_.size()) {
                d.class_name = class_names_[d.class_id];
            } else {
                d.class_name = "unknown";
            }
            results.push_back(d);
        }
    }

    std::cout << "[DEBUG] infer(): Fin.\n";
    return results;
}
