#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>

// Estructura para almacenar una detección
struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
    std::string class_name;
};

// Clase para realizar la inferencia usando OpenVINO
class CBNDetectorInference {
public:
    CBNDetectorInference();
    ~CBNDetectorInference();

    // Inicializa OpenVINO y carga el modelo y las etiquetas
    bool init(const std::string& model_path, const std::string& labels_path, int input_w = 640, int input_h = 640);

    // Ejecuta inferencia sobre una imagen (ya procesada por el pipeline) y retorna detecciones válidas
    std::vector<Detection> infer(const cv::Mat& processed_image, float conf_threshold = 0.5f, float nms_threshold = 0.45f);

    // Obtiene la lista de nombres de clases cargados
    const std::vector<std::string>& getClassNames() const { return class_names_; }

private:
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    std::vector<std::string> class_names_;
    int input_width_;
    int input_height_;
};
