#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {
constexpr int kDefaultCameraIndex = 0;
}

int main(int argc, char** argv) {
    int cameraIndex = kDefaultCameraIndex;
    if (argc > 1) {
        try {
            cameraIndex = std::stoi(argv[1]);
        } catch (const std::exception&) {
            std::cerr << "Indice de camara invalido: " << argv[1] << '\n';
            return 1;
        }
    }

    cv::VideoCapture capture(cameraIndex, cv::CAP_ANY);
    if (!capture.isOpened()) {
        std::cerr << "No se pudo abrir la camara " << cameraIndex << '\n';
        return 1;
    }

    const bool has_display = (std::getenv("DISPLAY") != nullptr);

    cv::Mat frame;

    if (has_display) {
        std::cout << "Display detectado. Iniciando prueba de camara " << cameraIndex << " con GUI...\n";
        const char* kWindowName = "NeuralBottlesCBN Camera Test";
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowName, 1280, 720);

        std::cout << "Presiona 'q' o ESC para salir. Camara: " << cameraIndex << '\n';

        while (true) {
            if (!capture.read(frame) || frame.empty()) {
                std::cerr << "No se pudo leer un frame de la camara." << '\n';
                break;
            }

            cv::putText(frame, "Camera test - press q or ESC to exit",
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::imshow(kWindowName, frame);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        cv::destroyAllWindows();
    } else {
        std::cout << "No se detecto DISPLAY. Iniciando prueba de camara " << cameraIndex << " en modo headless (sin GUI)...\n";
        int valid_frames = 0;
        const int num_frames_to_capture = 30;

        for (int i = 0; i < num_frames_to_capture; ++i) {
            if (capture.read(frame) && !frame.empty()) {
                valid_frames++;
                // Delay para permitir a la cámara ajustar su exposición/balance de blancos
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            } else {
                std::cerr << "Advertencia: Fallo al leer el frame " << i << '\n';
            }
        }

        if (valid_frames > 0) {
            std::cout << "Exito: Se leyeron " << valid_frames << "/" << num_frames_to_capture << " frames.\n";
            std::cout << "Resolucion del ultimo frame: " << frame.cols << "x" << frame.rows << '\n';
            
            cv::putText(frame, "Headless Camera Test Successful", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            
            if (cv::imwrite("test_frame.jpg", frame)) {
                std::cout << "Se ha guardado un frame de prueba en 'build/test_frame.jpg'\n";
            } else {
                std::cerr << "No se pudo guardar la imagen de prueba.\n";
            }
        } else {
            std::cerr << "Error: No se pudo leer ningun frame valido.\n";
            return 1;
        }
    }
    return 0;
}
